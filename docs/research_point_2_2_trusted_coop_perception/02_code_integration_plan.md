# 代码集成计划

## 1. 推荐新增目录

建议后续实现时新增：

```text
coop_safety/trust/
  __init__.py
  interfaces.py              # 可用性、偏移、证据链数据结构
  association.py             # 多源目标关联
  offset_estimator.py        # SE(2) 偏移估计与可校正性分析
  usability.py               # 当前消息可用性评分
  trust_manager.py           # 长期车辆可信度更新
  evidence_exchange.py       # 证据消息生成、验证、融合
  perception_calibrator.py   # 主入口：Raw messages -> calibrated PerceptionResult
```

实验侧新增：

```text
experiments/run_trust_calib_deepaccident.py
experiments/run_trust_calib_sumo.py
experiments/anomaly_injection.py
```

首版建议只新增上述模块，不改动 `HybridSafetyConstraint` 主逻辑。若要把消息可用性影响几何 margin，可在第二阶段扩展。

## 2. 主入口接口

建议主类命名为 `TrustCalibLayer`：

```python
class TrustCalibLayer:
    def __init__(self, config: dict | None = None):
        ...

    def calibrate(
        self,
        ego_perception: PerceptionResult,
        coop_messages: list[CooperativeMessage],
        peer_evidence: list[EvidenceMessage] | None = None,
    ) -> CalibrationResult:
        ...
```

输出：

```python
CalibrationResult(
    perception=calibrated_perception,
    source_reports={
        "cav_1": SourceAvailabilityReport(...),
        "cav_2": SourceAvailabilityReport(...),
    },
    evidence_to_broadcast=[...],
)
```

其中 `calibrated_perception` 继续使用当前 `PerceptionResult`，保证能直接调用：

```python
calib = trust_layer.calibrate(ego_perception, coop_messages, peer_evidence)
modified_wp, stats = hybrid.constrain_waypoints(waypoints, calib.perception)
```

## 3. 数据结构设计

### 3.1 CooperativeMessage

用于包装来自其他车的感知消息：

```python
@dataclass
class CooperativeMessage:
    source_id: str
    timestamp: float
    pose_xyh: tuple[float, float, float]
    perception: PerceptionResult
    claimed_confidence: float = 1.0
    diagnostics: dict = field(default_factory=dict)
```

`perception.agents` 可以仍是发送方坐标系，也可以是已粗变换到 ego 坐标系。建议首版统一要求进入 `TrustCalibLayer` 前已经完成基础坐标转换，`offset_estimator` 只估计残余偏移。

### 3.2 OffsetEstimate

```python
@dataclass
class OffsetEstimate:
    dx: float
    dy: float
    dtheta: float
    covariance: np.ndarray
    residual_before: float
    residual_after: float
    residual_p95: float
    inlier_ratio: float
    match_count: int
    correctable_score: float
```

### 3.3 SourceAvailabilityReport

```python
@dataclass
class SourceAvailabilityReport:
    source_id: str
    vehicle_trust: float
    message_usability: float
    correctable_score: float
    recommended_action: str
    offset: OffsetEstimate | None
    evidence_chain: list[EvidenceItem]
```

### 3.4 EvidenceItem / EvidenceMessage

```python
@dataclass
class EvidenceItem:
    kind: str
    value: float | dict
    weight: float
    description: str

@dataclass
class EvidenceMessage:
    issuer_id: str
    target_id: str
    time_window: tuple[float, float]
    trust_alpha: float
    trust_beta: float
    offset: OffsetEstimate | None
    residual_summary: dict
    action: str
    evidence_hash: str
```

## 4. 模块职责

### 4.1 `association.py`

输入：自车 perception、协同车 perception。

输出：匹配对：

```text
[(ego_agent_idx, coop_agent_idx, cost), ...]
```

实现步骤：

1. 过滤距离过远或类别明显不一致的目标。
2. 构造代价矩阵。
3. 使用 `scipy.optimize.linear_sum_assignment` 做 Hungarian matching。
4. 对高代价匹配置为 unmatched。

首版若不想增加 scipy 依赖，可以实现贪心匹配：按 cost 升序依次选择未使用目标。

### 4.2 `offset_estimator.py`

输入：匹配对中的点坐标、速度和协方差。

输出：`OffsetEstimate`。

首版算法：

1. 计算校正前残差 `residual_before`。
2. 用匹配对坐标估计平移：

```text
dx = weighted_mean(x_ego - x_coop)
dy = weighted_mean(y_ego - y_coop)
```

3. 匹配数 `>= 2` 时，额外估计 `dtheta`。
4. 使用 MAD 或 p95 残差剔除离群点，重新估计。
5. 计算 `correctable_score`。

第二阶段再加入 RANSAC + Huber 优化。

### 4.3 `usability.py`

输入：车辆历史信任、当前 offset、残差、一致性、安全影响。

输出：`message_usability` 和 `recommended_action`。

建议默认阈值：

```python
ACCEPT_THR = 0.75
CORRECT_THR = 0.55
DOWNWEIGHT_THR = 0.35
CORRECTABLE_THR = 0.60
```

### 4.4 `trust_manager.py`

维护每个 source 的：

```text
alpha, beta
offset_ema
offset_var
last_seen_timestamp
recent_actions
```

接口：

```python
update(source_id, report, downstream_stats=None) -> float
get_trust(source_id) -> float
```

`downstream_stats` 可接收 Hybrid 输出的：

```text
collision_prob
n_geometric_threats
min_ttc
target_speed_factor
modification_rate
```

用于估计该消息对安全决策的影响。

### 4.5 `evidence_exchange.py`

负责：

1. 从 `SourceAvailabilityReport` 生成轻量 `EvidenceMessage`。
2. 接收其他车辆 evidence 后做时效性过滤。
3. 按 `issuer_trust * evidence_quality * freshness` 融合证据。
4. 保留证据链摘要，支持可解释输出。

### 4.6 `perception_calibrator.py`

主流程：

```text
for each coop message:
  matched = associate(ego_perception, coop_message.perception)
  offset = estimate_offset(matched)
  trust = trust_manager.get_trust(source_id)
  usability = score_message(trust, offset, matched, peer_evidence)
  report = build_report(...)
  calibrated_agents = apply_action(message.agents, report)

merge ego agents + accepted/corrected/downweighted coop agents
deduplicate by association
return CalibrationResult
```

`apply_action` 规则：

| 动作 | 处理 |
|---|---|
| `accept` | 直接加入融合候选 |
| `correct` | 对位置和速度方向应用 `Delta` 后加入 |
| `downweight` | 校正后加入，但 `Agent.confidence *= U`，并标记 `source` |
| `quarantine` | 不加入下游 perception，仅写证据 |

## 5. 与 `HybridSafetyConstraint` 的两阶段结合

### 阶段 A：零侵入接入

不改 `HybridSafetyConstraint`。实验脚本中做：

```python
calib = trust_layer.calibrate(frame.perception, coop_messages)
mw, stats = hybrid.constrain_waypoints(bw, calib.perception)
```

优点：不会影响当前最佳结果和已有基线。

### 阶段 B：让可用性影响安全 margin

在 `Agent.confidence` 表示消息可用性后，可以修改 `_get_safety_margin()`：

```text
effective_uncertainty = 1 - agent.confidence
margin = base_margin * (1 + k_uncertainty * effective_uncertainty)
```

这样低可用但未完全丢弃的信息不会被当作高确定目标处理，而是进入更保守的安全边界。

### 阶段 C：把可用性输入 V1

当前 V1 输入第 10 维为占位 `0`。后续可改为：

```text
x_i = [x, y, vx, vy, heading, length, width, speed, visible_flag, usability]
```

这需要重新训练 V1，并和原 V1 做公平对比。首版不建议直接改。

## 6. 最小可运行版本

为了快速形成结果，建议第一版只实现：

1. 双源 ego + coop 的贪心目标匹配。
2. 平移偏移估计 `dx,dy`，暂不估计旋转。
3. 规则式 `U/C/R` 评分。
4. DeepAccident 上注入平移失配、噪声、目标删除、伪造目标。
5. 输出校正前/后 WPC%、FA(f)、offset MAE、quarantine rate。

这个版本足以验证论文主张：“不是简单丢弃异常信息，而是识别可校正失配并修复，从而提升下游安全感知输入质量。”

## 7. 里程碑

| 阶段 | 目标 | 产物 |
|---|---|---|
| M1 | 离线可用性评估 | `coop_safety/trust/*` + DeepAccident 注入实验 |
| M2 | 偏移校正闭环 | 校正后 `PerceptionResult` 接入 Hybrid，报告 WPC/FA/TTC 变化 |
| M3 | 协同证据交换 | 多车 evidence 融合和证据链输出 |
| M4 | SUMO 闭环 | 异常协同消息下的 CollRate、急刹、误避障评测 |
| M5 | 学习化增强 | 用轻量 MLP 替代规则评分，V1 输入加入 usability |
