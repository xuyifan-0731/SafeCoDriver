# 方法设计

## 1. 问题形式化

设当前时刻为 `t`，自车为 `e`，协同车辆集合为 `V={1..N}`。车辆 `j` 发送的协同感知消息记为：

```text
M_j,t = {
  source_id,
  timestamp,
  pose_j,
  object_list_j,
  local_confidence_j,
  optional_self_diagnosis
}
```

每个目标观测包含：

```text
o_j,k = {
  object_id or track_id,
  class,
  position,
  velocity,
  size,
  heading,
  detection_confidence,
  covariance
}
```

当前要解决的问题不是直接判断 `M_j,t` 是否正常，而是输出：

```text
A_j,t = {
  vehicle_trust R_j,t,          # 长期车辆可信度
  message_usability U_j,t,      # 当前消息可用性
  correctable_score C_j,t,      # 当前异常是否可校正
  offset_estimate Delta_j,t,    # 可用于校正的 SE(2) 偏移
  calibrated_message M'_j,t,
  recommended_action,           # accept / correct / downweight / quarantine
  evidence_chain
}
```

其中 `Delta_j,t = (dx, dy, dtheta)`，表示从车辆 `j` 的上报坐标到自车融合坐标的残余空间校正量。

## 2. 统一异常建模

异常信息统一看作“观测与可验证证据之间的偏差”。不同来源对应不同偏差形态：

| 异常来源 | 典型表现 | 可校正性 |
|---|---|---|
| 空间标定失配 | 多个目标存在稳定平移/旋转偏移 | 高，可估计 `dx,dy,dtheta` |
| 时间同步误差 | 位置残差与速度方向相关，随目标速度增大 | 中，可估计时间偏移或扩大协方差 |
| 传感器故障 | 缺失、置信度异常、噪声突然变大 | 低到中，可降权或短时隔离 |
| 恶意篡改 | 目标伪造、轨迹跳变、与多车共识冲突 | 低，通常隔离 |
| 固有测量噪声 | 小范围随机残差 | 高，通过协方差和鲁棒融合吸收 |

因此评估不是 `normal/anomaly`，而是四个连续量：

```text
spatial_consistency      S_pos
temporal_consistency     S_tmp
cross_source_consensus   S_cns
calibration_stability    S_cal
```

这些量共同决定消息可用性和可校正性。

## 3. 双层协同评估机制

### 3.1 长期车辆可信度

车辆可信度 `R_j,t in [0,1]` 描述车辆 `j` 在一段时间内作为信息源的可靠程度。建议用带遗忘因子的 Beta reputation 表示：

```text
R_j,t = alpha_j,t / (alpha_j,t + beta_j,t)

alpha_j,t = lambda * alpha_j,t-1 + good_evidence_j,t
beta_j,t  = lambda * beta_j,t-1  + bad_evidence_j,t
```

其中 `lambda` 是遗忘因子，建议初值 `0.98`。证据来源包括：

1. **历史偏移稳定性**：若 `Delta_j,t` 长期稳定且残差低，增加正证据。
2. **当前一致性**：当前消息与自车、地图、其他车辆的一致性。
3. **其他车辆回传判断**：接收其他车对 `j` 的可信度判断，但必须按证据链质量加权，而不是直接投票。
4. **安全影响惩罚**：若异常目标会显著改变下游碰撞判断或路点修正，负证据权重提高。

建议更新项：

```text
good = w1*S_pos + w2*S_tmp + w3*S_cns + w4*S_cal
bad  = w5*(1-S_pos) + w6*(1-S_tmp) + w7*(1-S_cns) + w8*risk_impact
```

### 3.2 即时消息可用性

消息可用性 `U_j,t` 描述当前帧消息是否可用于融合。它不等同于车辆可信度，因为一个长期可信车辆也可能当前传感器遮挡或标定漂移。

建议先采用规则可解释版本，后续再替换为轻量学习模型：

```text
U_j,t = sigmoid(
  b0
  + b1 * logit(R_j,t-1)
  + b2 * S_pos
  + b3 * S_tmp
  + b4 * S_cns
  + b5 * C_j,t
  - b6 * risk_impact
  - b7 * missing_rate
)
```

输出动作分四类：

| 条件 | 动作 | 含义 |
|---|---|---|
| `U >= 0.75` 且残差低 | `accept` | 直接融合 |
| `U >= 0.55` 且 `C >= 0.6` | `correct` | 使用 `Delta` 校正后融合 |
| `0.35 <= U < 0.55` | `downweight` | 降低 `Agent.confidence`，扩大安全 margin |
| `U < 0.35` 或高风险不可校正 | `quarantine` | 暂不进入融合，仅保留审计证据 |

## 4. 偏移估计与可校正性分析

### 4.1 目标关联

偏移估计首先需要在不同来源之间建立目标匹配。建议两级关联：

1. **粗匹配**：类别一致、尺寸相近、距离门限、速度方向相近。
2. **全局匹配**：用 Hungarian matching 最小化代价：

```text
cost = a1 * position_distance
     + a2 * velocity_distance
     + a3 * size_distance
     + a4 * class_mismatch
```

只保留代价低于门限的匹配对，形成 `matched_pairs`。

### 4.2 SE(2) 残余偏移估计

对车辆 `j` 的观测，估计残余变换 `Delta_j,t=(dx,dy,dtheta)`：

```text
min_Delta sum_i rho(
  || p_e,i - T(Delta) * p_j,i ||_Sigma_i
)
```

其中 `rho` 使用 Huber 或 Tukey loss，降低伪造目标和错误匹配影响。实现上先做轻量版本：

1. 若匹配数 `< 2`：只估计平移均值，不估计旋转。
2. 若匹配数 `>= 2`：用加权 Procrustes/Kabsch 求 `dx,dy,dtheta`。
3. 使用 RANSAC 重复采样，选内点最多且残差最低的解。
4. 输出 `offset_covariance`、`inlier_ratio`、`residual_mean`、`residual_p95`。

### 4.3 可校正性判定

可校正异常应满足：

```text
C_j,t 高，当且仅当：
  residual_before 大
  residual_after 小
  inlier_ratio 高
  offset 与历史 offset 连续
  offset_covariance 小
```

建议公式：

```text
C = sigmoid(
  c0
  + c1 * improvement_ratio
  + c2 * inlier_ratio
  - c3 * residual_after
  - c4 * offset_jump
  - c5 * offset_uncertainty
)
```

若 `residual_before` 很大但 `residual_after` 仍大，说明不是稳定空间失配，可能是恶意伪造、目标级错误或严重故障，应降权或隔离。

## 5. 协同证据交换机制

通信内容不传原始点云或完整检测框，只传最小充分证据：

```text
EvidenceMessage = {
  issuer_id,              # 谁发出的判断
  target_id,              # 被评价的信息源
  time_window,
  trust_posterior,        # alpha,beta 或均值+方差
  offset_mean,            # dx,dy,dtheta
  offset_covariance,
  residual_summary,       # mean,p95,inlier_ratio,match_count
  action,                 # accept/correct/downweight/quarantine
  evidence_chain_hashes,  # 证据链摘要
  signature_placeholder
}
```

接收方融合其他车辆证据时，按“证据质量”和“发布者可信度”加权：

```text
weight_i = R_issuer * evidence_quality * freshness * viewpoint_diversity
```

避免单个低可信车辆通过广播恶意评价影响全局判断。

## 6. 检测-校正闭环

每一帧闭环如下：

```text
1. 接收多源消息 M_j,t
2. 与自车和其他来源做关联，提取一致性证据
3. 估计 Delta_j,t 和可校正性 C_j,t
4. 计算消息可用性 U_j,t 和推荐动作
5. 对 correct/accept/downweight 消息生成 calibrated PerceptionResult
6. 将校正后的 perception 输入 HybridSafetyConstraint
7. 使用下游安全结果反哺 risk_impact 和 trust update
8. 生成 EvidenceMessage 回传给其他车辆
```

其中第 7 步是关键：如果某条消息导致 V1 碰撞概率、几何威胁数量、TTC 或路点修正发生大幅变化，则说明该消息对安全决策影响高。高影响消息需要更严格的一致性要求，避免异常信息直接放大为错误避障或急刹。

## 7. 与当前 SafeCoDriver 的具体衔接

当前代码已经有以下可复用能力：

| 现有能力 | 用法 |
|---|---|
| `PerceptionResult` / `Agent` | 作为校正后统一感知输出 |
| `Agent.confidence` | 承载消息可用性降权结果 |
| `Agent.source` | 标记目标来自 ego、coop 或某个 CAV |
| `HybridSafetyConstraint._get_safety_margin()` | 可扩展为基于 `confidence/usability` 的动态 margin |
| `HybridSafetyConstraint.constrain_waypoints()` | 使用校正后的 perception 评估下游安全收益 |
| DeepAccident `coop_perception` | 可用于构造多源失配和噪声注入实验 |
| SUMO attacker/coop 场景 | 可用于闭环验证异常信息对碰撞、急刹、误避障的影响 |

首版不需要修改 V1 网络结构。研究点 2.2 先作为 V1/Hybrid 前置模块落地，后续再考虑把 `usability` 作为 V1 输入特征替换当前第 10 维占位特征。
