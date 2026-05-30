# 实验方案

## 1. 验证目标

研究点 2.2 需要证明四件事：

1. 能统一识别不同异常来源：攻击、故障、空间失配、噪声。
2. 能区分“不可用”和“可校正但有偏差”的信息，避免简单丢弃。
3. 校正后的协同感知能提升下游 SafeCoDriver 的安全指标。
4. 协同证据交换能提升判断准确性和可解释性。

## 2. 数据与场景

### 2.1 DeepAccident 离线评测

直接复用当前：

```bash
python experiments/run_deepaccident_unified_metrics.py
```

新增脚本建议：

```bash
python experiments/run_trust_calib_deepaccident.py \
  --anomaly spatial_shift \
  --shift-x 2.0 \
  --shift-y 1.0 \
  --out-dir results/trust_calib_deepaccident/spatial_shift
```

DeepAccident 适合验证：

| 能力 | 说明 |
|---|---|
| offset estimation | 对 `coop_perception` 注入已知偏移，计算估计误差 |
| message usability | 对正常/异常消息计算 AUC、F1、ECE |
| downstream safety | 比较校正前后 WPC%、FA(f)、Det(s)、Mod% |
| correctability | 区分稳定偏移和伪造目标 |

### 2.2 SUMO 闭环评测

复用当前：

```bash
python experiments/run_modified_sumo_comparison.py --scenario-set base --key-methods --out-dir results/modified_sumo_v7_base_key
python experiments/run_modified_sumo_comparison.py --scenario-set stress --key-methods --out-dir results/modified_sumo_v7_stress_key
```

新增脚本建议：

```bash
python experiments/run_trust_calib_sumo.py \
  --scenario-set stress \
  --anomaly fake_obstacle \
  --out-dir results/trust_calib_sumo/fake_obstacle
```

SUMO 适合验证：

| 能力 | 说明 |
|---|---|
| false brake reduction | 伪造障碍是否导致急刹/误避障 |
| closed-loop collision | 校正/隔离异常消息后 CollRate 是否下降 |
| rear-risk interaction | 异常消息是否诱发后向追尾风险 |
| evidence exchange | 多车判断是否比单车判断更稳 |

## 3. 异常注入设计

### 3.1 空间失配

对协同车消息中所有目标统一施加：

```text
x' = x + dx
y' = y + dy
heading' = heading + dtheta
```

建议网格：

| 等级 | `dx,dy` | `dtheta` |
|---|---:|---:|
| mild | 0.5m, 0.5m | 1 deg |
| medium | 2.0m, 1.0m | 3 deg |
| severe | 4.0m, 2.0m | 6 deg |

预期：TrustCalib 能估计偏移，`correct` 后恢复 WPC 和检测指标。

### 3.2 随机噪声

对每个目标独立加入：

```text
epsilon_xy ~ N(0, sigma_pos)
epsilon_v  ~ N(0, sigma_vel)
```

建议：

```text
sigma_pos = [0.2, 0.5, 1.0, 2.0] m
sigma_vel = [0.2, 0.5, 1.0] m/s
```

预期：轻度噪声应 `accept/downweight`，重度噪声应降低 usability 并扩大安全 margin。

### 3.3 目标删除和传感器故障

随机删除协同消息中的一部分目标：

```text
drop_rate = [0.1, 0.3, 0.5, 0.7]
```

或仅删除高风险目标：

```text
drop only agents with TTC < 4s
```

预期：`missing_rate` 上升，消息可用性下降；若 ego 或其他车辆能补充证据，下游仍保持安全。

### 3.4 伪造目标攻击

注入不存在的障碍：

```text
fake target at ego planned waypoint t=1..3s
```

预期：若伪造目标缺少其他来源支持且与历史轨迹不连续，应被 `quarantine` 或 `downweight`，降低误刹和误避障。

### 3.5 轨迹篡改攻击

对真实目标速度或位置做非物理跳变：

```text
velocity scale = 2.0 or reverse direction
position jump = 5m within one frame
```

预期：时间一致性 `S_tmp` 显著下降，车辆可信度被扣减。

## 4. 评价指标

### 4.1 可用性评估指标

| 指标 | 含义 |
|---|---|
| `Usability-AUC` | 区分正常/异常消息的能力 |
| `Correctability-F1` | 判断异常是否可校正的准确率 |
| `Action-Acc` | accept/correct/downweight/quarantine 分类准确率 |
| `Trust-ECE` | 车辆可信度校准误差 |
| `Evidence-Len` | 平均证据链长度，越短越适合通信 |

### 4.2 偏移估计指标

| 指标 | 含义 |
|---|---|
| `Offset-MAE-xy` | `dx,dy` 平均绝对误差 |
| `Offset-MAE-theta` | `dtheta` 平均绝对误差 |
| `Residual-Before/After` | 校正前后匹配残差 |
| `Inlier-Ratio` | 偏移估计内点比例 |
| `Correction-Gain` | `(before-after)/before` |

### 4.3 下游安全指标

复用当前统一指标：

| 指标 | 来源 |
|---|---|
| `WPC%` | DeepAccident 路点碰撞率 |
| `FA(f)` / `FA(s)` | 帧级/场景级误报 |
| `Det(f)` / `Det(s)` | 帧级/场景级检出 |
| `Mod%` | 路点修改比例 |
| `CollRate` | SUMO 闭环碰撞率 |
| `2ndColl` | SUMO 二次碰撞 |
| `Sev` | SUMO 碰撞严重度 |
| `HardBrakeRate` | 新增，强刹比例 |
| `FalseAvoidRate` | 新增，因伪造目标导致的无必要避让 |

## 5. 对比方法

建议最少包含：

| 方法 | 说明 |
|---|---|
| `Raw-Coop` | 不做可信评估，直接融合协同消息 |
| `Ego-Only` | 丢弃所有协同信息 |
| `Hard-Filter` | 检测异常后直接丢弃 |
| `Trust-Only` | 只用长期车辆可信度，不估计当前偏移 |
| `Calib-Only` | 只做偏移校正，不做长期 trust |
| `TrustCalib` | 完整方法 |
| `TrustCalib+Evidence` | 加入协同证据交换 |
| `Oracle-Calib` | 使用真实注入偏移做上界 |

关键要证明：`TrustCalib` 优于 `Hard-Filter`，因为它能保留“有偏但可校正”的有价值信息。

## 6. 消融实验

| 消融 | 目的 |
|---|---|
| 去掉历史 trust | 验证长期可信度的贡献 |
| 去掉 offset correction | 验证校正优于直接降权 |
| 去掉 evidence exchange | 验证协同判断价值 |
| 去掉 safety impact | 验证高安全影响消息需要更严格处理 |
| 固定阈值 vs 自适应阈值 | 验证可用性连续评分的价值 |
| 不同异常强度 | 验证鲁棒性边界 |

## 7. 结果表建议

### 7.1 DeepAccident 主表

```text
method, anomaly, severity, WPC%, FA(f), Det(s), Mod%, Offset-MAE, Correctability-F1
```

### 7.2 SUMO 主表

```text
method, scenario_set, anomaly, CollRate, 2ndColl, Sev, HardBrakeRate, FalseAvoidRate
```

### 7.3 证据交换表

```text
method, vehicles, malicious_ratio, Usability-AUC, Trust-ECE, Evidence bytes/frame
```

## 8. 预期结论

1. 在稳定空间失配下，`TrustCalib` 应显著降低 residual，并使 WPC% 接近无异常设置。
2. 在伪造目标攻击下，`TrustCalib` 应降低误刹和误避障，FA(f) 低于 `Raw-Coop`。
3. 在目标删除/故障下，`TrustCalib` 应表现优于 `Ego-Only` 和 `Hard-Filter`，因为它能保留仍可用的协同信息。
4. 加入 evidence exchange 后，单车视角证据不足的场景中，可用性判断更稳定，证据链可解释性更强。
