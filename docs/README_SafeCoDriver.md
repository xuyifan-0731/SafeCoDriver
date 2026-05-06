# SafeCoDriver — 算法说明、内容总结与复现指南

> 基于协同感知的可插拔安全约束框架
> 项目路径: `/raid/xuyifan/jiqiuyu/`
> 环境: `conda activate coop-safety` (Python 3.10, PyTorch 2.5.1+cu121)

---

## 一、算法总体说明

### 1.1 研究问题

协同感知 (V2X) 为自动驾驶提供了超越单车视野的环境信息，但现有端到端协同驾驶算法（如 CoDriving）缺乏独立的安全约束机制。本项目设计了一个**独立、可插拔的安全约束模块**（SafeCoDriver），接收协同感知结果和上游规划器输出的路点，输出安全修正后的路点。

### 1.2 核心架构: 检测-修正解耦

```
CoDriving Planner → predicted waypoints (10×2, 0.5s间隔, 5s预测)
                         │
         ┌───────────────┼───────────────┐
         │       SafeCoDriver            │
         │                               │
         │  ┌── V1 碰撞检测器 ──┐        │  → 输出: is_dangerous (检测信号)
         │  │  collision_prob   │        │     用于: DetRate / FalseAlm
         │  └───────────────────┘        │
         │                               │
         │  ┌── 几何路点修正器 ──┐        │  → 输出: safe waypoints
         │  │  visibility-aware │        │     用于: WPColl%
         │  │  approach-speed   │        │
         │  │  multi-agent      │        │
         │  └───────────────────┘        │
         └───────────────┬───────────────┘
                         │
                   safe waypoints (10×2)
```

**关键设计**: 碰撞检测（是否报警）和路点修正（如何避让）是**解耦**的。路点始终被修正（确保低碰撞率），但报警信号仅在 V1 网络确认危险时发出（确保低误报率）。

### 1.3 三项技术创新

| # | 创新 | 核心思想 | vs 对比方法 |
|---|------|---------|-----------|
| 1 | 盲区感知安全缓冲 | 不可见 agent (V2X-only) 安全边距 4.0m vs 可见 2.5m | UniE2EV2X/MAP 统一阈值 |
| 2 | 接近速度自适应碰撞区 | margin × (1 + 0.3×approach/v_max) | UniE2EV2X/MAP 固定阈值 |
| 3 | 多智能体排斥场 | 对所有威胁 agent 累加排斥力 | UniE2EV2X/MAP 只处理第一个碰撞 |

---

## 二、模块详细说明

### 2.1 碰撞预测网络 V1 (用于检测)

**文件**: `coop_safety/learned/collision_network.py`
**类名**: `CollisionPredictionNetwork`
**参数**: 46,659 (~191KB)
**功能**: 输入场景中所有 agent 特征，输出碰撞概率 P(collision in 30 frames)

```
输入: agents (B, 30, 10)  [x, y, vx, vy, heading, length, width, speed, visible, type]
      mask   (B, 30)       有效标记

AgentEncoderV2:     Linear(10→64) + ReLU + Linear(64→64) + ReLU     → (B, 30, 64)
InteractionModule:  MultiheadAttention(64, 4heads) + LayerNorm       → (B, 30, 64)
AttentionPool:      Attention(64→1) + Softmax + WeightedSum + Lin(64→128)  → (B, 128)
CollisionHead:      Linear(128→64) + ReLU + Linear(64→1) + Sigmoid  → (B, 1) = P(coll)
TTCHead:            Linear(128→64) + ReLU + Linear(64→1) + ReLU     → (B, 1) = TTC

训练: DeepAccident, Focal BCE(γ=2, α=0.75), Adam lr=5e-4, 80 epochs
结果: AUC=0.985, P=0.885, R=0.954, F1=0.918
```

### 2.2 碰撞预测网络 V2 (用于路点风险评分)

**文件**: `coop_safety/learned/collision_network_v2.py`
**类名**: `CollisionPredictionNetV2`
**参数**: 66,291 (~260KB)
**新增能力**: per-waypoint risk scoring

```
输入: agents (B, 30, 12)  [rel_x, rel_y, rel_vx, rel_vy, heading, l, w, speed, vis, type, approach, dist]
      ego    (B, 6)       [speed, yaw_rate, ax, ay, heading, 0]
      waypoints (B, 10, 2)

RelativeAgentEncoder:  spatial_mlp(8→64→64) + vis_embed(2,16) + type_embed(6,8) → fuse(88→64)
EgoCentricAttention:   ego→query, agents→k/v, MHA(64, 4heads) + agent_self_attn
SceneProjection:       Linear(64→128) + GELU
CollisionHead:         Linear(128→64→1) + Sigmoid
TTCHead:               Linear(128→64→1) + ReLU
WaypointRiskScorer:    [scene(128); wp(2)] → MLP(130→32→32→1) + Sigmoid  → per-wp risk

训练: DeepAccident, FocalBCE + 0.1×SmoothL1(TTC) + 0.2×BCE(wp_risk), AdamW lr=3e-4, 100 epochs
结果: AUC=0.978, 同时提供 waypoint risk scoring 能力
```

### 2.3 Hybrid 安全约束 (最终方法)

**文件**: `coop_safety/learned/hybrid_safety.py`
**类名**: `HybridSafetyConstraint`
**功能**: 结合 V1 检测 + 几何路点修正

```python
def constrain_waypoints(waypoints, perception):
    # 1. V2 路点风险评分 (可选辅助)
    wp_risks = V2.waypoint_risk(perception, waypoints)  # 可选

    # 2. 几何碰撞检测 + 多智能体排斥修正 (始终执行)
    for t in range(10):
        for each agent:
            predict position at dt = (t+1) × 0.5s
            margin = get_safety_margin(agent, ego_speed)  # 可见性+接近速度
            if dist < margin: add to threats
        if threats:
            waypoints[t] = multi_agent_repulsion(waypoints[t], threats)

    # 3. 轨迹平滑 + 速度约束

    # 4. V1 碰撞检测 (仅影响报警, 不影响路点)
    prob = V1(agents, mask)
    is_dangerous = prob > 0.3

    return modified_waypoints, {"n_collisions_detected": 1 if is_dangerous else 0}
```

### 2.4 三层规则安全约束 (Ours-Rule, 对比用)

**文件**: `coop_safety/interface.py` → `SafetyConstraintModule`

```
流程: 盲区推理 → RiskMap + RiskGraph + RiskEvents → 可行域 → 分层收紧 → 可行性检查 → 最小伤害
子模块:
  coop_safety/perception/blind_spot.py    — 遮挡几何 + 幽灵 agent
  coop_safety/perception/dynamics.py      — 自行车运动学模型
  coop_safety/perception/prediction.py    — CV/CA/CTRA 轨迹预测
  coop_safety/risk/risk_map.py            — 网格风险 (40m, 8m cell)
  coop_safety/risk/risk_graph.py          — 成对 TTC + 碰撞概率
  coop_safety/risk/risk_events.py         — 碰撞事件枚举
  coop_safety/constraint/feasible_region.py   — 可达集 ∩ 道路 - 碰撞区
  coop_safety/constraint/hierarchical.py      — TTC排除 + 风险区排除
  coop_safety/constraint/feasibility_check.py — 前瞻 3s 可行性
  coop_safety/constraint/min_harm.py          — 最小伤害候选动作评估
```

### 2.5 对比方法

| 方法 | 文件 | 类名 | 核心算法 |
|------|------|------|---------|
| RSS [Shalev-Shwartz17] | `experiments/methods.py` | `RSSOnly` | 纵向安全距离公式 |
| APF [Rasekhipour17] | `experiments/methods_modern.py` | `RiskPotentialField` | 各向异性排斥势场 |
| UniE2EV2X [Li24] | `experiments/methods_new_baselines.py` | `UniE2EV2XSafety` | 多边形碰撞后处理 |
| MAP [Yin25] | `experiments/methods_new_baselines.py` | `MAPSafety` | BBox碰撞+最小位移 |
| RiskMM [Lei25] | `experiments/methods_new_baselines.py` | `RiskMMSafety` | 高斯风险场+梯度MPC |

---

## 三、实验结果总结

### 3.1 主实验 (10 methods × 104 scenarios)

| 方法 | DetRate↑ | EarlyWarn↑ | FalseAlm↓ | WPColl%↓ | ModRate↓ | ms/frame↓ |
|------|---------|-----------|----------|---------|---------|-----------|
| NoConstraint | 0.0% | 0.0 | 0.0% | 2.9% | 0.0% | — |
| RSS | 0.0% | 0.0 | 0.0% | 2.9% | 0.0% | 84.1 |
| APF | 0.0% | 0.0 | 0.0% | 2.9% | 0.0% | 85.5 |
| UniE2EV2X | 100% | 57.2 | 100% | 0.7% | 65.9% | 10.2 |
| MAP | 96.2% | 42.3 | 94.2% | 0.4% | 29.6% | 6.6 |
| RiskMM | 100% | 60.0 | 100% | 2.7% | 98.6% | 4.9 |
| Ours-Rule | 100% | 59.9 | 100% | 2.9% | 92.3% | 384.6 |
| Ours-V1 | 100% | 27.8 | 26.9% | 2.9% | 16.7% | — |
| Ours-V2 | 17.3% | 12.3 | 3.8% | 2.8% | 0.8% | — |
| **Ours-Hybrid** | **100%** | **27.8** | **26.9%** | **0.2%** | **16.7%** | **1.8** |

### 3.2 补充实验

| 实验 | 关键结论 |
|------|---------|
| 计算效率 | Ours-Hybrid **1.8ms/frame** (最快，满足实时) |
| 协同感知消融 | ego+V2X 误报 26.9% vs ego-only 67.3% — 协同降低误报 |
| 阈值敏感性 | DetRate=100% 在所有阈值 (0.1-0.7) 下保持 |
| 噪声鲁棒性 | DetRate=100% 在高噪声 (1.0m) 下仍保持 |
| 多agent消融 | 多agent 0.2% vs 单agent 0.7% (↓71%) |

---

## 四、复现方法

### 4.1 环境配置

```bash
# 创建 conda 环境
conda create -n coop-safety python=3.10 -y
conda activate coop-safety
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy shapely scikit-learn

# 项目路径
cd /raid/xuyifan/jiqiuyu
```

### 4.2 数据准备

**DeepAccident 数据集** (~93GB):
- 下载: https://github.com/tianqi-wang1996/DeepAccident (val_part1 + val_part2)
- 解压到: `data/DeepAccident/`
- 结构:
  ```
  data/DeepAccident/
  ├── type1_subtype1_accident/    (27 scenarios)
  │   ├── ego_vehicle/label/      (每场景 50-100 帧 .txt)
  │   ├── other_vehicle/label/    (协同感知标注)
  │   └── meta/                   (碰撞帧信息)
  ├── type1_subtype1_normal/      (27 scenarios, 配对)
  ├── type1_subtype2_accident/    (25 scenarios)
  └── type1_subtype2_normal/      (25 scenarios)
  ```

### 4.3 训练

```bash
cd /raid/xuyifan/jiqiuyu

# 训练 V1 碰撞预测网络 (~2 min on 1×4090)
python coop_safety/learned/train_collision.py
# 输出: models/collision_net_best.pt (AUC=0.985)

# 训练 V2 改进网络 (~3 min on 1×4090)
python coop_safety/learned/train_collision_v2.py
# 输出: models/collision_net_v2_best.pt (AUC=0.978)
```

### 4.4 评测

```bash
# 主实验: 10方法统一评测 (~90 min, CPU-only)
python experiments/run_deepaccident_unified.py
# 输出: 终端打印结果表

# 补充实验: 7项消融+效率 (~11 min)
python experiments/run_supplementary.py
# 输出: 终端打印 + experiments/supplementary_results.log
```

### 4.5 已有模型权重

| 文件 | 大小 | 说明 |
|------|------|------|
| `models/collision_net_best.pt` | 191KB | V1 最佳 (AUC=0.985) |
| `models/collision_net_v2_best.pt` | ~260KB | V2 最佳 (AUC=0.978) |
| `models/collision_net_final.pt` | 191KB | V1 最终 epoch |
| `models/collision_net_v2_final.pt` | ~260KB | V2 最终 epoch |

---

## 五、代码结构说明

```
/raid/xuyifan/jiqiuyu/
│
├── coop_safety/                        # 核心算法包
│   ├── interface.py                    # 主接口: SafetyConstraintModule, 数据结构定义
│   │                                   #   PerceptionResult, VehicleState, Agent,
│   │                                   #   SafeActionSpace, ConstraintMode
│   ├── perception/
│   │   ├── dynamics.py                 # BicycleModel 运动学, get_dynamics_params()
│   │   ├── prediction.py              # predict_agent(): CV/CA/CTRA 轨迹预测
│   │   └── blind_spot.py              # compute_occlusion_polygon(), infer_hidden_agents()
│   ├── risk/
│   │   ├── risk_map.py                # RiskMapBuilder.build() → list[RiskRegion]
│   │   ├── risk_graph.py              # RiskGraphBuilder.build() → list[ConflictEdge]
│   │   └── risk_events.py             # RiskEventEnumerator.enumerate() → list[CollisionEvent]
│   ├── constraint/
│   │   ├── feasible_region.py         # FeasibleRegionComputer: 可达集 ∩ 道路 - 碰撞区
│   │   ├── hierarchical.py            # HierarchicalConstraint: TTC收紧 + 风险排除
│   │   ├── feasibility_check.py       # FeasibilityChecker: 前瞻3s模拟
│   │   └── min_harm.py                # MinimumHarmPlanner: 12候选动作评估
│   ├── learned/
│   │   ├── collision_network.py       # V1: CollisionPredictionNetwork (46K)
│   │   ├── collision_network_v2.py    # V2: CollisionPredictionNetV2 (66K)
│   │   ├── hybrid_safety.py          # ★ HybridSafetyConstraint (最终方法)
│   │   ├── train_collision.py         # V1 训练脚本
│   │   └── train_collision_v2.py      # V2 训练脚本
│   └── utils/
│       ├── metrics.py                 # compute_ttc(), compute_safety_distance()
│       └── visualization.py           # plot_scene_overview()
│
├── experiments/                        # 实验脚本
│   ├── deepaccident_loader.py         # DeepAccidentLoader: GT感知加载
│   ├── methods.py                     # 基线: NoConstraint, RSSOnly, CBFBased
│   ├── methods_modern.py              # APF (RiskPotentialField)
│   ├── methods_new_baselines.py       # UniE2EV2X, MAP, RiskMM
│   ├── run_deepaccident_unified.py    # ★ 主评测: 10方法 × 104场景
│   ├── run_supplementary.py           # ★ 补充实验: 消融/效率/噪声
│   └── analyze_signals.py             # 信号分析 (accident vs normal)
│
├── models/                             # 训练好的模型权重
│   ├── collision_net_best.pt          # V1 (AUC=0.985)
│   └── collision_net_v2_best.pt       # V2 (AUC=0.978)
│
├── data/DeepAccident/                  # 数据集 (~93GB)
│
├── paper/
│   └── SafeCoDriver_draft.md          # 论文初稿
│
├── 260429方法试验说明.md               # 最新方法+实验完整文档
├── 260428-v2方法试验说明.md            # Hybrid 方法文档
└── README.md                           # 项目概述
```

### 5.1 关键入口点

| 场景 | 调用方式 |
|------|---------|
| 使用安全约束 (推理) | `HybridSafetyConstraint(detector_model=v1).constrain_waypoints(wp, perception)` |
| 训练 V1 | `python coop_safety/learned/train_collision.py` |
| 训练 V2 | `python coop_safety/learned/train_collision_v2.py` |
| 运行评测 | `python experiments/run_deepaccident_unified.py` |
| 补充实验 | `python experiments/run_supplementary.py` |

### 5.2 数据流

```
DeepAccidentLoader.load_frame(si, fi)
  → DeepAccidentFrame
    .perception: PerceptionResult
      .ego: VehicleState (x=0, y=0, heading=0, velocity, vx, vy)
      .agents: list[Agent]
        .state: VehicleState (x, y, heading, velocity, vx, vy, length, width)
        .is_visible: bool (ego能否直接观测到)
        .confidence: float
    .is_accident_scenario: bool
    .ego_speed, .ego_yaw_rate: float

simulate_codriving_waypoints(frame)
  → numpy (10, 2): constant-velocity 路点 [x, y] in ego frame

HybridSafetyConstraint.constrain_waypoints(waypoints, perception)
  → (modified_waypoints, stats_dict)
     modified_waypoints: numpy (10, 2) 安全修正后的路点
     stats: {"n_collisions_detected": 0|1, "collision_prob": float, ...}
```

### 5.3 评测逻辑 (run_deepaccident_unified.py)

```python
for scenario in 104_scenarios:
    for frame in scenario.frames:
        base_wp = simulate_codriving_waypoints(frame)  # CoDriving 模拟
        for method in all_methods:
            if method has constrain_waypoints:
                modified_wp, stats = method.constrain_waypoints(base_wp, perception)
                was_modified = stats['n_collisions_detected'] > 0
            else:
                safe = method.constrain(perception)
                was_modified = safe.mode != NORMAL
                modified_wp = base_wp  # 旧方法不修改路点

            # 记录: 首次预警帧、误报帧、路点碰撞数
            wp_collisions += check_waypoint_collision(modified_wp, frame)

    # 汇总: DetRate, EarlyWarn, FalseAlm, WPColl%, ModRate
```

---

## 六、快速上手

```python
# 最小使用示例
import torch
import numpy as np
from coop_safety.learned.collision_network import CollisionPredictionNetwork
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from coop_safety.interface import PerceptionResult, VehicleState, Agent

# 加载模型
v1 = CollisionPredictionNetwork()
v1.load_state_dict(torch.load("models/collision_net_best.pt", map_location='cpu', weights_only=False)['model'])
v1.eval()

# 创建安全约束
safety = HybridSafetyConstraint(
    detector_model=v1,
    base_margin_visible=2.5,
    base_margin_invisible=4.0,
    detection_threshold=0.3,
)

# 构造输入
ego = VehicleState(id='ego', x=0, y=0, heading=0, velocity=10, vx=10, vy=0)
agents = [
    Agent(state=VehicleState(id='car1', x=15, y=2, heading=3.14, velocity=8, vx=-8, vy=0),
          is_visible=True),
    Agent(state=VehicleState(id='car2', x=20, y=-1, heading=3.14, velocity=12, vx=-12, vy=0),
          is_visible=False),  # 仅 V2X 可见
]
perception = PerceptionResult(timestamp=0, ego=ego, agents=agents)

# 上游规划器的路点 (constant velocity, 10步×0.5s)
waypoints = np.array([[10*(t+1)*0.5, 0] for t in range(10)])

# 安全约束
safe_wp, stats = safety.constrain_waypoints(waypoints, perception)
print(f"碰撞概率: {stats['collision_prob']:.2f}")
print(f"检测到危险: {stats['n_collisions_detected'] > 0}")
print(f"几何威胁数: {stats['n_geometric_threats']}")
print(f"原始路点[0]: {waypoints[0]} → 安全路点[0]: {safe_wp[0]}")
```

---

## 七、文档索引

| 文件 | 内容 | 优先级 |
|------|------|--------|
| `260429方法试验说明.md` | 最新完整方法+网络+实验说明 | ★★★ |
| `260428-v2方法试验说明.md` | Hybrid 方法详细设计+结果 | ★★ |
| `paper/SafeCoDriver_draft.md` | 论文初稿 (中文) | ★★ |
| `AGENTS.md` | 开发指引 + 历史进度 | ★ |
| `260424方法实验说明.md` | 早期 Ours-Rule + 基线结果 | ★ |
| `260427方法试验说明.md` | V1 碰撞预测网络 + 训练 | ★ |
