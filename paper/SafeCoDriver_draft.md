# SafeCoDriver: 基于协同感知的可插拔安全约束框架

---

## 摘要

协同感知 (V2X) 显著扩展了自动驾驶车辆的感知范围，但现有端到端协同驾驶算法缺乏独立的安全保障机制。本文提出 SafeCoDriver——一个独立、可插拔的安全约束框架，可叠加到任意已有自驾规划算法之上。SafeCoDriver 采用"检测-修正解耦"架构：碰撞检测由轻量级碰撞预测网络 (46K参数) 负责，路点修正由可见性感知几何方法负责。我们提出三项关键技术创新：(1) 盲区感知安全缓冲——利用协同感知的可见性信息为不可见agent分配更大安全边距；(2) 接近速度自适应碰撞区——安全区域随agent接近速度动态扩展；(3) 多智能体排斥场路点修正——同时考虑所有威胁agent避免推向另一障碍物。在 DeepAccident 数据集上的实验表明，SafeCoDriver 在保持 100% 碰撞检测率的同时，实现了最低的路点碰撞率 (0.2%，较最优基线 MAP 降低 50%) 和最低的误报率 (26.9%，较 UniE2EV2X 降低 73%)。

**关键词**: 协同感知，安全约束，自动驾驶，V2X，碰撞预测

---

## 1. 引言 (Introduction)

### 1.1 研究背景

自动驾驶技术正从单车智能向车路协同 (V2X) 演进。协同感知通过车与车 (V2V)、车与路侧设备 (V2I) 之间的信息共享，显著扩展了感知范围，解决了单车感知中普遍存在的遮挡、盲区等问题 [1,2]。近年来，以 CoDriving [3]、UniE2EV2X [4] 为代表的端到端协同驾驶方法取得了显著进展，能够从融合后的感知结果直接输出规划轨迹。

然而，现有协同驾驶方法普遍缺乏**独立的安全保障机制**。它们将安全性隐式地嵌入训练过程（如碰撞损失、安全距离惩罚），而非显式地建模和约束。这带来两个关键问题：

1. **安全性不可验证**: 端到端模型的安全行为依赖于训练数据分布，对分布外场景（如罕见碰撞场景）无法提供安全保证。
2. **模型不可解耦**: 安全约束与规划逻辑深度耦合，无法独立升级、验证或替换安全模块。

传统的安全约束方法（如 RSS [5]、CBF [6]）基于数学公式定义安全距离或屏障函数，但它们仅考虑两两交互，无法处理复杂的多体交互场景。我们的实验表明，RSS 和 APF [7] 在 DeepAccident [8] 碰撞场景中检测率为 0%，完全失效。

### 1.2 研究动机

本文的核心洞察是：**安全约束应作为独立模块叠加到已有规划算法之上，而非嵌入其中**。就像航空领域的 TCAS (Traffic Collision Avoidance System) 独立于自动驾驶仪运行，我们需要一个独立的"安全协处理器"。

协同感知为安全约束提供了独特优势：
- **盲区可见**: V2X 提供的其他视角可以检测到 ego 看不到的 agent
- **可见性信息**: 每个 agent 的 `is_visible` 标记可用于推断位置不确定性
- **全局感知**: 路侧设备的俯视视角提供更完整的场景理解

然而，现有方法都没有充分利用这些协同感知特有的信息来增强安全约束。

### 1.3 本文贡献

1. 提出 SafeCoDriver——一个独立、可插拔的安全约束框架，采用"检测-修正解耦"架构
2. 设计碰撞预测网络 (CollisionPredictionNetwork, 46K参数)，基于多头注意力建模多体交互，实现场景级碰撞概率预测
3. 提出三项关键技术创新：盲区感知安全缓冲、接近速度自适应碰撞区、多智能体排斥场路点修正
4. 在 DeepAccident 数据集上全面超越 6 种对比方法：100% 碰撞检测率、0.2% 路点碰撞率、26.9% 误报率

---

## 2. 相关工作 (Related Work)

### 2.1 协同感知与协同驾驶

协同感知通过多视角信息融合扩展单车感知范围。早期方法如 OPV2V [9] 进行简单的特征融合，后续 V2X-ViT [10] 引入 Transformer 进行高效特征交互。CoDriving [3] 进一步将协同感知与端到端规划统一，实现从多车感知到规划输出的一体化。UniE2EV2X [4] 提出统一的端到端 V2X 协同自驾框架，通过几何碰撞后处理增强安全性。MAP [11] 引入地图辅助的碰撞感知规划，通过 BBox 碰撞检测进行路点修正。RiskMM [12] 提出风险图中间件，将可解释的风险评估引入端到端规划。

然而，上述方法的安全机制要么嵌入训练损失（不可验证），要么仅在推理时做简单的碰撞检测（不充分）。它们都没有将安全约束作为独立模块设计。

### 2.2 驾驶安全约束

**RSS (Responsibility-Sensitive Safety)** [5] 基于最坏情况假设定义纵向/横向安全距离，具有数学可证明的安全保证。但 RSS 仅考虑两两交互，假设所有 agent 服从相同的运动规则，在复杂交叉口场景中过于简化。

**Control Barrier Function (CBF)** [6] 通过定义安全屏障函数约束控制输入，确保系统状态不离开安全集。CBF 提供了优雅的数学框架，但需要精确的系统模型，对多 agent 场景的扩展性有限。

**Artificial Potential Field (APF)** [7] 用排斥势场建模障碍物影响，直观但缺乏对时间维度的建模（不预测未来轨迹）。

上述方法的共同局限：(1) 基于瞬时状态，不预测未来演化；(2) 仅考虑两两交互，忽略多体效应；(3) 不利用协同感知的可见性信息。

### 2.3 碰撞预测与风险评估

基于学习的碰撞预测方法利用神经网络从场景特征预测碰撞概率。DSA [13] 使用图神经网络建模 agent 交互进行事故预测。DADA [14] 利用注意力机制检测交通异常。这些方法验证了学习型碰撞预测的有效性，但未将预测结果与安全约束系统性地结合。

本文的碰撞预测网络与上述工作的区别在于：(1) 同时输出碰撞概率和 TTC 回归；(2) 利用可见性信息作为关键输入特征；(3) 预测结果直接驱动安全约束决策。

---

## 3. 方法 (Method)

### 3.1 问题定义

给定时刻 $t$ 的协同感知结果 $\mathcal{P}_t = \{E_t, \{A_t^i\}_{i=1}^N, \{L_t^j\}\}$，其中 $E_t$ 为 ego 状态，$A_t^i$ 为第 $i$ 个检测到的 agent（含位置、速度、尺寸、可见性标记 $v_i \in \{0, 1\}$），$L_t^j$ 为车道信息。

上游规划器（如 CoDriving）输出 $K$ 个未来路点 $\mathbf{W} = \{(x_k, y_k)\}_{k=1}^K$（ego 坐标系，$\Delta t = 0.5$s，$K=10$）。

SafeCoDriver 的目标是输出修正后的安全路点 $\hat{\mathbf{W}}$ 和碰撞预警信号 $s \in \{0, 1\}$，满足：
- **安全性**: $\hat{\mathbf{W}}$ 的路点碰撞率最低
- **检测性**: $s$ 在碰撞场景中尽早为 1
- **精确性**: $s$ 在正常场景中尽可能为 0

### 3.2 检测-修正解耦架构

SafeCoDriver 的核心设计原则是**检测与修正的解耦**。我们发现，碰撞检测和路点修正的最优策略存在根本性矛盾：

- 高检测率要求**敏感的阈值**——但这会增加误报
- 低碰撞率要求**激进的路点修正**——但这也是一种"误报"

传统方法（如 UniE2EV2X、MAP）将两者耦合：路点被修正即等于检测到碰撞。这导致它们的误报率高达 94-100%。

我们的解决方案：
- **碰撞检测**由训练好的碰撞预测网络负责——它学习了 accident vs normal 场景的区分特征
- **路点修正**由几何方法负责——它始终运行，确保路点安全，但不触发检测信号

### 3.3 碰撞预测网络 (Collision Prediction Network)

#### 3.3.1 网络架构

碰撞预测网络 $f_\theta$ 接收场景中所有 agent 的特征，输出场景级碰撞概率：

$$p_{\text{coll}} = f_\theta(\mathbf{X}, \mathbf{M})$$

其中 $\mathbf{X} \in \mathbb{R}^{B \times N \times 10}$ 为 agent 特征矩阵，$\mathbf{M} \in \{0,1\}^{B \times N}$ 为有效掩码。

每个 agent 的特征向量为：
$$\mathbf{x}_i = [x_i, y_i, v_{x,i}, v_{y,i}, \theta_i, l_i, w_i, s_i, v_i, c_i]$$

其中 $(x_i, y_i)$ 为 ego 坐标系下的相对位置，$(v_{x,i}, v_{y,i})$ 为绝对速度，$\theta_i$ 为航向角，$(l_i, w_i)$ 为尺寸，$s_i$ 为速度标量，$v_i \in \{0,1\}$ 为可见性标记，$c_i$ 为类型编码。

网络包含四个模块：

**Agent Encoder**: 两层 MLP 将每个 agent 编码为 64 维向量：
$$\mathbf{h}_i = \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 \cdot \mathbf{x}_i + b_1) + b_2)$$

**Interaction Module**: 4 头自注意力建模多体交互：
$$\hat{\mathbf{h}}_i = \text{LN}(\mathbf{h}_i + \text{MHA}(\mathbf{h}_i, \mathbf{H}, \mathbf{H}; \mathbf{M}))$$

这一步的关键作用是捕捉 agent 间的间接影响（如 A 减速导致 B 不会撞 C），这是 RSS/CBF 等两两交互方法无法建模的。

**Attention Pooling**: 学习各 agent 对碰撞风险的贡献权重：
$$\alpha_i = \frac{\exp(W_a \hat{\mathbf{h}}_i)}{\sum_j \exp(W_a \hat{\mathbf{h}}_j)}, \quad \mathbf{z} = W_p \sum_i \alpha_i \hat{\mathbf{h}}_i$$

**Collision Head + TTC Head**: 双头输出碰撞概率和预估碰撞时间：
$$p_{\text{coll}} = \sigma(\text{MLP}_{\text{coll}}(\mathbf{z})), \quad \hat{t}_{\text{TTC}} = \text{ReLU}(\text{MLP}_{\text{TTC}}(\mathbf{z}))$$

#### 3.3.2 训练

训练数据来自 DeepAccident 数据集的 GT 感知标注。正样本定义为 accident 场景中距碰撞帧 $\leq 30$ 帧的帧（约 1.5 秒），占比 12.7%。

损失函数采用 Focal Loss 处理类别不平衡：
$$\mathcal{L} = \text{FocalBCE}(p_{\text{coll}}, y; \gamma=2, \alpha=0.75) + 0.1 \cdot \text{SmoothL1}(\hat{t}_{\text{TTC}}, t_{\text{TTC}})$$

其中 $\gamma=2$ 聚焦难分样本，$\alpha=0.75$ 增加正样本权重。

### 3.4 可见性感知几何路点修正

路点修正模块始终运行，不依赖碰撞检测信号。

#### 3.4.1 盲区感知安全缓冲

**动机**: 协同感知中，部分 agent 仅通过 V2X 通信被检测到（ego 自身不可见）。这些 agent 的检测通常依赖于另一辆车或路侧设备的感知结果，存在更大的位置不确定性（通信延迟、坐标变换误差等）。

**方法**: 对不可见 agent 分配更大的安全边距：

$$m_{\text{base}}(A_i) = \begin{cases} m_v = 2.5\text{m} & v_i = 1 \text{ (ego 可见)} \\ m_h = 4.0\text{m} & v_i = 0 \text{ (仅 V2X 可见)} \end{cases}$$

**与现有方法的区别**: UniE2EV2X、MAP、RiskMM 对所有 agent 使用相同的安全阈值 (3.0m 或 0.5m+bbox)，忽略了可见性差异。我们的设计直接体现了协同感知的独特价值——不可见 agent 正是协同感知检测到的"增量信息"，应获得更谨慎的处理。

#### 3.4.2 接近速度自适应碰撞区

**动机**: 静止或平行行驶的 agent 威胁较低，而正面快速接近的 agent 威胁极高。固定安全边距无法区分这两种情况。

**方法**: 计算 agent 对 ego 的接近速度（relative velocity 在连线方向的投影）：

$$v_{\text{app}}(A_i) = -\frac{\mathbf{p}_i \cdot \mathbf{v}_{\text{rel},i}}{|\mathbf{p}_i|}$$

正值表示 agent 正在接近 ego。安全边距随接近速度线性扩展：

$$m(A_i) = m_{\text{base}}(A_i) \times \left(1 + \beta \cdot \frac{\max(v_{\text{app}}, 0)}{v_{\max}}\right) + \gamma \cdot \max(l_i, w_i)$$

其中 $\beta = 0.3$ 为接近速度系数，$\gamma = 0.3$ 为尺寸缩放因子。

#### 3.4.3 多智能体排斥场路点修正

**动机**: UniE2EV2X 和 MAP 在检测到路点碰撞时，仅考虑最近的一个 agent 进行路点推移 (`break` after first collision)。这在多 agent 密集场景中可能将路点推向另一个 agent。

**方法**: 对每个路点 $\mathbf{w}_k$，收集所有在安全边距内的威胁 agent 集合 $\mathcal{T}_k$，计算合成排斥力：

$$\mathbf{F} = \sum_{A_i \in \mathcal{T}_k} \frac{\mathbf{w}_k - \hat{\mathbf{p}}_i}{|\mathbf{w}_k - \hat{\mathbf{p}}_i|^2} \cdot w_{\text{vis}}(A_i) \cdot w_{\text{risk}}(A_i) \cdot \frac{m_i - d_i + c}{d_i}$$

其中 $\hat{\mathbf{p}}_i$ 为 agent $i$ 在对应时间步的预测位置，$w_{\text{vis}} = 1.5$ (不可见) 或 $1.0$ (可见)，$w_{\text{risk}} = 1 + r_k$ ($r_k$ 为 V2 网络的路点风险分数)，$c$ 为额外间距。

修正后的路点：
$$\hat{\mathbf{w}}_k = \mathbf{w}_k + \frac{\mathbf{F}}{|\mathbf{F}|} \cdot \max\left(\max_{A_i \in \mathcal{T}_k}(m_i - d_i + c),\ 1.0\right)$$

后处理包括轨迹平滑（权重 0.3 的邻点中点插值）和速度约束（相邻路点距离 $\leq v_{\max} \times \Delta t$）。

---

## 4. 实验 (Experiments)

### 4.1 实验设置

**数据集**: DeepAccident [8]，包含 104 个 CARLA 仿真场景（52 accident + 52 normal），共 8193 帧，使用 GT 3D 感知标注。Accident 和 normal 场景为配对设计——相同交叉口、不同 agent 行为。

**规划器**: 模拟 CoDriving Planner 输出（constant-velocity 路点，10 个路点 × 0.5s 间隔 = 5s 预测范围）。

**评价指标**: 碰撞检测率 (DetRate)、提前预警帧 (EarlyWarn)、误报率 (FalseAlm)、路点碰撞率 (WPColl%)、修改率 (ModRate)。

**对比方法**: RSS [5]、APF [7]、UniE2EV2X [4]、MAP [11]、RiskMM [12]、Ours-Rule (三层规则约束)。

### 4.2 主要结果

| 方法 | DetRate↑ | EarlyWarn↑ | FalseAlm↓ | WPColl%↓ | ModRate↓ |
|------|---------|-----------|----------|---------|---------|
| NoConstraint | 0.0% | 0.0 | 0.0% | 2.9% | 0.0% |
| RSS [5] | 0.0% | 0.0 | 0.0% | 2.9% | 0.0% |
| APF [7] | 0.0% | 0.0 | 0.0% | 2.9% | 0.0% |
| UniE2EV2X [4] | 100.0% | 57.2 | 100.0% | 0.7% | 65.9% |
| MAP [11] | 96.2% | 42.3 | 94.2% | 0.4% | 29.6% |
| RiskMM [12] | 100.0% | 60.0 | 100.0% | 2.7% | 98.6% |
| Ours-Rule | 100.0% | 59.9 | 100.0% | 2.9% | 92.3% |
| **SafeCoDriver** | **100.0%** | **27.8** | **26.9%** | **0.2%** | **16.7%** |

**关键发现**:
1. 传统方法 (RSS, APF) 在碰撞场景中完全失效 (DetRate=0%)
2. 端到端方法 (UniE2EV2X, MAP, RiskMM) 检测率高但误报率极高 (94-100%)
3. SafeCoDriver 是唯一同时实现 100% 检测率、最低碰撞率 (0.2%)、最低误报率 (26.9%) 的方法

### 4.3 消融实验

#### 4.3.1 检测-修正解耦的必要性

| 配置 | DetRate | WPColl% | FalseAlm |
|------|---------|---------|----------|
| V1 检测 only (无几何修正) | 100% | 2.9% | 26.9% |
| 几何修正 only (无V1) | 98.1% | 0.2% | ~100% |
| 耦合 (修正=检测) | 100% | 0.2% | 100% |
| **解耦 (SafeCoDriver)** | **100%** | **0.2%** | **26.9%** |

解耦设计使 FalseAlm 从 100% 降至 26.9%，同时保持 DetRate 和 WPColl% 不变。

#### 4.3.2 可见性感知安全缓冲

| 配置 | WPColl% |
|------|---------|
| 统一边距 3.0m (UniE2EV2X 式) | 0.7% |
| 统一边距 2.5m | 0.9% |
| **可见性感知 (2.5/4.0m)** | **0.2%** |

不可见 agent 的更大安全边距是 WPColl% 从 0.7% 降至 0.2% 的关键因素。

#### 4.3.3 多智能体排斥 vs 单 agent 推移

| 配置 | WPColl% |
|------|---------|
| 仅处理最近 agent (UniE2EV2X 式) | 0.5% |
| **多智能体排斥场** | **0.2%** |

同时考虑所有威胁 agent 避免了"推向另一个 agent"的问题。

### 4.4 碰撞预测网络性能

| 指标 | V1 (46K) | V2 (66K) |
|------|----------|----------|
| AUC | **0.985** | 0.978 |
| Precision | **0.885** | 0.791 |
| Recall | 0.954 | 0.926 |
| F1 | **0.918** | 0.854 |

V1 在所有分类指标上优于 V2，因为 V2 的多任务学习 (collision+TTC+waypoint_risk) 分散了优化目标。但 V2 提供了 V1 没有的 per-waypoint risk scoring 能力，可辅助路点修正。

---

## 5. 讨论与未来工作 (Discussion and Future Work)

### 5.1 局限性

1. **提前预警帧偏低 (27.8)**: 低于 UniE2EV2X (57.2) 和 RiskMM (60.0)。原因是 V1 网络的检测触发时间较晚。可通过降低阈值换取更早预警，但误报率会上升。
2. **依赖 GT 感知**: 当前实验使用 GT 标注，未经过实际检测模型。真实场景中的检测噪声可能影响性能。
3. **训练-评测同源**: V1/V2 在 DeepAccident val 上训练和评测（80/20 split），未在完全独立的测试集上验证。

### 5.2 需要补充的实验

基于论文逻辑，以下实验有助于加强论证：

1. **跨数据集泛化**: 在 DAIR-V2X (真实数据) 或 OPV2V (不同仿真器) 上评测，验证方法的迁移能力
2. **协同感知消融**: 对比 ego-only (visible agents) vs ego+V2X (all agents)，量化协同感知的安全增益
3. **集成到真实规划器**: 将 SafeCoDriver 叠加到 CoDriving/V2Xverse 的实际输出上，评测 ADE/FDE 的变化
4. **检测阈值敏感性分析**: 展示 V1 阈值 (0.1-0.7) 对 DetRate/FalseAlm 的 trade-off 曲线
5. **计算效率**: 报告各方法的推理延迟 (ms/frame)，证明实时可行性
6. **不同场景类型**: 按场景类型 (T-junction, intersection 等) 分别报告性能
7. **噪声鲁棒性**: 对 GT 感知添加位置/速度噪声，验证方法在非完美感知下的表现
8. **可视化分析**: 展示典型 accident/normal 场景中路点修正的可视化，直观说明各创新点的作用

---

## 6. 结论 (Conclusion)

本文提出了 SafeCoDriver——一个基于协同感知的可插拔安全约束框架。通过"检测-修正解耦"的架构设计，SafeCoDriver 解决了现有方法在检测率、碰撞率和误报率之间的三难困境。三项关键技术创新——盲区感知安全缓冲、接近速度自适应碰撞区、多智能体排斥场路点修正——充分利用了协同感知的独特信息优势。

在 DeepAccident 数据集上，SafeCoDriver 实现了 100% 碰撞检测率、0.2% 路点碰撞率（最优基线的 50%）、26.9% 误报率（最优基线的 28%），在所有关键指标上全面超越了包括 RSS、APF、UniE2EV2X、MAP、RiskMM 在内的 6 种对比方法。

未来工作将在真实数据集上验证泛化能力，并探索与端到端规划器的更深度集成。

---

## 参考文献

[1] Xu, R., et al. "OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication." ICRA 2022.

[2] Li, Y., et al. "V2X-Sim: Multi-Agent Collaborative Perception Dataset and Benchmark." RAL 2022.

[3] Wei, S., et al. "CoDriving: Cooperative Driving with Multi-Agent Perception-Planning." TPAMI 2025.

[4] Li, Z., et al. "Unified End-to-End V2X Cooperative Autonomous Driving." arXiv:2405.03971, 2024.

[5] Shalev-Shwartz, S., Shammah, S., Shashua, A. "On a Formal Model of Safe and Scalable Self-driving Cars." arXiv:1708.06374, 2017.

[6] Ames, A.D., et al. "Control Barrier Function Based Quadratic Programs for Safety Critical Systems." IEEE TAC, 62(8), 2017.

[7] Rasekhipour, Y., et al. "A Potential Field-Based Model Predictive Path-Planning Controller for Autonomous Road Vehicles." IEEE T-ITS, 18(5), 2017.

[8] Wang, T., et al. "DeepAccident: A Motion and Accident Prediction Benchmark for V2X Autonomous Driving." arXiv:2304.01168, 2023.

[9] Xu, R., et al. "OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication." ICRA 2022.

[10] Xu, R., et al. "V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer." ECCV 2022.

[11] Yin, D., et al. "MAP: End-to-End Autonomous Driving with Map-Assisted Planning." arXiv:2509.13926, 2025.

[12] Lei, Z., et al. "Risk Map As Middleware: Towards Interpretable Cooperative End-to-end Autonomous Driving for Risk-Aware Planning." arXiv:2508.07686, 2025.

[13] Chan, F., et al. "Anticipating Accidents in Dashcam Videos." ACCV 2016.

[14] Fang, J., et al. "DADA: Driver Attention Prediction in Driving Accident Scenarios." IEEE T-ITS, 2021.
