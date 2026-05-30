# 研究点 2.2：空间失配场景下考虑异常信息的安全可信协同感知优化

本文档夹给出一个可直接落到当前 SafeCoDriver 代码体系中的实现方案。核心定位是：在现有 `HybridSafetyConstraint` 之前增加一层 **可信协同感知校准层**，先对多车协同消息做可用性评估、偏移估计、证据交换和校正，再把校正后的 `PerceptionResult` 交给现有 V1 检测与几何路点修正模块。

## 文件结构

| 文件 | 内容 |
|---|---|
| `01_method_design.md` | 方法设计：信息可用性、车辆可信度、消息可用性、偏移估计、证据交换、检测-校正闭环 |
| `02_code_integration_plan.md` | 代码落地计划：新增模块、数据结构、接口、与现有 SafeCoDriver 的衔接方式 |
| `03_experiment_plan.md` | 实验方案：异常注入、指标、消融、DeepAccident/SUMO 接入路径 |
| `api_blueprint.py` | 后续实现时可参考的数据结构和接口蓝图，不参与当前运行 |
| `prototype/` | 当前研究点 2.2 的可复现实验原型代码 |
| `paper_ready/` | 面向 A 类论文整理的实验流程、主表、消融表、统计区间、runtime 表 |
| `results_summaries/` | 关键实验的轻量结果摘要，不包含大型中间记录 |
| `research_notes/` | 自动 research、文献梳理、方法约束、实验日志与论文定位笔记 |
| `MANIFEST.md` | 本次归档上传清单、未上传内容说明、核心结论索引 |

## 当前上传状态

本目录已经整理为远程仓库可审阅版本。为了避免污染代码仓库，未上传 DeepAccident 数据集、PDF 原文、大型中间 `cluster_records.csv`、完整 results 缓存和 conda 环境文件。论文论证所需的代码、表格、摘要结果和诊断记录已保留。

当前最重要的总览文档是：

```text
paper_ready/A_CLASS_EXPERIMENT_FLOW_AND_STATUS_2026-05-30.md
```

关键结论：

```text
可靠 GT-derived 多源证据下，最终方法在 clean/drop/fake_front/noise+fake_front
等模式上恢复到 clean-level WPC。

真实 DeepAccident 多源标签经坐标对齐和目标自车副本过滤后，real multi-source
evidence guard 从 EgoOnly 1.850% 改善到 1.625%，但仍明显低于 CleanCoop
oracle 0.425%，因此真实多源结论必须作为边界和诊断结果表述。
```

## 总体思路

当前最优主线是：

```text
PerceptionResult
  -> HybridSafetyConstraint
       -> V1 collision detector
       -> visibility-aware geometric correction
       -> TTC / RearEscape closed-loop control
```

研究点 2.2 建议扩展为：

```text
Raw multi-source cooperative messages
  -> TrustCalibLayer
       -> source association
       -> offset estimation
       -> message usability scoring
       -> long-term vehicle trust update
       -> evidence exchange and consensus
       -> calibrated PerceptionResult
  -> HybridSafetyConstraint
       -> V1 detection
       -> geometric waypoint correction
       -> TTC / min-harm / rear-aware control
```

关键设计原则：

1. **统一异常建模**：恶意攻击、传感器故障、空间标定失配、固有噪声都转化为多源观测之间的残差、置信度退化和可校正性问题。
2. **可用性不是二分类**：每条消息输出 `usable_score`、`correctable_score`、`risk_score` 和建议动作，而不是简单丢弃。
3. **长期信任与即时可用性分离**：车辆可信度描述长期表现，消息可用性描述当前帧是否可用、是否需要校正。
4. **校正优先于丢弃**：对稳定空间偏移和轻中度噪声，优先估计偏移并校正；对不可解释跳变、疑似伪造和高安全影响消息才降权或隔离。
5. **证据链可解释**：每次判断保留残差、匹配数量、偏移估计、共识来源和历史趋势，支撑论文中的可解释可信判断。

## 与现有研究点三的关系

研究点三关注“感知结果进入规划后如何约束动作空间和路点”，当前代码已有 V1 + 几何修正 + TTC/RearEscape 的实用版本。研究点 2.2 关注“进入安全约束之前的协同感知信息是否可信、是否可校正”。两者可以解耦组合：

```text
研究点 2.2：可信感知输入
研究点 3：安全动作空间/路点约束
```

这样论文叙事上可以形成完整链路：先保证协同感知输入可信，再保证基于该输入的下游规划安全。
