# Expert 模块与动态博弈代码链路的融合方案

本文档提出一套在不破坏现有 `DroneCombatSystem` 决策链路的前提下，将 `airbattle/expert` 目录下的攻防专家系统与 `airbattle` 包内的图融合 + 动态贝叶斯策略模块协同使用的落地方案。

## 设计目标
- **保持现有态势-决策闭环**：延续战场环境、GNN 信息融合、威胁评估、动态贝叶斯策略推理的流程。 【F:airbattle/DroneCombatSystem.py†L19-L102】
- **在高威胁场景启用专家兜底**：当威胁值或导弹锁定等条件触发时，调用专家系统的精细防御或攻击规则替换/修正动作。 【F:airbattle/expert/expertsystem.py†L160-L288】
- **复用专家系统输出结构**：对齐专家系统返回的 `actions`、`explanations` 与现有机动执行模块需要的决策格式，方便后续 `AircraftManeuvering` 直接消费。 【F:airbattle/expert/expertsystem.py†L357-L388】

## 核心改造点

1. **态势状态适配层 (`ExpertStateAdapter`)**
   - 功能：将 `BattlefieldEnvironment.get_all_observations('blue')` 的字典结构转换成专家系统 `process` 所需的 `state`：包含 `our_aircrafts`、`enemies`、`missiles` 三个数组字段，并在需要时补齐导弹威胁信息。 【F:airbattle/BattlefieldEnvironment/BattlefieldEnvironment.py†L61-L78】【F:airbattle/expert/expertsystem.py†L205-L288】
   - 关键步骤：
     1. 遍历蓝方无人机观测，提取 `self_state` 作为 `our_aircrafts`；从 `enemies` 列表生成敌机数组。
     2. 若环境当前没有导弹信息，可引入简单的威胁生成器（例如根据 `ThreatAnalyzer.assign_targets` 结果构造虚拟导弹条目），满足专家系统锁定识别逻辑。 【F:airbattle/DroneCombatSystem.py†L70-L101】
     3. 维护索引映射，确保专家系统返回的 `actions` 能映射回原始无人机 ID。

2. **决策融合器 (`DecisionFusionManager`)**
   - 功能：封装 DBN 策略决策与专家输出的合并策略。
   - 流程：
     1. 默认执行 `StrategyPredictor.predict_strategy` 得到基础策略标签。 【F:airbattle/StrategyPredictor/StrategyPredictor.py†L49-L143】
     2. 将适配后的 `state` 输入 `ExpertSystem.process`，获取每架无人机的专家建议动作及解释。 【F:airbattle/expert/expertsystem.py†L205-L388】
     3. 根据业务规则融合：
        - 若专家结果 `source` 为 `defend` 或 `attack`，并且与 DBN 策略冲突（例如 DBN 输出进攻而专家判定高威胁防御），优先采用专家动作并记录来源。
        - 否则保留 DBN 策略，同时可把专家解释附着在最终决策元组中，供机动执行模块或 UI 展示。
     4. 形成统一格式：`(drone_id, strategy, params, metadata)`，其中 `metadata` 可包含专家解释、威胁值等信息，为后续训练或日志分析提供依据。

3. **机动执行桥接 (`ManeuverCommandBuilder`)**
   - 功能：将融合后的策略+专家动作翻译成 `AircraftManeuvering` 可直接执行的命令对象。
   - 关键点：
     1. 对于纯策略决策，沿用现有 `(drone_id, strategy, params)` 结构进入 `AircraftManeuvering`。 【F:airbattle/DroneCombatSystem.py†L88-L102】
     2. 对于专家返回的具体姿态/规避动作，需在命令构建器中映射到机动模块提供的动作函数（如规避、撤退、编队），必要时扩展 `AircraftManeuvering` 的动作表以接受“专家驱动”指令。
     3. 通过统一的数据类（例如 `DecisionCommand`）封装策略标签、速度向量、解释文本，方便日志记录与后续调参。

## 迭代流程建议
1. **最小可行整合 (MVP)**
   - 实现 `ExpertStateAdapter` 与 `DecisionFusionManager`，在 `DroneCombatSystem.get_strategy_decisions` 尾部增加专家融合步骤，但暂时仅在高威胁（`threat_level == 'high'`）或存在导弹锁定时覆盖 DBN 策略。
   - 输出结果保持现有集合格式，额外提供一个解释字典供调试使用。

2. **增强版融合**
   - 引入 `ManeuverCommandBuilder`，为专家动作提供机动执行映射。
   - 在 `HierarchicalDecisionSystem` 中复用融合器，使分层 RL 可以访问专家解释与威胁指标，作为奖励 shaping 的附加信号。

3. **长期演进**
   - 将专家系统输出纳入 GNN 训练数据或 DBN 特征，形成“专家经验蒸馏”。
   - 通过日志统计分析专家与 DBN 冲突场景，动态调整阈值或融合权重，实现自适应专家介入策略。

## 开发与验证要点
- **接口一致性测试**：构造包含导弹威胁的模拟状态，验证 `ExpertSystem.process` 输出与 DBN 决策合并后的正确性。
- **性能监控**：专家系统调用涉及防御/优化模块，建议缓存转换结果并限制调用频率，避免在实时仿真中成为性能瓶颈。
- **可解释性输出**：保留专家解释文本，辅助调试 DBN 与专家冲突时的行为。

通过以上步骤，可以在保持现有动态博弈决策优势的同时，引入专家系统的可解释、可控策略，实现博弈模型与专家知识的互补协同。
