# 项目整合蓝图

## 1. 系统整体架构
- **程序入口**：`code/main.py` 负责实例化 `DroneCombatSystem`，拉取态势决策，并驱动 `AircraftManeuvering` 执行机动，形成“观测→决策→机动”的核心循环。
- **战场环境**：`code/BattlefieldEnvironment/__init__.py` 随机生成蓝红双方初始状态，并可聚合友军/敌军信息，作为全局观测输入。【F:code/BattlefieldEnvironment/__init__.py†L1-L231】
- **威胁评估**：`code/ThreatAnalyzer/__init__.py` 依据距离、能量、高度差等特征输出敌方威胁值，并为每架无人机挑选威胁最大的目标。【F:code/ThreatAnalyzer/__init__.py†L1-L216】
- **策略推理**：`code/DroneCombatSystem.py` 将环境观测与威胁分析结果输入 `StrategyPredictor`，得到每架无人机的战术指令集合。【F:code/DroneCombatSystem.py†L1-L228】
- **机动执行**：`code/AircraftManeuvering.py` 按策略类型生成速度、位置更新与能量消耗，输出状态报告。【F:code/AircraftManeuvering.py†L1-L312】

## 2. 强化学习与分层控制
- **核心配置**：`HierarchicalRLConfig` 定义高、低层策略的学习率、折扣因子、ε-贪心系数与经验回放容量，是分层强化学习的参数入口。【F:code/Hierarchical.py†L13-L75】
- **专家与策略融合**：`HierarchicalRLAgent` 中的低层网络输出七种机动动作（攻击、规避、待机、撤退、侧翼、爬升、俯冲），并与 `ExpertKnowledgeSystem` 的先验概率按 0.3:0.7 融合，再通过 ε-贪心采样确定最终动作索引。【F:code/Hierarchical.py†L132-L197】【F:code/Hierarchical.py†L188-L240】【F:code/Hierarchical.py†L368-L387】
- **策略结果组织**：`HierarchicalDecisionSystem.get_hierarchical_decisions` 将动作索引映射为中文策略标签，并在需要时附上目标敌机 ID，最终生成 `(drone_id, 策略标签, 目标参数)` 的三元组集合。【F:code/Hierarchical.py†L566-L616】
- **自适应反馈**：`update_rewards_and_train` 根据生存率、能量、击毁数量与编队紧凑度计算奖励，写入回放缓冲区并触发高低层训练，使动作分布随任务表现持续演化。【F:code/Hierarchical.py†L623-L685】

## 3. 元学习迁移方案
- **任务级经验池**：在 `HierarchicalManeuverIntegrator` 的训练循环中按战场场景拆分经验，构建“任务→梯度轨迹”缓冲区，为元学习准备数据。【F:code/Hierarchical.py†L704-L719】
- **MAML 风格更新**：在 `train_high_level` 和 `train_low_level` 例程中追加“内循环快速适配 + 外循环共享更新”，让策略在少量新样本下快速迁移至全新任务。【F:code/Hierarchical.py†L389-L520】
- **动态触发逻辑**：当检测到敌友数量突变或奖励剧烈波动时，可基于当前元初始化执行一次“小样本微调”，随后恢复常规 ε-贪心探索，以保证策略对动态环境的快速响应。【F:code/Hierarchical.py†L362-L387】【F:code/Hierarchical.py†L566-L621】

## 4. 自适应博弈的现状与补强
- **现有能力**：动态贝叶斯策略通过查表与启发式规则实现态势驱动的战术选择，但缺乏跨回合自更新能力。【F:code/StrategyPredictor/__init__.py†L1-L255】
- **不足分析**：`adaptive_game_assessment.md` 指出静态概率表和未激活的强化学习训练闭环是实现真正自适应博弈的瓶颈。
- **增强路径**：结合第 3 节的元学习流程与第 2 节的分层训练循环，可以在保留专家规则稳定性的同时，实现策略的持续自适应与跨场景迁移。

## 5. 集成与打包建议
1. **统一文档**：本文档汇总原先分散的分析、策略说明与迁移方案，可作为项目交付时的“技术总览”。
2. **执行流程**：
   - 运行 `code/main.py` 可快速体验单轮“态势→决策→机动”的主流程。
   - 使用 `code/demo.py` 中的 `MultiMissileCooperativeGuidanceSystem` 可加载分层强化学习与机动集成器，进行多回合训练。
3. **打包脚本**：仓库根目录新增 `package_project.py`，可自动打包代码与文档，详见 README 更新。

