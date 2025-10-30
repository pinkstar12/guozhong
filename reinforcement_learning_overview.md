# `code` 包中的强化学习策略解析

## 涉及的强化学习模块
- `HierarchicalRLConfig` 在同一个文件里设定了高、低两层策略的学习率、折扣因子、ε-贪心探索率、经验回放容量等关键超参数，是整套分层强化学习逻辑的配置入口。【F:code/Hierarchical.py†L13-L75】
- `HierarchicalRLAgent` 负责把高层/低层策略网络、专家系统、经验回放、目标网络和训练例程连接到一起，从而形成可持续更新的强化学习闭环。【F:code/Hierarchical.py†L268-L476】
- `HierarchicalDecisionSystem.get_hierarchical_decisions` 会在每个时间步采集观测、选择领导者、调用强化学习策略生成动作并记录训练所需的状态转移，是策略真正落地执行的入口函数。【F:code/Hierarchical.py†L566-L655】

## 强化学习策略的组成与决策流程
### 七种低层动作空间
当前低层策略将网络输出的离散动作索引映射为七种机动策略：`attack`、`evade`、`hold`、`retreat`、`flank`、`climb`、`dive`。这一映射直接出现在决策主循环中，是强化学习可选择的全部动作集合。【F:code/Hierarchical.py†L585-L616】

### 动作概率的融合逻辑
1. **专家先验**：`ExpertKnowledgeSystem.get_expert_action_probability` 会根据无人机与敌机的距离、威胁分级和任务类型生成七维概率分布，提供战术直觉权重。【F:code/Hierarchical.py†L188-L240】
2. **神经策略**：低层网络将局部态势向量与高层子目标拼接编码，输出学习得到的动作分布及对应的状态价值评估。【F:code/Hierarchical.py†L132-L197】【F:code/Hierarchical.py†L368-L387】
3. **融合与探索**：策略决策阶段以 0.3:0.7 的比例融合专家概率和神经网络概率，再使用 ε=0.2 的贪心策略采样动作索引，实现经验规则与学习策略的互补。【F:code/Hierarchical.py†L368-L385】

### 策略结果的组织形式
- 每架无人机都会把最终的动作索引转换为对应的中文策略标签，并在需要瞄准敌机的策略（如 `attack`、`flank`）下附带最近敌机的 ID 作为目标参数，最终形成 `(drone_id, strategy_name, target_id_or_None)` 的策略三元组集合。【F:code/Hierarchical.py†L585-L616】
- 系统同时打印领导/跟随角色与价值评估，便于分析该策略在当前态势下的期望收益，体现出强化学习对结果给出的信心度量。【F:code/Hierarchical.py†L615-L616】

## 强化学习策略结果的自适应来源
- `update_rewards_and_train` 会在机动执行后对生存率、剩余能量、击毁数量和编队紧凑度等指标进行打分，并把回报写入回放缓冲区，触发高、低层策略网络的训练更新。随着回报反馈的累积，动作概率分布会不断调整，实现自适应的策略结果。【F:code/Hierarchical.py†L623-L685】
- 奖励函数 `calculate_reward` 明确了上述指标的具体计算方式，例如基于生存率的基础奖励、按敌机消灭数量叠加的任务奖励，以及基于队形紧凑度的协同性奖励，从而引导策略朝着任务完成与团队协同的方向优化。【F:code/Hierarchical.py†L656-L685】

## 总结
`code/Hierarchical.py` 已经提供了分层强化学习的完整链条：上层负责任务拆解，下层在专家规则与神经策略的融合下输出七种机动策略，执行结果再通过奖励反馈回到经验池中持续训练。这些机制共同保证了策略结果能够随任务表现自适应地演化。
