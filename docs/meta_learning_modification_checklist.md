# 元学习改造清单

下面列出的修改步骤说明了如何在现有的分层强化学习框架中植入基于任务的元学习流程，并与强化学习训练循环打通，确保策略能够在动态环境中快速适配。

## 1. 扩展配置项以描述元学习节奏
- 在 `HierarchicalRLConfig` 中新增元批次大小、内循环学习率/步数、外循环步长、触发频率以及任务缓冲容量、奖励阈值等字段，用于统一驱动元训练的节奏与触发条件。【F:code/Hierarchical.py†L23-L38】

## 2. 为智能体引入任务级经验池与计数器
- 在 `HierarchicalRLAgent` 初始化阶段，创建按高、低层分别维护的 `meta_memory`，并按任务 ID 聚合轨迹；同时为两层策略维护独立的 `meta_counters` 以便按周期触发元更新。【F:code/Hierarchical.py†L311-L319】

## 3. 在常规训练循环中挂接元学习入口
- 在 `train_high_level`、`train_low_level` 成功执行一次梯度更新后，递增对应的 `meta_counters`，当计数达到配置阈值时调用 `_perform_meta_update` 完成一次外循环权重回写。【F:code/Hierarchical.py†L444-L503】
- 在经验写入阶段，通过 `store_meta_experience` 把同一任务的状态-动作-奖励序列保存到任务缓冲区，为后续内循环抽样提供素材。【F:code/Hierarchical.py†L512-L534】【F:code/Hierarchical.py†L794-L835】

## 4. 实现 Reptile 风格的内外循环更新
- `_perform_meta_update` 负责按任务批次随机采样、构造内循环批次，并调用 `_reptile_update` 完成参数快照、SGD 微调与外循环步长合并。【F:code/Hierarchical.py†L521-L593】
- `_reptile_update` 针对高层与低层策略分别建立快速副本，在支持集上执行 `inner_steps` 次梯度下降后，用差值乘以 `meta_step_size` 回写主网络，形成一阶近似的元梯度更新。【F:code/Hierarchical.py†L546-L593】

## 5. 在决策闭环中接入任务识别与快速适配
- `classify_task_type` 根据无人机数量、敌情密度、威胁等级等指标输出 `task_id`，保证元经验按任务场景聚类。【F:code/Hierarchical.py†L650-L668】
- `update_rewards_and_train` 在写入高、低层经验的同时将任务标签传入 `store_meta_experience`，并在奖励绝对值超过阈值时调用 `rapid_adaptation` 使用最新任务样本执行一次快速 Reptile 更新，实现对环境突变的即时响应。【F:code/Hierarchical.py†L784-L846】

通过以上改造，分层强化学习主体即可获得按任务聚合经验、周期性执行 Reptile 式元更新以及奖励波动触发快速适配的完整闭环，保证在动态战场环境中持续保持策略鲁棒性与迁移性。
