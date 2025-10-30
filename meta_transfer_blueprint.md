# 分层强化学习与元学习结合的迁移性增强蓝图

## 目标
在保持 `airbattle/Hierarchical.py` 现有分层强化学习框架的基础上，引入元学习机制，使策略在面对未见过的动态环境与任务切换时，能够通过少量采样快速完成适配，提升决策算法的迁移性和稳定性。

## 模块化扩展思路
1. **任务级数据缓冲区**：沿用 `HierarchicalManeuverIntegrator` 当前的仿真循环，将每个任务回合的状态轨迹、动作和奖励拆分存入新的 `MetaReplayBuffer`。可以在现有的 `training_interval` 节点调用扩展接口，将 `update_rewards_and_train` 过程中生成的状态与奖励序列复制到按任务索引的缓冲区，为元训练提供多样化任务样本。【F:airbattle/Hierarchical.py†L704-L719】【F:airbattle/Hierarchical.py†L623-L655】
2. **内循环快速适配**：在 `HierarchicalRLAgent.train_high_level` 和 `train_low_level` 内部，增加“参数快照 → 若干步梯度更新 → 计算查询损失”的流程：
   - 先复制当前网络参数作为初始点；
   - 使用任务缓冲区中的支持集执行 1-3 次梯度下降，得到任务专属参数；
   - 再用查询集计算损失，并对原始参数执行外循环更新，实现近似 MAML 的元梯度训练。【F:airbattle/Hierarchical.py†L389-L472】
3. **任务嵌入与条件化策略**：在 `extract_global_state` 与 `extract_local_state` 产生的特征向量基础上，增加“任务描述符”维度（如敌友比、通信约束标记），并在低层网络前向时将其视作条件输入，使网络在不同任务间共享结构但保持可区分性。【F:airbattle/Hierarchical.py†L520-L564】【F:airbattle/Hierarchical.py†L362-L387】
4. **快速再训练触发器**：通过监控奖励移动平均与价值估计的 KL 散度，一旦发现异常波动，就在执行线程中调用一次快速适配步骤（使用上一阶段维护的元初始化），把最新几步轨迹当作支持集、查询集，在不打断主流程的前提下完成少量梯度步更新。【F:airbattle/Hierarchical.py†L566-L621】【F:airbattle/Hierarchical.py†L623-L655】

## 预期收益
- **跨场景迁移**：元初始化提供的通用策略能让无人机编队在不同威胁分布、任务优先级或感知噪声下保持稳定性能。
- **数据效率提升**：只需少量适配样本即可让策略重新对齐最新态势，适用于实时战场环境。
- **专家经验共存**：保留原有的专家概率融合机制，确保在极端场景下仍有规则兜底，同时元学习带来的快速适配能力能缩小规则与学习策略之间的差距。【F:airbattle/Hierarchical.py†L362-L387】

## 实施建议
- 在实现阶段优先搭建 `MetaReplayBuffer` 与任务采样逻辑，再逐步引入 MAML 风格的内外循环更新，确保每一步都可回退。
- 针对计算压力，可在元训练阶段降低模型隐藏维度或采用一阶近似（FOMAML/Reptile），平衡实时性与性能。
