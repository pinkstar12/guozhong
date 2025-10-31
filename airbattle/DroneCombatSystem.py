import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from .BattlefieldEnvironment.BattlefieldEnvironment import BattlefieldEnvironment
from .BattlefieldEnvironment.BattlefieldEnvironment import DroneGNN
from .ThreatAnalyzer.ThreatAnalyzer import ThreatAnalyzer
from .StrategyPredictor.StrategyPredictor import StrategyPredictor

# ====================== 5. 主控制系统 (聚焦DBN决策) ======================
class DroneCombatSystem:
    """
    无人机集群协同博弈主控制系统。
    核心功能：根据战场态势，通过动态贝叶斯网络为每个单位生成决策。
    """

    def __init__(self, red_invasion_mode='concentrated'):
        # 1. 初始化基础模块
        self.environment = BattlefieldEnvironment(
            blue_drones=5,
            red_drones=5,
            red_invasion_mode=red_invasion_mode
        )
        self.threat_analyzer = ThreatAnalyzer(strategy_space=None)

        # 2. 初始化核心决策模块：动态贝叶斯网络
        self.strategy_predictor = StrategyPredictor()

        # 初始化 GNN 模型为空
        self.trained_gnn_model = None

        print("决策系统初始化完成，使用动态贝叶斯网络作为决策核心。")

    def get_strategy_decisions(self):
        """
        获取当前战场态势下，所有己方无人机的策略决策指令集合。
        这是系统对外的主要接口。

        返回:
            set: 一个包含元组(tuple)的集合，每个元组代表一个决策指令，
                 格式为 (无人机ID, 策略, 参数)。
                 例如：{('blue_0', 'attack', 'red_3'), ('blue_1', 'evade', None), ...}
        """

        # 1. 驱动环境运行一步，以产生新的态势
        print("\n--- 驱动战场环境生成新态势 ---")

        # 2. 获取所有己方单位的当前观测数据
        blue_observations = self.environment.get_all_observations('blue')

        if not blue_observations:
            print("警告：战场上已没有己方单位。")
            return set()
        
        
        # =============================cfy：调用信息融合=======================================
        # 🔵 信息融合：通过GATv2为每架无人机生成融合后的状态向量
        print("\n--- 进行图神经网络信息融合 ---")
        fused_observations, updated_model = DroneGNN.fuse_with_gnn(
            blue_observations,
            strategy_predictor=self.strategy_predictor,
            pretrained_model=self.trained_gnn_model
        )
        self.trained_gnn_model = updated_model  # 保存训练好的模型供下一轮复用
        if fused_observations:
            blue_observations = fused_observations
        # ====================================================================================
    

        # 3. 进行目标分配
        target_assignments = self.threat_analyzer.assign_targets(blue_observations)
        print(f"目标分配结果: {target_assignments}")

        # 4. 遍历每个己方单位，使用DBN进行决策推理并生成指令
        final_decisions = set()
        print("\n--- 开始使用动态贝叶斯网络进行决策推理 ---")
        for blue_id, obs in blue_observations.items():
            assigned_target = target_assignments.get(blue_id)

            # a. 调用策略预测器（DBN核心）来推演策略
            predicted_strategy = self.strategy_predictor.predict_strategy(obs, assigned_target)

            # --- 核心修改：根据策略构建包含参数的决策指令 ---

            # b. 定义哪些策略是需要目标的
            strategies_requiring_target = {'attack', 'flank', 'high_cover', 'low_cover'}

            # c. 生成最终的决策元组 (drone_id, strategy, params)
            if predicted_strategy in strategies_requiring_target:
                # 如果策略需要目标，参数就是分配到的目标ID
                params = assigned_target
                decision = (blue_id, predicted_strategy, params)
                print(f"单位 {blue_id} 的决策: {predicted_strategy}, 目标: {params}")
            else:
                # 如果策略不需要目标，参数为 None
                params = None
                decision = (blue_id, predicted_strategy, params)
                print(f"单位 {blue_id} 的决策: {predicted_strategy}")

            final_decisions.add(decision)

        return final_decisions

