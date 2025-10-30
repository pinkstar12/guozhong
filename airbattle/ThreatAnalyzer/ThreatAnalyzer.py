import numpy as np
from collections import defaultdict


# ====================== 2. 威胁评估与目标匹配模块 (逻辑修正版) ======================
class ThreatAnalyzer:
    """
    威胁评估与目标分配系统。
    采用经过修正的、更灵敏的威胁计算模型。
    """

    def __init__(self, strategy_space=None):
        # 在纯决策模式下，这个构造函数可以为空
        pass

    def assess_threats(self, observation):
        """
        评估单个己方单位面对的所有敌人的威胁值。
        """
        threat_assessment = {}
        self_state = observation.get('self_state')
        if not self_state:
            return {}

        self_pos = np.array(self_state['position'])
        self_health = self_state.get('health', 1.0)
        self_energy = self_state.get('energy', 1.0)

        for enemy in observation.get('enemies', []):
            enemy_pos = np.array(enemy['position'])
            enemy_health = enemy.get('health', 1.0)

            # --- 1. 基础威胁：基于距离 (采用反比函数，更灵敏) ---
            dist = np.linalg.norm(enemy_pos - self_pos)
            # 距离缩放因子，这个值越小，威胁随距离衰减得越快
            distance_scale_factor = 200.0
            base_threat = 1.0 / (1.0 + dist / distance_scale_factor)

            # --- 2. 威胁调节因子：综合考虑多种态势因素 ---

            # a. 高度差调节：高度差过大时，威胁降低
            height_advantage_factor = 1.0
            height_diff = abs(self_pos[2] - enemy_pos[2])
            if height_diff > 200:
                height_advantage_factor = 0.7
            elif height_diff < 50:
                # 敌人在我方上方时，威胁更大
                if enemy_pos[2] > self_pos[2]:
                    height_advantage_factor = 1.2

            # b. 能量差调节：我方能量低于敌方时，感知到的威胁增加
            energy_factor = 1.0
            # (假设敌方能量可见，如果不可见，可以去掉这个因子)
            enemy_energy = enemy.get('energy', 1.0)
            if self_energy < enemy_energy * 0.8:
                energy_factor = 1.2

            # c. 敌方健康度调节：敌方越健康，威胁越大
            health_factor = 0.5 + enemy_health * 0.5  # 将健康度影响范围缩放到0.5-1.0

            # --- 3. 计算最终威胁得分 ---
            # 最终得分 = 基础威胁 * 所有调节因子的乘积
            final_threat_score = base_threat * height_advantage_factor * energy_factor * health_factor

            # 确保威胁值在0到1之间
            threat_assessment[enemy['id']] = np.clip(final_threat_score, 0, 1)

        # 在循环结束后返回结果
        return threat_assessment

    def assign_targets(self, blue_observations):
        """
        为每个蓝方单位，独立分配对其自身威胁最大的敌方目标。
        """
        if not blue_observations:
            return {}

        assignments = {}
        print("\n--- 开始进行分布式目标威胁评估与分配 ---")
        # 没有敌人被所有蓝方集中攻击，体现了**“分布式”目标分配机制**

        for blue_id, obs in blue_observations.items():
            threat_assessment = self.assess_threats(obs)
            if threat_assessment:
                highest_threat_enemy_id = max(threat_assessment, key=threat_assessment.get)
                assignments[blue_id] = highest_threat_enemy_id
                print(
                    f"单位 {blue_id} 的最高威胁目标是: {highest_threat_enemy_id} (威胁值: {threat_assessment[highest_threat_enemy_id]:.2f})")
            else:
                assignments[blue_id] = None
                print(f"单位 {blue_id} 未发现威胁目标。")

        return assignments