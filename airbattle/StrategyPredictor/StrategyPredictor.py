import numpy as np
import math

# ====================== 3. 动态贝叶斯网络决策模块 ======================
class StrategyPredictor:
    """
    自定义贝叶斯网络策略预测器 (采用四维决策空间和高级战术知识库)
    """

    def __init__(self):
        self.strategies = ['attack', 'evade', 'hold', 'retreat', 'flank', 'climb', 'dive']
        self.cpds = self._build_custom_bayesian_network()

    def _build_custom_bayesian_network(self):
        """
        构建一个基于“威胁（high/medium/low）、能量（high/medium/low）、距离（close/medium/far）、高度优势（superior/neutral/inferior）”四维空间的、
        经过专家知识重构的条件概率表。
        """
        # --- 全新的、更智能的策略条件概率表 (CPT) ---
        # 格式: (threat, energy, distance, altitude_advantage) -> [attack, evade, hold, retreat, flank, climb, dive]
        strategy_cpd = {
            # === 优势战况 (低威胁, 高能量) ===
            # 我方占据高度优势 ('superior') -> 俯冲攻击 (dive) 是首选
            ('low', 'high', 'close', 'superior'): [0.2, 0.0, 0.1, 0.0, 0.1, 0.0, 0.6],
            ('low', 'high', 'medium', 'superior'): [0.1, 0.0, 0.1, 0.0, 0.2, 0.1, 0.5],
            # 我方高度均势 ('neutral') -> 攻击和侧翼并重
            ('low', 'high', 'close', 'neutral'): [0.6, 0.1, 0.1, 0.0, 0.2, 0.0, 0.0],
            # 我方高度劣势 ('inferior') -> 爬升 (climb) 夺回能量优势
            ('low', 'high', 'far', 'inferior'): [0.1, 0.0, 0.1, 0.0, 0.3, 0.5, 0.0],

            # === 均势战况 (中等威胁, 中等能量) ===
            # 我方占据高度优势 ('superior') -> 依然可以俯冲攻击
            ('medium', 'medium', 'close', 'superior'): [0.3, 0.2, 0.0, 0.0, 0.1, 0.0, 0.4],
            # 我方高度均势 ('neutral') -> 缠斗，攻击、规避、侧翼是主要选择
            ('medium', 'medium', 'medium', 'neutral'): [0.3, 0.3, 0.1, 0.0, 0.3, 0.0, 0.0],
            # 我方高度劣势 ('inferior') -> 必须规避或后撤
            ('medium', 'medium', 'close', 'inferior'): [0.1, 0.5, 0.1, 0.2, 0.1, 0.0, 0.0],

            # === 劣势战况 (高威胁, 低能量) ===
            # 任何高度优势都已无意义，生存是第一要务
            ('high', 'low', 'close', 'superior'): [0.1, 0.6, 0.1, 0.2, 0.0, 0.0, 0.0],  # 规避或后撤
            ('high', 'low', 'close', 'neutral'): [0.1, 0.7, 0.0, 0.2, 0.0, 0.0, 0.0],  # 必须规避
            ('high', 'low', 'close', 'inferior'): [0.0, 0.6, 0.0, 0.4, 0.0, 0.0, 0.0],  # 必须后撤
            ('high', 'low', 'far', 'inferior'): [0.0, 0.2, 0.1, 0.7, 0.0, 0.0, 0.0],  # 果断后撤
        }
        # (为了简洁，这里只列出了部分关键规则，一个完整的CPT会更庞大)
        return {'strategy': strategy_cpd}

    def predict_strategy(self, observation, assigned_target):
        """
        预测最优策略（使用概率性加权选择）。
        """
        features = self._extract_features(observation, assigned_target)
        # 使用新的四维决策键
        key = (features['threat_level'], features['energy_level'], features['distance'], features['altitude_advantage'])

        if key in self.cpds['strategy']:
            strategy_probs = self.cpds['strategy'][key]
            prob_sum = sum(strategy_probs)
            if prob_sum <= 0:
                return np.random.choice(self.strategies)
            normalized_probs = [p / prob_sum for p in strategy_probs]
            return np.random.choice(self.strategies, p=normalized_probs)
        else:
            # 当CPT中没有精确匹配的规则时，执行更智能的默认策略
            if features['threat_level'] == 'high':
                return 'evade' if features['local_superiority'] < 0 else 'retreat'
            elif features['altitude_advantage'] == 'inferior':
                return 'climb'  # 高度劣势时，优先爬升
            elif features['energy_level'] == 'low':
                return 'retreat'  # 能量不足时，优先后撤
            else:
                return 'attack' if features['distance'] == 'close' else 'flank'

    def _extract_features(self, observation, assigned_target):
        """从观测中提取所有决策所需的、包含高度优势的离散化特征"""
        default_features = {
            'threat_level': 'medium', 'energy_level': 'medium',
            'distance': 'medium', 'local_superiority': 0,
            'altitude_advantage': 'neutral'  # 新增默认值
        }

        self_state = observation.get('self_state')
        if not self_state: return default_features

        target_drone = next((e for e in observation['enemies'] if e['id'] == assigned_target), None)
        if not target_drone: return default_features

        self_pos = np.array(self_state['position'])
        target_pos = np.array(target_drone['position'])

        # --- 1. 计算基础特征 ---
        dist = np.linalg.norm(self_pos - target_pos)
        threat = target_drone.get('threat_level', 0.5)
        energy = self_state.get('energy', 1.0)

        # --- 2. 计算新增的“高度优势”特征 ---
        altitude_diff = self_state['altitude'] - target_drone.get('altitude', self_state['altitude'])

        # --- 3. 将所有特征离散化为字符串标签 ---
        if threat < 0.4:
            threat_str = 'low'
        elif threat < 0.7:
            threat_str = 'medium'
        else:
            threat_str = 'high'

        if energy < 0.4:
            energy_str = 'low'
        elif energy < 0.7:
            energy_str = 'medium'
        else:
            energy_str = 'high'

        if dist < 150:
            dist_str = 'close'
        elif dist > 400:
            dist_str = 'far'
        else:
            dist_str = 'medium'

        # 将高度差离散化
        if altitude_diff > 100:
            alt_adv_str = 'superior'  # 高度优势
        elif altitude_diff < -100:
            alt_adv_str = 'inferior'  # 高度劣势
        else:
            alt_adv_str = 'neutral'  # 均势

        # --- 4. 计算关系型特征 (用于默认策略) ---
        radius = 250
        allies_in_radius = 1 + sum(1 for ally in observation.get('allies', []) if
                                   np.linalg.norm(np.array(ally['position']) - self_pos) < radius)
        enemies_in_radius = sum(1 for enemy in observation.get('enemies', []) if
                                np.linalg.norm(np.array(enemy['position']) - self_pos) < radius)

        # --- 5. 整合并返回所有特征 ---
        return {
            'threat_level': threat_str,
            'energy_level': energy_str,
            'distance': dist_str,
            'altitude_advantage': alt_adv_str,  # 返回新的维度
            'local_superiority': allies_in_radius - enemies_in_radius
        }