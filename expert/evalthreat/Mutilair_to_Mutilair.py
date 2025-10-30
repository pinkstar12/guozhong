import numpy as np
from expert.evalthreat.AirToAirThreatModels.BaseAttackArea import BaseAttackAreaThreatAnalysis as SingleThreatAnalysis

class ThreatModel:
    def __init__(self, our_params=None, enemy_params=None, speed_params=None, **kwargs):
        # 整理成与 initialize(config) 一致的 dict
        config = {
        'our_params': our_params or {},
        'enemy_params': enemy_params or {},
        'speed_params': speed_params or {},
        }
        self.num_our = None
        self.situation_matrix = None  # 新增：存储态势矩阵 (友机数 × 敌机数)
        self.last_state = None  # 新增：存储最后一次处理的状态

        # 初始化单对单威胁分析模块
        self.single_threat_analyzer = SingleThreatAnalysis(
            our_params=our_params,
            enemy_params=enemy_params,
            speed_params=speed_params,
            **kwargs
        )
        
        self.initialize(config)

    def initialize(self, config):
        """
        从配置初始化态势评估模块
        :param config: 配置字典，包含角度、距离、速度参数和威胁权重函数
        """
        # 将配置传递给单对单分析模块
        self.single_threat_analyzer.initialize(config)

    def threat_weights(self, our_aircrafts_params, enemy):
        """
        考虑团队协作的威胁权重计算
        :param our_aircrafts_params: 我方飞机参数列表
        :param enemy: 单个敌机参数字典
        :return: 威胁权重值
        """
        # 1. 距离因素：最近的我机距离
        distances = []
        for our in our_aircrafts_params:
            dist = np.linalg.norm(our['position'] - enemy['position'])
            distances.append(dist)
        
        min_distance = min(distances)
        distance_threat = 1 / (min_distance + 1e-5)

        # 2. 角度因素：最佳攻击角度的我机数量
        good_angle_count = 0
        for our in our_aircrafts_params:
            enemy_pos = enemy['position']
            our_pos = our['position']
            dx = enemy_pos[0] - our_pos[0]
            dy = enemy_pos[1] - our_pos[1]
            target_azimuth = np.arctan2(dy, dx) * 180 / np.pi
            phi = abs(our['heading'] - target_azimuth) % 360
            phi = phi if phi <= 180 else 360 - phi

            if phi < 30:  # 良好攻击角度
                good_angle_count += 1

        angle_threat = good_angle_count * 0.5

        # 3. 高度优势
        height_threat = 0
        for our in our_aircrafts_params:
            if our['height'] > enemy['height'] + 1000:
                height_threat += 0.2

        # 4. 武器状态（简化示例）
        weapon_threat = 1.0  # 假设所有飞机都有武器

        # 综合威胁
        total_threat = (
                0.5 * distance_threat +
                0.3 * angle_threat +
                0.1 * height_threat +
                0.1 * weapon_threat
        )

        return total_threat

    def process_state(self, state):
        """
        处理状态数据并计算态势值
        :param state: 全局状态字典，格式为：
        {
            'our_aircrafts': [{'speed': float, 'height': float, 'heading': float, 'position': np.array}, ...],
            'enemies': [{'speed': float, 'height': float, 'heading': float, 'position': np.array}, ...]
        }
        :return: 团队优势值, 团队风险值
        """
        enemies = state['enemies']
        our_aircrafts = state['our_aircrafts']
        self.num_our = len(our_aircrafts)
        self.last_state = state  # 保存状态供新功能使用

        # 初始化态势矩阵 (友机数 × 敌机数)
        num_enemies = len(enemies)
        self.situation_matrix = np.zeros((self.num_our, num_enemies))

        # 计算每架敌机的全局威胁权重
        global_threats = [self.threat_weights(our_aircrafts, enemy) for enemy in enemies]
        total_global_threat = sum(global_threats) if sum(global_threats) > 0 else 1e-5
        global_weights = [t / total_global_threat for t in global_threats]

        # 计算协同分数
        coordination_score = self.calculate_coordination(our_aircrafts, enemies)

        team_advantage = 0.0
        team_risk = 0.0

        for i, our_params in enumerate(our_aircrafts):
            aircraft_advantage = 0.0
            aircraft_risk = 0.0

            for j, enemy in enumerate(enemies):
                # 使用单对单威胁分析模块计算X值
                X_ij = self.single_threat_analyzer.calculate_single_threat(our_params, enemy)
                
                # 存储到态势矩阵
                self.situation_matrix[i, j] = X_ij

                # 使用混合权重（全局+局部）
                local_threat = self.threat_weights([our_params], enemy)
                total_local = sum(self.threat_weights([our_params], e) for e in enemies)
                local_weights = local_threat / total_local if total_local > 0 else 0

                mixed_weight = 0.6 * global_weights[j] + 0.4 * local_weights

                # 累积优势值和风险值
                aircraft_advantage += mixed_weight * X_ij
                aircraft_risk += mixed_weight * (1 - X_ij)

            # 考虑协同效应调整单机优势
            adjusted_advantage = aircraft_advantage * coordination_score[i]
            team_advantage += adjusted_advantage
            team_risk += aircraft_risk

        # 归一化
        if self.num_our > 0:
            team_advantage /= self.num_our
            team_risk /= (self.num_our * len(enemies)) if len(enemies) > 0 else 1

        return team_advantage, team_risk

    def evaluate_situation(self, advantage, risk):
        """
        评估当前战场态势
        :param advantage: 团队优势值 (0-1)
        :param risk: 团队风险值 (0-1)
        :return: "advantage", "disadvantage" 或 "neutral"
        """
        #态势判断逻辑
        if advantage > 0.55 and risk < 0.45:
            return "advantage"
        elif advantage < 0.15 and risk > 0.75:
            return "disadvantage"
        return "neutral"

    def calculate_coordination(self, our_aircrafts, enemies):
        """
        计算我方飞机之间的协同分数
        :param our_aircrafts: 我方飞机参数列表
        :param enemies: 敌方飞机参数列表
        :return: 每架飞机的协同分数列表
        """
        scores = []
        positions = [ac['position'] for ac in our_aircrafts]
        headings = [ac['heading'] for ac in our_aircrafts]
        num_our = len(our_aircrafts)

        if num_our <= 1:
            return [1.0] * num_our

        # 计算飞机间距离矩阵
        dist_matrix = np.zeros((num_our, num_our))
        for i in range(num_our):
            for j in range(i + 1, num_our):
                # 确保positions是numpy数组
                pos_i = np.array(positions[i]) if not isinstance(positions[i], np.ndarray) else positions[i]
                pos_j = np.array(positions[j]) if not isinstance(positions[j], np.ndarray) else positions[j]
                dist = np.linalg.norm(pos_i - pos_j)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # 计算每架飞机的协同分数
        for i in range(num_our):
            # 1. 距离协同：与最近友机的距离
            other_indices = [idx for idx in range(num_our) if idx != i]
            if other_indices:
                min_dist = np.min(dist_matrix[i, other_indices])
                dist_score = 1.0 if 10 <= min_dist <= 20 else np.exp(-0.1 * abs(min_dist - 15))
            else:
                dist_score = 1.0

            # 2. 角度协同：与友机航向差
            heading_diffs = []
            for j in range(num_our):
                if i != j:
                    angle_diff = abs(headings[i] - headings[j]) % 360
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    heading_diffs.append(angle_diff)

            if heading_diffs:
                avg_angle_diff = np.mean(heading_diffs)
                angle_score = np.exp(-0.01 * avg_angle_diff ** 2)
            else:
                angle_score = 1.0

            # 3. 目标覆盖：攻击不同目标
            target_coverage = min(1.0, len(enemies) / num_our) if len(enemies) > 0 else 1.0

            # 综合协同分数
            coord_score = 0.4 * dist_score + 0.4 * angle_score + 0.2 * target_coverage
            scores.append(coord_score)

        return scores

    def get_detailed_analysis(self, state):
        """
        获取详细的威胁分析结果
        :param state: 全局状态字典
        :return: 详细分析结果字典
        """
        enemies = state['enemies']
        our_aircrafts = state['our_aircrafts']
        
        detailed_results = {
            'single_to_single_analysis': [],
            'team_coordination': self.calculate_coordination(our_aircrafts, enemies),
            'global_threat_weights': []
        }
        
        # 计算每对我机-敌机的详细分析
        for i, our_aircraft in enumerate(our_aircrafts):
            aircraft_analysis = {
                'aircraft_id': i,
                'threats_to_enemies': []
            }
            
            for j, enemy in enumerate(enemies):
                # 使用单对单分析模块获取详细的分量分析
                situation_components = self.single_threat_analyzer.calculate_situation_components(
                    our_aircraft, enemy
                )
                
                threat_detail = {
                    'enemy_id': j,
                    'situation_components': situation_components,
                    'threat_weight': self.threat_weights([our_aircraft], enemy)
                }
                aircraft_analysis['threats_to_enemies'].append(threat_detail)
            
            detailed_results['single_to_single_analysis'].append(aircraft_analysis)
        
        # 计算全局威胁权重
        global_threats = [self.threat_weights(our_aircrafts, enemy) for enemy in enemies]
        total_global_threat = sum(global_threats) if sum(global_threats) > 0 else 1e-5
        detailed_results['global_threat_weights'] = [t / total_global_threat for t in global_threats]
        
        return detailed_results

    def calculate_threat(self, state):
        """
        兼容性接口：计算威胁态势
        :param state: 全局状态字典
        :return: 团队优势值, 团队风险值
        """
        return self.process_state(state)

    # ===== 新增功能函数 =====

    def get_friendly_situations(self, advantage_threshold=0.55, disadvantage_threshold=0.25):
        """
        获取各友方飞机的态势信息（优势、等势、劣势）
        
        :param advantage_threshold: float - 优势阈值，默认0.55
        :param disadvantage_threshold: float - 劣势阈值，默认0.45
        :return: list[str] - 态势信息列表，每个元素为'advantage'、'neutral'或'disadvantage'
        """
        if self.situation_matrix is None:
            return []
        
        situations = []
        for i in range(self.situation_matrix.shape[0]):
            # 计算该友机对所有敌机的平均态势值
            avg_situation = np.mean(self.situation_matrix[i, :])
            
            if avg_situation > advantage_threshold:
                situations.append("advantage")
            elif avg_situation < disadvantage_threshold:
                situations.append("disadvantage")
            else:
                situations.append("neutral")
        
        return situations

    def get_friendly_threat_ranking(self):
        """
        获取各友方飞机的威胁排名列表
        
        :return: list[list[tuple]] - 威胁排名列表，每个元素是[(敌机索引, 威胁值), ...]，按威胁值降序排列
        """
        if self.situation_matrix is None:
            return []
        
        rankings = []
        for i in range(self.situation_matrix.shape[0]):
            # 计算威胁值（1 - 态势值，态势值越低威胁越大）
            threat_values = 1 - self.situation_matrix[i, :]
            
            # 创建(敌机索引, 威胁值)的元组列表
            indexed_threats = [(j, threat) for j, threat in enumerate(threat_values)]
            
            # 按威胁值降序排序
            indexed_threats.sort(key=lambda x: x[1], reverse=True)
            
            rankings.append(indexed_threats)
        
        return rankings

    def get_max_threat_enemy_per_friendly(self):
        """
        获取各友方飞机面临的威胁最高的敌方飞机索引
        
        :return: list[int] - 敌机索引列表，每个元素对应一个友方飞机面临的最大威胁敌机
        """
        if self.situation_matrix is None:
            return []
        
        max_threat_indices = []
        for i in range(self.situation_matrix.shape[0]):
            # 计算威胁值（1 - 态势值）
            threat_values = 1 - self.situation_matrix[i, :]
            
            # 找到最大威胁值对应的敌机索引
            max_index = np.argmax(threat_values)
            max_threat_indices.append(int(max_index))
        
        return max_threat_indices

    def get_optimal_target_per_friendly(self):
        """
        获取各友方飞机对敌态势最优的敌方飞机索引
        
        :return: list[int] - 敌机索引列表，每个元素对应一个友方飞机的最佳攻击目标
        """
        if self.situation_matrix is None:
            return []
        
        optimal_targets = []
        for i in range(self.situation_matrix.shape[0]):
            # 找到态势值最大的敌机索引（友军优势最大）
            max_index = np.argmax(self.situation_matrix[i, :])
            optimal_targets.append(int(max_index))
        
        return optimal_targets

    def get_overall_situation(self):
        """
        获取友军整体态势值（优势值考虑协同效应）

        :return: tuple[float, float] - (团队优势值, 团队风险值)
        """
        if self.situation_matrix is None or self.last_state is None:
            return 0.0, 0.0

        enemies = self.last_state['enemies']
        our_aircrafts = self.last_state['our_aircrafts']

        # 计算协同分数
        coordination_score = self.calculate_coordination(our_aircrafts, enemies)

        # 计算每架友机的威胁权重（与process_state中的逻辑一致）
        global_threats = [self.threat_weights(our_aircrafts, enemy) for enemy in enemies]
        total_global_threat = sum(global_threats) if sum(global_threats) > 0 else 1e-5
        global_weights = [t / total_global_threat for t in global_threats]

        team_advantage = 0.0
        team_risk = 0.0

        for i in range(self.situation_matrix.shape[0]):
            aircraft_advantage = 0.0
            aircraft_risk = 0.0

            # 重新计算每架友机的优势值和风险值（考虑混合权重）
            for j, enemy in enumerate(enemies):
                X_ij = self.situation_matrix[i, j]

                # 计算混合权重（与process_state逻辑一致）
                local_threat = self.threat_weights([our_aircrafts[i]], enemy)
                total_local = sum(self.threat_weights([our_aircrafts[i]], e) for e in enemies)
                local_weights = local_threat / total_local if total_local > 0 else 0

                mixed_weight = 0.6 * global_weights[j] + 0.4 * local_weights

                aircraft_advantage += mixed_weight * X_ij
                aircraft_risk += mixed_weight * (1 - X_ij)

            # 考虑协同效应：只调整优势值，风险值不变
            adjusted_advantage = aircraft_advantage * coordination_score[i]

            team_advantage += adjusted_advantage
            team_risk += aircraft_risk  # 风险值不乘协同分数

        # 归一化
        if self.num_our > 0:
            team_advantage /= self.num_our
            team_risk /= (self.num_our * len(enemies)) if len(enemies) > 0 else 1

        return float(team_advantage), float(team_risk)

    def get_friendly_situation_details(self):
        """
        获取友方飞机详细态势信息
        
        :return: list[dict] - 详细态势信息列表，每个元素包含友机的完整态势分析
        """
        if self.situation_matrix is None:
            return []
        
        details = []
        situations = self.get_friendly_situations()
        threat_rankings = self.get_friendly_threat_ranking()
        max_threats = self.get_max_threat_enemy_per_friendly()
        optimal_targets = self.get_optimal_target_per_friendly()
        
        for i in range(self.situation_matrix.shape[0]):
            detail = {
                'aircraft_id': i,
                'situation_status': situations[i],
                'average_situation_value': float(np.mean(self.situation_matrix[i, :])),
                'threat_ranking': threat_rankings[i],
                'max_threat_enemy_index': max_threats[i],
                'max_threat_value': float(1 - np.min(self.situation_matrix[i, :])),
                'optimal_target_index': optimal_targets[i],
                'optimal_target_advantage': float(np.max(self.situation_matrix[i, :])),
                'situation_values_to_enemies': [float(val) for val in self.situation_matrix[i, :]]
            }
            details.append(detail)
        
        return details


# ===== 模块级便捷函数 =====

def get_friendly_situations_from_state(state, model_or_params=None, advantage_threshold=0.55, disadvantage_threshold=0.45):
    """
    便捷函数：从状态直接获取友方飞机态势信息
    
    :param state: 全局状态字典
    :param model_params: 模型参数字典，可选
    :param advantage_threshold: 优势阈值，默认0.55
    :param disadvantage_threshold: 劣势阈值，默认0.45
    :return: list[str] - 态势信息列表
    """
    if isinstance(model_or_params, ThreatModel):
        # 如果传入的是 ThreatModel 实例，直接使用
        model = model_or_params
    else:
        # 如果传入的是参数字典（或 None），创建新实例
        model = ThreatModel(**(model_or_params or {}))
    model.process_state(state)
    return model.get_friendly_situations(advantage_threshold, disadvantage_threshold)


def get_friendly_threat_rankings_from_state(state, model_or_params=None):
    """
    便捷函数：从状态直接获取友方飞机威胁排名
    
    :param state: 全局状态字典
    :param model_params: 模型参数字典，可选
    :return: list[list[tuple]] - 威胁排名列表
    """
    if isinstance(model_or_params, ThreatModel):
        # 如果传入的是 ThreatModel 实例，直接使用
        model = model_or_params
    else:
        # 如果传入的是参数字典（或 None），创建新实例
        model = ThreatModel(**(model_or_params or {}))
    model.process_state(state)
    return model.get_friendly_threat_ranking()


def get_max_threats_from_state(state, model_or_params=None):
    """
    便捷函数：从状态直接获取各友机面临的最大威胁敌机索引
    
    :param state: 全局状态字典
    :param model_params: 模型参数字典，可选
    :return: list[int] - 最大威胁敌机索引列表
    """
    if isinstance(model_or_params, ThreatModel):
        # 如果传入的是 ThreatModel 实例，直接使用
        model = model_or_params
    else:
        # 如果传入的是参数字典（或 None），创建新实例
        model = ThreatModel(**(model_or_params or {}))
    model.process_state(state)
    return model.get_max_threat_enemy_per_friendly()


def get_optimal_targets_from_state(state,model_or_params=None):
    """
    便捷函数：从状态直接获取各友机的最佳攻击目标索引
    
    :param state: 全局状态字典
    :param model_params: 模型参数字典，可选
    :return: list[int] - 最佳攻击目标索引列表
    """
    if isinstance(model_or_params, ThreatModel):
        # 如果传入的是 ThreatModel 实例，直接使用
        model = model_or_params
    else:
        # 如果传入的是参数字典（或 None），创建新实例
        model = ThreatModel(**(model_or_params or {}))
    model.process_state(state)
    return model.get_optimal_target_per_friendly()


def get_overall_situation_from_state(state, model_or_params=None):
    """
    便捷函数：从状态直接获取友军整体态势值（优势值考虑协同效应）

    :param state: 全局状态字典
    :param model_or_params: 模型参数字典，可选
    :return: tuple[float, float] - (团队优势值, 团队风险值)
    """
    if isinstance(model_or_params, ThreatModel):
        model = model_or_params
    else:
        model = ThreatModel(**(model_or_params or {}))

    enemies = state['enemies']
    our_aircrafts = state['our_aircrafts']
    num_our = len(our_aircrafts)
    num_enemies = len(enemies)

    if num_our == 0 or num_enemies == 0:
        return 0.0, 0.0

    # 计算协同分数
    coordination_score = model.calculate_coordination(our_aircrafts, enemies)

    # 计算全局威胁权重
    global_threats = [model.threat_weights(our_aircrafts, enemy) for enemy in enemies]
    total_global_threat = sum(global_threats) if sum(global_threats) > 0 else 1e-5
    global_weights = [t / total_global_threat for t in global_threats]

    team_advantage = 0.0
    team_risk = 0.0

    for i, our_params in enumerate(our_aircrafts):
        aircraft_advantage = 0.0
        aircraft_risk = 0.0

        for j, enemy in enumerate(enemies):
            # 计算单对单威胁值
            X_ij = model.single_threat_analyzer.calculate_single_threat(our_params, enemy)

            # 计算混合权重
            local_threat = model.threat_weights([our_params], enemy)
            total_local = sum(model.threat_weights([our_params], e) for e in enemies)
            local_weights = local_threat / total_local if total_local > 0 else 0

            mixed_weight = 0.6 * global_weights[j] + 0.4 * local_weights

            # 累积优势值和风险值
            aircraft_advantage += mixed_weight * X_ij
            aircraft_risk += mixed_weight * (1 - X_ij)

        # 考虑协同效应：只调整优势值，风险值不变
        adjusted_advantage = aircraft_advantage * coordination_score[i]

        team_advantage += adjusted_advantage
        team_risk += aircraft_risk  # 风险值不乘协同分数

    # 归一化
    team_advantage /= num_our
    team_risk /= (num_our * num_enemies)

    return float(team_advantage), float(team_risk)


def get_complete_analysis_from_state(state, model_or_params=None):
    """
    便捷函数：从状态直接获取完整的态势分析
    
    :param state: 全局状态字典
    :param model_params: 模型参数字典，可选
    :return: dict - 包含所有分析结果的字典
    """
    if isinstance(model_or_params, ThreatModel):
        # 如果传入的是 ThreatModel 实例，直接使用
        model = model_or_params
    else:
        # 如果传入的是参数字典（或 None），创建新实例
        model = ThreatModel(**(model_or_params or {}))
    team_advantage, team_risk = model.process_state(state)
    
    analysis = {
        'team_advantage': team_advantage,
        'team_risk': team_risk,
        'team_situation': model.evaluate_situation(team_advantage, team_risk),
        'friendly_situations': model.get_friendly_situations(),
        'friendly_threat_rankings': model.get_friendly_threat_ranking(),
        'max_threat_enemies': model.get_max_threat_enemy_per_friendly(),
        'optimal_targets': model.get_optimal_target_per_friendly(),
        'overall_situation': model.get_overall_situation(),
        'detailed_analysis': model.get_friendly_situation_details()
    }
    
    return analysis


# ===== 示例使用 =====
if __name__ == "__main__":
    # 示例数据
    state = {
        'our_aircrafts': [
            {
                'speed': 300,
                'height': 8000,
                'heading': 90,
                'position': np.array([0, 0])
            },
            {
                'speed': 320,
                'height': 7500,
                'heading': 85,
                'position': np.array([5000, 1000])
            }
        ],
        'enemies': [
            {
                'speed': 280,
                'height': 7000,
                'heading': 270,
                'position': np.array([10000, 2000])
            },
            {
                'speed': 290,
                'height': 8500,
                'heading': 260,
                'position': np.array([12000, -1000])
            },
            {
                'speed': 310,
                'height': 6500,
                'heading': 280,
                'position': np.array([8000, 3000])
            }
        ]
    }
    
    # 创建威胁模型
    model = ThreatModel()
    
    # 处理状态
    team_advantage, team_risk = model.process_state(state)
    print(f"团队优势值: {team_advantage:.4f}, 团队风险值: {team_risk:.4f}")
    
    # 使用新功能
    print("\n=== 新功能演示 ===")
    
    # 1. 获取友方飞机态势信息
    situations = model.get_friendly_situations()
    print(f"友方飞机态势: {situations}")
    
    # 2. 获取威胁排名
    threat_rankings = model.get_friendly_threat_ranking()
    print(f"威胁排名:")
    for i, ranking in enumerate(threat_rankings):
        print(f"  友机{i}: {ranking}")
    
    # 3. 获取最大威胁敌机索引
    max_threats = model.get_max_threat_enemy_per_friendly()
    print(f"最大威胁敌机索引: {max_threats}")
    
    # 4. 获取最佳攻击目标索引
    optimal_targets = model.get_optimal_target_per_friendly()
    print(f"最佳攻击目标索引: {optimal_targets}")
    
    # 5. 获取整体态势值
    overall_adv, overall_risk = model.get_overall_situation()
    print(f"整体态势: 优势值={overall_adv:.4f}, 风险值={overall_risk:.4f}")
    
    # 使用便捷函数
    print("\n=== 便捷函数演示 ===")
    
    situations_direct = get_friendly_situations_from_state(state)
    print(f"直接获取友方态势: {situations_direct}")
    
    complete_analysis = get_complete_analysis_from_state(state)
    print(f"完整分析键: {list(complete_analysis.keys())}")
