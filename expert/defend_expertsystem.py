import yaml
import numpy as np
import copy
import time
from common.registry import build
from expert.evalthreat.SingleAir_to_MulitMissile import SingleAirToMultiMissile
from expert.optimize.DE import DEOptimization
from expert.behavior_tree.nodes import SequenceNode, ActionNode, NodeStatus, SelectorNode

# 默认决策标志字典
DEFAULT_DECISION_FLAGS = {
    'use_defense_decision': 0,
    'use_attack_decision': 0,
    'use_support_decision': 0,
    'use_radar_decision': 1  # 保留字段
}

def convert_units_to_meters(obs):
    """
    将输入数据的位置单位从千米转换为米，供底层ClassicMethod.py使用
    
    参数:
        obs: 观测数据字典，包含our_aircrafts、enemies、missiles等
    
    返回:
        转换后的观测数据字典（位置单位为米）
    """
    converted_obs = copy.deepcopy(obs)
    
    # 转换我方飞机位置（千米 → 米）
    if 'our_aircrafts' in converted_obs:
        for aircraft in converted_obs['our_aircrafts']:
            if 'position' in aircraft:
                aircraft['position'] = [coord * 1000 for coord in aircraft['position']]
    
    # 转换敌方单位位置（千米 → 米）
    if 'enemies' in converted_obs:
        for enemy in converted_obs['enemies']:
            if 'position' in enemy:
                enemy['position'] = [coord * 1000 for coord in enemy['position']]
    
    # 转换导弹位置（千米 → 米）
    if 'missiles' in converted_obs:
        for missile in converted_obs['missiles']:
            if 'position' in missile:
                missile['position'] = [coord * 1000 for coord in missile['position']]
    
    return converted_obs

def convert_action_to_kilometers(action):
    """
    将输出动作的位置单位从米转换为千米，保持与输入数据单位一致
    
    参数:
        action: 动作字典，包含position等字段
    
    返回:
        转换后的动作字典（位置单位为千米）
    """
    if not isinstance(action, dict):
        return action
    
    converted_action = copy.deepcopy(action)
    
    # 转换位置（米 → 千米）
    if 'position' in converted_action:
        converted_action['position'] = [coord / 1000 for coord in converted_action['position']]
    
    return converted_action

def convert_state_units_to_meters(state):
    """
    将状态数据的位置单位从千米转换为米
    
    参数:
        state: 状态字典，包含aircraft、missiles等
    
    返回:
        转换后的状态字典（位置单位为米）
    """
    converted_state = copy.deepcopy(state)
    
    # 转换飞机位置（千米 → 米）
    if 'aircraft' in converted_state and 'position' in converted_state['aircraft']:
        converted_state['aircraft']['position'] = [coord * 1000 for coord in converted_state['aircraft']['position']]

        
    # 转换敌机位置（千米 → 米）
    if 'aircraft' in converted_state and 'assigned_enemy' in converted_state['aircraft'] and converted_state['aircraft']['assigned_enemy'] is not None:
        converted_state['aircraft']['assigned_enemy'] = [coord * 1000 for coord in converted_state['aircraft']['assigned_enemy']]
    # 转换导弹位置（千米 → 米）
    if 'missiles' in converted_state:
        for missile in converted_state['missiles']:
            if 'position' in missile:
                missile['position'] = [coord * 1000 for coord in missile['position']]
    
    return converted_state

class DefendExpertSystem:
    """
    飞机威胁规避专家系统（支持DE优化和行为树模式）
    """
    
    def __init__(self, threat_model=None, optimization_module=None, selection=None, mode="de", **kwargs):
        """
        初始化专家系统
        
        参数:
            threat_model: 威胁评估模型实例
            optimization_module: DE优化模块实例
            selection: 选择策略配置
            mode: 模式选择 ("de": DE优化模式, "bt": 行为树模式)
            kwargs: 其他配置参数
        """
        print(f"初始化防御专家系统（模式: {mode}）...")
        print("威胁评估模型:", threat_model, type(threat_model))
        print("优化模块:", optimization_module, type(optimization_module))
        
        self.threat_model = threat_model or SingleAirToMultiMissile()
        self.optimization_module = optimization_module
        self.selection_config = selection or {}
        self.selection_strategy = self.selection_config.get('strategy', 'best')
        self.mode = mode
        
        # DE优化配置
        self.de_config = kwargs.get('de_config', {})
        self.verbose = kwargs.get('verbose', False)
        
        # 行为树配置
        self.bt_config = kwargs.get('bt_config', {})
        
        # 威胁度历史记录（用于计算威胁度变化趋势）
        self.threat_history = {}
        
        # 导弹历史位置跟踪（用于计算导弹真实heading）
        self.missile_history = {}  # 格式: {missile_id: [position_history]}
        self.max_history_length = 5  # 保留最近5个位置点
    
    def process(self, state):
        """
        处理状态并生成优化的飞机动作
        
        参数:
            state: 当前环境状态字典，包含:
                - aircraft: 飞机状态字典（位置单位：千米）
                - missiles: 导弹列表（位置单位：千米）
        
        返回:
            字典格式的结果，包含:
                - action: 优化后的飞机动作（位置单位：千米）
                - explanation: 决策解释 (仅行为树模式)
        """
        if self.verbose:
            print(f"处理状态（原始输入单位 - 位置：千米）: {state}")
        
        # 验证输入状态
        if not self._validate_state(state):
            raise ValueError("输入状态格式不正确")
        
        # 将输入数据单位转换为米（供威胁评估使用）
        converted_state = convert_state_units_to_meters(state)
        
        if self.verbose:
            print(f"单位转换后状态（位置：米）: {converted_state}")
        
        # 提取转换后的飞机和导弹信息
        aircraft = converted_state['aircraft']
        missiles = converted_state['missiles']
        
        # 更新导弹历史位置
        self._update_missile_history(missiles)
        
        if not missiles:
            if self.verbose:
                print("没有导弹威胁，返回原始飞机状态")
            if self.mode == "bt":
                return {
                    "action": state['aircraft'],  # 返回原始单位（千米）
                    "explanation": "当前无导弹威胁，保持原有飞行状态"
                }
            return state['aircraft']  # 返回原始单位（千米）
        
        # 根据模式选择处理方式
        if self.mode == "bt":
            # 行为树模式
            result = self._process_with_behavior_tree(converted_state)
            # 将输出动作从米转换回千米
            if isinstance(result, dict) and 'action' in result:
                result['action'] = convert_action_to_kilometers(result['action'])
            return result
        else:
            # DE优化模式
            if self.optimization_module:
                # 使用配置的优化模块
                optimized_actions = self._optimize_with_module(converted_state)
            else:
                # 内置DE优化已被移除，返回原始飞机状态
                if self.verbose:
                    print("内置DE优化已被移除，返回原始飞机状态")
                aircraft_with_flags = aircraft.copy()
                aircraft_with_flags.update({
                    "use_defense_decision": 1,
                    "use_attack_decision": 0,
                    "use_support_decision": 0,
                    'use_radar_decision': 1
                })
                optimized_actions = aircraft_with_flags
            
            # 将输出动作从米转换回千米
            return convert_action_to_kilometers(optimized_actions)
    
    def _validate_state(self, state):
        """验证状态格式"""
        required_keys = ['aircraft', 'missiles']
        if not all(key in state for key in required_keys):
            return False
        
        aircraft = state['aircraft']
        required_aircraft_keys = ['position', 'height', 'speed']
        if not all(key in aircraft for key in required_aircraft_keys):
            return False
        
        return True
    
    def _optimize_with_module(self, state):
        """使用配置的优化模块进行优化"""
        # 转换状态格式以兼容优化模块
        converted_state = {
            'our_aircrafts': [state['aircraft']],
            'missiles': state['missiles']
        }
        
        # 执行优化
        pareto_set = self.optimization_module.process_state(converted_state)
        
        # 生成优化解释
        if hasattr(self.optimization_module, 'generate_explanation'):
            # 构造适合解释生成器的结果字典
            explanation_result = {
                'success': True,
                'pareto_set': pareto_set,
                'best_solution': pareto_set[0] if pareto_set else None,
                'optimization_type': 'DE'
            }
            self.last_optimization_explanation = self.optimization_module.generate_explanation(
                explanation_result, converted_state
            )
        else:
            self.last_optimization_explanation = {
                'text_explanation': '优化模块不支持解释生成',
                'optimization_type': 'Unknown'
            }
        
        if not pareto_set:
            # 为原始状态添加决策标志
            original_aircraft = state['aircraft'].copy()
            original_aircraft.update({
                "use_defense_decision": 1,
                "use_attack_decision": 0,
                "use_support_decision": 0,
                'use_radar_decision': 1
            })
            return original_aircraft
        
        # 选择最优解并添加决策标志
        optimized_action = self._select_solution(pareto_set)
        if isinstance(optimized_action, dict):
            optimized_action.update({
                "use_defense_decision": 1,
                "use_attack_decision": 0,
                "use_support_decision": 0,
                'use_radar_decision': 1
            })
        
        return optimized_action
    
    
    def _select_solution(self, pareto_set):
        """从帕累托解集中选择解"""
        if not pareto_set:
            return None
        
        if self.selection_strategy == 'best':
            # 选择威胁度最低的解
            best_idx = 0
            best_threat = pareto_set[0][1][0]  # 第一个目标是威胁度
            
            for i, (actions, objectives) in enumerate(pareto_set):
                if objectives[0] < best_threat:
                    best_threat = objectives[0]
                    best_idx = i
            
            return pareto_set[best_idx][0][0]  # 返回第一架飞机的动作
        
        elif self.selection_strategy == 'first':
            return pareto_set[0][0][0]
        
        else:
            # 默认选择第一个解
            return pareto_set[0][0][0]
    
    def evaluate_current_threat(self, aircraft, missiles):
        """评估当前威胁度"""
        if not missiles:
            return 0.0
        
        threats = self.threat_model.evaluate_threats(aircraft, missiles)
        return max(threats) if threats else 0.0
    
    def get_threat_analysis(self, aircraft, missiles):
        """获取详细威胁分析"""
        if not missiles:
            return {
                'total_missiles': 0,
                'max_threat': 0.0,
                'threat_list': [],
                'high_threat_missiles': []
            }
        
        threats = self.threat_model.evaluate_threats(aircraft, missiles)
        
        analysis = {
            'total_missiles': len(missiles),
            'max_threat': max(threats) if threats else 0.0,
            'threat_list': threats,
            'threat_rankings': self.threat_model.get_threat_rankings(aircraft, missiles),
            'threat_levels': self.threat_model.get_threat_levels(aircraft, missiles),
            'high_threat_missiles': self.threat_model.filter_high_threats(aircraft, missiles, threshold=0.3)
        }
        
        return analysis
    
    def _calculate_threat_trend(self, missile_id, current_threat):
        """
        计算导弹威胁度变化率
        
        参数:
            missile_id: 导弹ID（使用导弹索引作为ID）
            current_threat: 当前威胁度
        
        返回:
            float: 威胁度变化率
        """
        if missile_id not in self.threat_history:
            self.threat_history[missile_id] = []
        
        # 保留最近3次记录用于计算趋势
        self.threat_history[missile_id].append(current_threat)
        if len(self.threat_history[missile_id]) > 3:
            self.threat_history[missile_id].pop(0)
        
        # 计算变化率（最近两次变化的平均值）
        if len(self.threat_history[missile_id]) >= 2:
            changes = [
                self.threat_history[missile_id][i] - self.threat_history[missile_id][i-1]
                for i in range(1, len(self.threat_history[missile_id]))
            ]
            return sum(changes) / len(changes)
        
        return 0.0
    
    def _should_intercept(self, missile_idx, current_threat, aircraft, missile):
        """
        判断是否应该进行拦截（优化版）
        
        参数:
            missile_idx: 导弹索引
            current_threat: 当前威胁度
            aircraft: 飞机状态字典
            missile: 导弹状态字典
        
        返回:
            bool: 是否应该拦截
        """
        # 计算威胁度变化趋势
        threat_trend = self._calculate_threat_trend(missile_idx, current_threat)

        # 新增条件1: 判断剩余拦截弹个数
        intercept_launch_condition =  missile["Interceptor_missile_nums"] > 0
        
        # 新增条件2: 导弹速度高于飞机速度
        speed_condition = missile["speed"] > aircraft["speed"]
        
        # 新增条件3: 计算预计相遇时间
        dx = missile["position"][0] - aircraft["position"][0]
        dz = missile["position"][1] - aircraft["position"][1]
        distance = np.sqrt(dx**2 + dz**2)
        
        # 计算相对速度向量
        aircraft_heading_rad = np.radians(aircraft["heading"])
        missile_heading_rad = np.radians(missile["heading"])
        
        aircraft_vel = np.array([
            aircraft["speed"] * np.cos(aircraft_heading_rad),
            aircraft["speed"] * np.sin(aircraft_heading_rad)
        ])
        
        missile_vel = np.array([
            missile["speed"] * np.cos(missile_heading_rad),
            missile["speed"] * np.sin(missile_heading_rad)
        ])
        
        relative_vel = missile_vel - aircraft_vel
        direction_vector = np.array([dx, dz])
        direction_norm = np.linalg.norm(direction_vector)
        
        if direction_norm > 0:
            direction_vector = direction_vector / direction_norm
            closing_speed = np.dot(relative_vel, direction_vector)
        else:
            closing_speed = 0
        
        # 计算预计相遇时间（秒）
        if closing_speed > 0:
            time_to_impact = distance / closing_speed
            time_condition = time_to_impact <= 20
        else:
            time_condition = False  # 导弹未接近
        
        # 组合所有拦截条件
        should_intercept = (
            intercept_launch_condition
            and current_threat > 0.92 
            and threat_trend > 0.015 
            and speed_condition 
            and time_condition
        )
        
        if should_intercept and self.verbose:
            print(f"导弹#{missile_idx}满足拦截条件: "
                  f"威胁度={current_threat:.3f}, 变化率={threat_trend:.3f}, "
                  f"速度比={missile['speed']:.1f}>{aircraft['speed']:.1f}, "
                  f"预计时间={time_to_impact if closing_speed > 0 else 'N/A':.1f}s")
        
        return should_intercept
    
    def _turn_to_missile(self, context):
        """
        调转机头指向导弹
        
        参数:
            context: 行为树执行上下文
        
        返回:
            dict: 调转机头后的飞机状态
        """
        aircraft = context["aircraft"]
        missiles = context["missiles"]
        max_threat_idx = context["max_threat_idx"]
        
        threat_missile = missiles[max_threat_idx]
        
        # 计算指向导弹的方向
        dx = threat_missile["position"][0] - aircraft["position"][0]
        dz = threat_missile["position"][1] - aircraft["position"][1]
        
        # 计算新的航向角（指向导弹）
        new_heading = np.arctan2(dz, dx) * 180 / np.pi
        new_heading = (new_heading + 360) % 360  # 确保在0-360度范围内
        
        # 创建调转机头动作
        turn_action = {
            "position": aircraft["position"],
            "height": aircraft["height"],
            "speed": aircraft["speed"],
            "heading": new_heading,
            "do_maneuver": 1,
            "launch_missile": 0
        }
        
        return turn_action
    
    def _generate_intercept_sequence(self, context):
        """
        生成拦截动作序列
        
        参数:
            context: 行为树执行上下文
        
        返回:
            list: 拦截动作序列 [调转机头动作, 发射拦截弹动作]
        """
        # 阶段1：调转机头指向导弹
        turn_action = self._turn_to_missile(context)
        
        # 阶段2：发射拦截弹
        launch_action = copy.deepcopy(turn_action)
        launch_action["do_maneuver"] = 0
        launch_action["launch_missile"] = 1
        
        if self.verbose:
            print(f"生成拦截序列: 调转机头至{turn_action['heading']:.1f}度，然后发射拦截弹")
        
        return [turn_action, launch_action]
    
    def _process_with_behavior_tree(self, state):
        """
        使用行为树模式处理状态
        
        参数:
            state: 当前环境状态字典
        
        返回:
            包含action和explanation的字典
        """
        aircraft = state['aircraft']
        missiles = state['missiles']
        
        # 构建优化后的行为树
        root = SequenceNode("DefenseDecisionTree")
        
        # 节点1: 选择最高威胁导弹
        select_node = ActionNode("SelectMaxThreat", self._select_max_threat)
        
        # 节点2: 决策节点 - 选择拦截或逃逸
        decision_node = SelectorNode("InterceptOrEvade")
        
        # 拦截分支
        intercept_condition = ActionNode("CheckInterceptCondition", self._check_intercept_condition)
        intercept_action = ActionNode("GenerateIntercept", self._generate_intercept_action)
        intercept_sequence = SequenceNode("InterceptSequence")
        intercept_sequence.add_child(intercept_condition)
        intercept_sequence.add_child(intercept_action)
        
        # 逃逸分支（备选方案）
        evade_node = ActionNode("GenerateEvasion", self._generate_evasion)
        
        # 将分支添加到决策节点
        decision_node.add_child(intercept_sequence)
        decision_node.add_child(evade_node)
        
        # 节点3: 创建解释
        explain_node = ActionNode("CreateExplanation", self._create_explanation)
        
        root.add_child(select_node)
        root.add_child(decision_node)
        root.add_child(explain_node)
        
        # 初始化执行上下文
        context = {
            "state": state,
            "aircraft": aircraft,
            "missiles": missiles,
            "max_threat_idx": None,
            "evasion_action": None,
            "explanation": "",
            "threat_value": 0.0
        }
        
        # 执行行为树
        status = root.execute(context)
        
        if status == NodeStatus.SUCCESS:
            return {
                "action": context["evasion_action"],
                "explanation": context["explanation"]
            }
        else:
            # 行为树执行失败，返回原始状态
            return {
                "action": aircraft,
                "explanation": "行为树执行失败，保持原有飞行状态"
            }
    
    def _select_max_threat(self, context):
        """
        行为树节点：选择最高威胁导弹
        
        参数:
            context: 行为树执行上下文
        
        返回:
            bool: 执行是否成功
        """
        try:
            aircraft = context["aircraft"]
            missiles = context["missiles"]
            
            # 获取最大威胁导弹
            max_threat, max_idx = self.threat_model.get_max_threat(aircraft, missiles)
            
            if max_idx == -1:
                return False
            
            # 保存结果到上下文
            context["max_threat_idx"] = max_idx
            context["threat_value"] = max_threat
            
            if self.verbose:
                print(f"选择最高威胁导弹: 索引{max_idx}, 威胁值{max_threat:.4f}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"选择最高威胁导弹失败: {e}")
            return False
    
    def _check_intercept_condition(self, context):
        """
        行为树节点：检查是否满足拦截条件
        
        参数:
            context: 行为树执行上下文
        
        返回:
            bool: 是否满足拦截条件
        """
        try:
            max_threat_idx = context["max_threat_idx"]
            threat_value = context["threat_value"]
            
            if max_threat_idx is None:
                return False
            
            # 获取导弹对象
            missile = context["missiles"][max_threat_idx]
            
            # 检查拦截条件（传递飞机和导弹对象）
            should_intercept = self._should_intercept(
                max_threat_idx, 
                threat_value,
                context["aircraft"],
                missile
            )
            
            if should_intercept:
                context["action_type"] = "intercept"
                if self.verbose:
                    print(f"满足拦截条件，将执行拦截动作")
            
            return should_intercept
            
        except Exception as e:
            if self.verbose:
                print(f"检查拦截条件失败: {e}")
            return False
    
    def _generate_intercept_action(self, context):
        """
        行为树节点：生成拦截动作
        
        参数:
            context: 行为树执行上下文
        
        返回:
            bool: 执行是否成功
        """
        try:
            # 生成拦截动作序列
            intercept_sequence = self._generate_intercept_sequence(context)
            
            # 由于行为树一次只能返回一个动作，我们返回发射拦截弹的动作
            # 调转机头动作可以作为前置动作或在解释中说明
            launch_action = intercept_sequence[1]  # 发射拦截弹动作
            
            # 添加决策标志：防御决策
            launch_action.update({
                'use_defense_decision': 1,
                'use_attack_decision': 0,
                'use_support_decision': 0,
                'use_radar_decision': 1
            })
            
            # 保存到上下文
            context["evasion_action"] = launch_action
            context["intercept_sequence"] = intercept_sequence
            context["action_type"] = "intercept"
            
            if self.verbose:
                print(f"生成拦截动作: 调转机头并发射拦截弹")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"生成拦截动作失败: {e}")
            return False
    
    def _generate_evasion(self, context):
        """
        行为树节点：生成优化的导弹逃逸动作
        
        关键改进：
        1. 区分不同类型导弹的逃逸策略
        2. 动态计算逃逸方向和距离
        3. 能量管理（速度/高度协同）
        4. 物理约束保障
        
        参数:
            context: 行为树执行上下文
        
        返回:
            bool: 执行是否成功
        """
        try:
            aircraft = context["aircraft"]
            missiles = context["missiles"]
            max_threat_idx = context["max_threat_idx"]
            
            # 验证威胁导弹索引
            if max_threat_idx is None or max_threat_idx >= len(missiles):
                return False
            
            threat_missile = missiles[max_threat_idx]
            
            # 计算导弹相对位置和速度
            dx = threat_missile["position"][0] - aircraft["position"][0]
            dz = threat_missile["position"][1] - aircraft["position"][1]
            missile_range = np.sqrt(dx**2 + dz**2)
            
            # 获取导弹类型信息（默认雷达制导）
            missile_type = threat_missile.get("type", "radar")
            
            # 动态计算逃逸参数
            evasion_params = self._calculate_evasion_params(aircraft, threat_missile, missile_type)
            
            # 计算逃逸方向（关键优化）
            evasion_vector = self._calculate_evasion_vector(aircraft, threat_missile, missile_type)
            
            # 标准化逃逸方向
            evasion_norm = np.linalg.norm(evasion_vector)
            if evasion_norm > 0:
                evasion_vector = evasion_vector / evasion_norm
            
            # 计算逃逸目标位置
            evasion_distance = evasion_params["distance"]
            new_position = aircraft["position"] + evasion_vector * evasion_distance
            
            # 计算高度变化（基于导弹类型）
            new_height = aircraft["height"] + evasion_params["height_change"]
            
            # 计算速度变化（基于导弹接近速度）
            new_speed = aircraft["speed"] + evasion_params["speed_change"]
            
            # 计算新航向（沿逃逸方向）
            evasion_heading = np.arctan2(evasion_vector[1], evasion_vector[0]) * 180 / np.pi
            new_heading = (evasion_heading + 360) % 360
            
            # 构建逃逸动作
            evasion_action = {
                "position": new_position,
                "height": np.clip(new_height, 1000, 15000),  # 高度限制
                "speed": np.clip(new_speed, 200, 500),       # 速度限制
                "heading": new_heading,
                "do_maneuver": 1,
                "launch_missile": 1 if evasion_params.get("launch_interceptor", False) else 0,  # 修复：根据逃逸参数设置拦截弹发射标志
                # 添加决策标志：防御决策
                "use_defense_decision": 1,
                "use_attack_decision": 0,
                "use_support_decision": 0,
                'use_radar_decision': 1
            }
            
            # 保存到上下文
            context["evasion_action"] = evasion_action
            context["evasion_vector"] = evasion_vector
            context["evasion_distance"] = evasion_distance
            context["action_type"] = "evasion"
            
            if self.verbose:
                print(f"生成优化的逃逸动作: 位置{new_position}, 高度{new_height}, 速度{new_speed}")
                print(f"导弹类型: {missile_type}, 距离: {missile_range:.1f}米")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"生成逃逸动作失败: {e}")
            return False

    def _calculate_evasion_params(self, aircraft, missile, missile_type):
        """动态计算逃逸参数（基于导弹类型和距离）"""
        # 计算导弹接近速度
        dx = missile["position"][0] - aircraft["position"][0]
        dz = missile["position"][1] - aircraft["position"][1]
        missile_range = np.sqrt(dx**2 + dz**2)
        
        # 优化1: 动态计算导弹朝向 - 默认指向锁定的飞机
        # 计算导弹指向飞机的方向
        dx_to_aircraft = aircraft["position"][0] - missile["position"][0]
        dz_to_aircraft = aircraft["position"][1] - missile["position"][1]
        missile_heading = np.arctan2(dz_to_aircraft, dx_to_aircraft) * 180 / np.pi
        missile_heading = (missile_heading + 360) % 360  # 标准化到0-360度
        
        # 获取导弹速度向量
        missile_speed = missile.get("speed")
        missile_heading_rad = np.deg2rad(missile_heading)
        missile_velocity = np.array([
            np.cos(missile_heading_rad) * missile_speed,
            np.sin(missile_heading_rad) * missile_speed
        ])
        
        # 计算飞机速度向量
        aircraft_heading_rad = np.deg2rad(aircraft["heading"])
        aircraft_velocity = np.array([
            np.cos(aircraft_heading_rad) * aircraft["speed"],
            np.sin(aircraft_heading_rad) * aircraft["speed"]
        ])
        
        # 计算接近速度
        closing_velocity = np.dot(missile_velocity - aircraft_velocity, 
                                [dx, dz]) / max(missile_range, 1)
        
        # 计算接近时间 (秒)
        time_to_impact = missile_range / max(abs(closing_velocity), 1)  # 避免除零
        
        # 根据导弹类型确定策略
        params = {}
        
        # 优化2: 根据导弹相对高度动态设置height_change
        height_diff = missile["height"] - aircraft["height"]
        
        # 近距离紧急规避 (距离<40000米) - 修正版：基于导弹位置和高度差
        if missile_range < 40000:
            # 计算导弹相对位置判断前后侧
            missile_angle = np.degrees(np.arctan2(dz_to_aircraft, dx_to_aircraft)) % 360
            aircraft_heading = aircraft["heading"]
            angle_diff = (missile_angle - aircraft_heading + 360) % 360
            is_behind = 90 < angle_diff < 270  # 导弹在后方
            is_front = angle_diff <= 90 or angle_diff >= 270  # 导弹在前方
            
            if is_front:
                # 前侧导弹：根据高度差决定升降
                if height_diff > 1000:
                    # 导弹高于飞机1000m以上，降高从导弹下方掠过
                    params["height_change"] = -min(2000, aircraft["height"] - 1000)
                    params["evasion_type"] = "dive_under_missile"
                    if self.verbose:
                        print(f"前侧导弹高度差{height_diff:.0f}m > 1000m，执行降高策略")
                else:
                    # 导弹不够高，升高从导弹上方掠过
                    params["height_change"] = min(2000, 15000 - aircraft["height"])
                    params["evasion_type"] = "climb_over_missile"
                    if self.verbose:
                        print(f"前侧导弹高度差{height_diff:.0f}m ≤ 1000m，执行升高策略")
            elif is_behind:
                # 后侧导弹：计算撞击时间决定策略
                impact_time = missile_range / max(closing_velocity, 1) if closing_velocity > 0 else float('inf')
                
                if impact_time <= 3.0:
                    # 3秒内撞击，准备发射拦截弹并机动
                    params["height_change"] = 500  # 轻微爬升保持机动性
                    params["evasion_type"] = "intercept_preparation"
                    params["launch_interceptor"] = True  # 设置拦截弹发射标志
                    if self.verbose:
                        print(f"后侧导弹{impact_time:.1f}秒内撞击，准备发射拦截弹")
                else:
                    # 时间充足，常规机动
                    params["height_change"] = 800 if aircraft["height"] < 10000 else -500
                    params["evasion_type"] = "normal_maneuver"
                    if self.verbose:
                        print(f"后侧导弹{impact_time:.1f}秒后撞击，执行常规机动")
            else:
                # 侧方导弹：垂直机动
                if closing_velocity > 0:  # 导弹接近
                    params["height_change"] = 1000 if aircraft["height"] < 12000 else -1000
                    params["evasion_type"] = "perpendicular_climb"
                else:
                    params["height_change"] = -800 if aircraft["height"] > 3000 else 800
                    params["evasion_type"] = "perpendicular_dive"
        # 中远距离决策
        else:
            if abs(height_diff) > 500:  # 显著高度差
                if missile_type == "radar":  # 雷达制导导弹
                    # 导弹在上方则下降，下方则上升
                    params["height_change"] = -np.sign(height_diff) * min(1000, abs(height_diff))
                elif missile_type == "heat":  # 红外制导导弹
                    # 导弹在上方则上升，下方则下降（避免热源）
                    params["height_change"] = np.sign(height_diff) * min(1500, abs(height_diff))
                else:  # 默认策略
                    params["height_change"] = -np.sign(height_diff) * min(1000, abs(height_diff))
            else:  # 同高度
                if missile_type == "radar":
                    # 雷达制导：优先改变高度
                    params["height_change"] = 500 if aircraft["height"] < 10000 else -500
                elif missile_type == "heat":
                    # 根据接近时间决策
                    if time_to_impact < 5:  # 5秒内命中
                        params["height_change"] = -800  # 紧急俯冲
                    else:
                        params["height_change"] = 600  # 适度爬升
                else:
                    params["height_change"] = 500 if closing_velocity > 0 else -500
        
        # 设置其他参数
        if missile_type == "heat":  # 红外制导导弹
            params["speed_change"] = -50  # 减速降低红外特征
            params["distance"] = min(aircraft["speed"] * 3, 10000)  # 3秒飞行距离
        elif missile_type == "radar":  # 雷达制导导弹
            params["speed_change"] = 80  # 加速脱离
            params["distance"] = min(aircraft["speed"] * 4, 15000)  # 4秒飞行距离
        else:  # 默认策略
            params["speed_change"] = 60
            params["distance"] = min(aircraft["speed"] * 3.5, 12000)
            params["deploy_flares"] = 2
        
        # 根据接近速度调整参数
        if closing_velocity > 200:  # 高速接近
            params["distance"] *= 1.5
            params["speed_change"] += 30
        elif closing_velocity < -100:  # 导弹远离
            params["distance"] *= 0.7
        
        # 根据剩余高度调整
        if aircraft["height"] < 3000:  # 低空避免俯冲
            params["height_change"] = max(params["height_change"], 500)
        
        return params

    def _is_missile_threatening(self, aircraft, missile):
        """
        判断导弹是否对飞机构成威胁
        
        参数:
            aircraft: 飞机状态字典
            missile: 导弹状态字典
        
        返回:
            bool: 是否构成威胁
        """
        try:
            # 计算导弹到飞机的位置向量
            dx = aircraft["position"][0] - missile["position"][0]
            dz = aircraft["position"][1] - missile["position"][1]
            distance = np.sqrt(dx**2 + dz**2)
            
            if distance < 1e-6:  # 避免除零
                return True  # 距离极近视为威胁
            
            # 计算飞机速度向量
            aircraft_heading_rad = np.radians(aircraft.get("heading", 0))
            aircraft_vel = np.array([
                aircraft["speed"] * np.cos(aircraft_heading_rad),
                aircraft["speed"] * np.sin(aircraft_heading_rad)
            ])
            
            # 计算导弹速度向量（假设导弹指向飞机）
            if "heading" not in missile:
                # 导弹指向飞机的方向
                missile_heading_rad = np.arctan2(dz, dx)
            else:
                missile_heading_rad = np.radians(missile["heading"])
            
            missile_vel = np.array([
                missile["speed"] * np.cos(missile_heading_rad),
                missile["speed"] * np.sin(missile_heading_rad)
            ])
            
            # 计算相对速度向量（导弹相对于飞机的速度）
            relative_vel = missile_vel - aircraft_vel
            
            # 计算位置方向向量（从导弹指向飞机）
            direction_vector = np.array([dx, dz]) / distance
            
            # 计算接近速度（相对速度在位置方向的投影）
            closing_speed = np.dot(relative_vel, direction_vector)
            
            # 威胁判断条件：
            # 1. 导弹正在接近（closing_speed > 0）
            # 2. 导弹速度 > 飞机速度 * 0.8（速度优势）
            is_approaching = closing_speed > 0
            has_speed_advantage = missile["speed"] > aircraft["speed"] * 0.8
            
            if self.verbose:
                print(f"威胁判断详情: 距离={distance:.1f}m, 接近速度={closing_speed:.1f}m/s, "
                      f"导弹速度={missile['speed']}m/s, 威胁阈值={aircraft['speed'] * 0.8:.1f}m/s")
                print(f"接近判断={is_approaching}, 速度优势={has_speed_advantage}")
            
            return is_approaching and has_speed_advantage
            
        except Exception as e:
            if self.verbose:
                print(f"威胁判断异常: {e}")
            return True  # 异常情况下保守处理，视为威胁

    def _calculate_evasion_direction(self, aircraft, missile):
        """
        计算最优逃逸方向（角度，0-360度）
        
        参数:
            aircraft: 飞机状态字典
            missile: 导弹状态字典
        
        返回:
            float: 逃逸方向角度（度）
        """
        # 使用arctan2计算导弹指向飞机的精确角度
        dx = aircraft["position"][0] - missile["position"][0]
        dz = aircraft["position"][1] - missile["position"][1]
        missile_to_aircraft_angle = np.degrees(np.arctan2(dz, dx)) % 360
        
        # 生成两个逃逸选项：导弹方向±90°
        option1 = (missile_to_aircraft_angle + 90) % 360
        option2 = (missile_to_aircraft_angle - 90) % 360
        
        # 获取敌机方向向量（如果有分配敌机）
        enemy_vector = None
        if 'assigned_enemy' in aircraft and aircraft['assigned_enemy'] is not None:
            try:
                dx_enemy = aircraft['assigned_enemy'][0] - aircraft['position'][0]
                dz_enemy = aircraft['assigned_enemy'][1] - aircraft['position'][1]
                enemy_distance = np.sqrt(dx_enemy**2 + dz_enemy**2)
                if enemy_distance > 1e-6:
                    enemy_vector = np.array([dx_enemy, dz_enemy]) / enemy_distance
            except (TypeError, IndexError, KeyError):
                enemy_vector = None
        
        # 选择最优逃逸方向
        if enemy_vector is not None:
            # 计算两个选项与敌机方向的夹角
            vec1 = np.array([np.cos(np.radians(option1)), np.sin(np.radians(option1))])
            vec2 = np.array([np.cos(np.radians(option2)), np.sin(np.radians(option2))])
            
            # 计算向量点积（越大表示越接近）
            dot1 = np.dot(vec1, enemy_vector)
            dot2 = np.dot(vec2, enemy_vector)
            
            # 选择点积更大的方向（更靠近敌机）
            selected_direction = option1 if dot1 > dot2 else option2
            
            if self.verbose:
                print(f"选择逃逸方向: {selected_direction:.1f}度 (靠近敌机方向)")
        else:
            # 无分配敌机时，选择与当前航向更接近的方向
            current_heading = aircraft["heading"]
            
            # 计算角度差（考虑360度循环）
            def angle_diff(a1, a2):
                diff = abs(a1 - a2)
                return min(diff, 360 - diff)
            
            diff1 = angle_diff(option1, current_heading)
            diff2 = angle_diff(option2, current_heading)
            
            selected_direction = option1 if diff1 < diff2 else option2
            
            if self.verbose:
                print(f"选择逃逸方向: {selected_direction:.1f}度 (接近当前航向)")
        
        return selected_direction

    def _update_missile_history(self, missiles):
        """更新导弹历史位置"""
        current_time = time.time()
        
        for i, missile in enumerate(missiles):
            missile_id = missile.get('id', f"missile_{i}")  # 使用ID或索引作为标识
            
            if missile_id not in self.missile_history:
                self.missile_history[missile_id] = []
            
            # 记录当前位置和时间
            position = missile['position'].copy() if isinstance(missile['position'], list) else list(missile['position'])
            self.missile_history[missile_id].append({
                'time': current_time,
                'position': position,
                'speed': missile.get('speed', 0)
            })
            
            # 保持历史长度
            if len(self.missile_history[missile_id]) > self.max_history_length:
                self.missile_history[missile_id].pop(0)

    def _calculate_missile_heading(self, missile, missile_idx):
        """计算导弹的真实heading基于历史位置"""
        missile_id = missile.get('id', f"missile_{missile_idx}")
        
        if missile_id not in self.missile_history:
            # 没有历史数据，使用导弹自身的heading或None
            return missile.get('heading', None)
        
        history = self.missile_history[missile_id]
        if len(history) < 2:
            # 历史数据不足，使用导弹自身的heading
            return missile.get('heading', None)
        
        # 计算速度向量（基于最近两个位置点）
        latest = history[-1]
        previous = history[-2]
        
        time_diff = latest['time'] - previous['time']
        if time_diff <= 0:
            # 时间差无效，使用导弹自身的heading
            return missile.get('heading', None)
        
        # 计算位移向量
        dx = latest['position'][0] - previous['position'][0]
        dz = latest['position'][1] - previous['position'][1]
        
        # 如果位移过小，认为导弹静止，使用导弹自身的heading
        if np.sqrt(dx**2 + dz**2) < 1e-3:
            return missile.get('heading', None)
        
        # 计算heading角度
        heading_rad = np.arctan2(dz, dx)
        heading_deg = np.degrees(heading_rad) % 360
        
        if self.verbose:
            print(f"导弹#{missile_idx}真实heading: {heading_deg:.1f}度 (基于历史位置)")
        
        return heading_deg

    def _is_missile_tracking_aircraft(self, aircraft, missile, missile_heading):
        """
        判断导弹是否还在追踪飞机（修正版：基于导弹历史轨迹和指向判断）
        
        关键修正：
        1. 导弹距离计算必须基于导弹指向飞机方向时才有效
        2. 结合导弹速度变化趋势判断
        
        参数:
            aircraft: 飞机状态字典
            missile: 导弹状态字典
            missile_heading: 导弹真实heading角度
        
        返回:
            bool: True表示导弹仍在追踪，False表示已被规避
        """
        if missile_heading is None:
            # 没有heading信息，保守判断为仍在追踪
            return True
        
        # 计算导弹到飞机的理论方向（导弹应该指向飞机的方向）
        dx = aircraft["position"][0] - missile["position"][0]
        dz = aircraft["position"][1] - missile["position"][1]
        distance = np.sqrt(dx**2 + dz**2)
        
        if distance < 1e-6:
            # 距离过近，保守判断为仍在追踪
            return True
        
        # 计算导弹指向飞机的理论方向
        theoretical_heading = np.degrees(np.arctan2(dz, dx)) % 360
        
        # 计算角度差
        angle_diff = abs(missile_heading - theoretical_heading)
        angle_diff = min(angle_diff, 360 - angle_diff)  # 处理360度循环
        
        # 修正1：50度内认为仍在追踪
        is_tracking = angle_diff < 50
        
        # 修正2：检查导弹是否在接近飞机（基于距离变化）
        missile_id = missile.get('id', f"missile_0")
        if missile_id in self.missile_history and len(self.missile_history[missile_id]) >= 2:
            # 计算距离变化趋势
            current_pos = np.array(missile["position"])
            aircraft_pos = np.array(aircraft["position"])
            current_distance = np.linalg.norm(current_pos - aircraft_pos)
            
            # 获取历史位置计算之前的距离
            prev_missile_pos = np.array(self.missile_history[missile_id][-2]["position"])
            prev_distance = np.linalg.norm(prev_missile_pos - aircraft_pos)
            
            # 如果距离在增加，可能表示导弹已被规避
            distance_change = current_distance - prev_distance
            if distance_change > 100:  # 距离增加超过100米
                is_tracking = False
                if self.verbose:
                    print(f"导弹距离增加{distance_change:.1f}m，判断为已被规避")
        
        # 修正3：检查导弹速度是否在下降（可能燃料耗尽）
        if missile_id in self.missile_history and len(self.missile_history[missile_id]) >= 2:
            current_speed = missile.get("speed", 0)
            prev_speed = self.missile_history[missile_id][-2].get("speed", 0)
            speed_drop = prev_speed - current_speed
            
            if speed_drop > 50:  # 速度下降超过50m/s
                is_tracking = False
                if self.verbose:
                    print(f"导弹速度下降{speed_drop:.1f}m/s，判断为失去动力")
        
        if self.verbose:
            print(f"导弹追踪判断: 真实heading={missile_heading:.1f}度, "
                  f"理论heading={theoretical_heading:.1f}度, "
                  f"角度差={angle_diff:.1f}度, 距离={distance:.1f}m, 仍在追踪={is_tracking}")
        
        return is_tracking

    def _calculate_missile_escape_direction(self, aircraft, missile):
        """
        计算导弹指向飞机方向（用于4-40km逃逸）
        
        关键修正：
        1. 逃逸方向是以导弹为起点指向飞机的方向
        2. 这个方向就是导弹追踪飞机的方向
        3. 飞机沿这个方向逃逸可以最大化与导弹的分离速度
        
        参数:
            aircraft: 飞机状态字典
            missile: 导弹状态字典
        
        返回:
            np.array: 标准化的逃逸方向向量（导弹指向飞机）
        """
        # 计算导弹指向飞机的向量（以导弹为起点）
        dx = aircraft["position"][0] - missile["position"][0]
        dz = aircraft["position"][1] - missile["position"][1]
        distance = np.sqrt(dx**2 + dz**2)
        
        if distance < 1e-6:
            # 距离过近，返回当前航向
            heading_rad = np.radians(aircraft["heading"])
            return np.array([np.cos(heading_rad), np.sin(heading_rad)])
        
        # 标准化导弹指向飞机的方向向量
        escape_vector = np.array([dx, dz]) / distance
        
        # 验证：获取导弹真实heading并与理论方向比较
        missile_idx = 0  # 简化处理
        missile_heading = self._calculate_missile_heading(missile, missile_idx)
        if missile_heading is None:
            missile_heading = self._calculate_missile_to_aircraft_direction(missile, aircraft)
        
        # 计算导弹指向飞机的理论方向
        theoretical_heading = np.degrees(np.arctan2(dz, dx)) % 360
        
        # 检查导弹是否真正指向飞机
        angle_diff = abs(missile_heading - theoretical_heading)
        angle_diff = min(angle_diff, 360 - angle_diff)
        
        if angle_diff > 45:  # 如果角度差太大，导弹可能不是指向飞机
            if self.verbose:
                print(f"警告：导弹heading={missile_heading:.1f}度与理论方向={theoretical_heading:.1f}度差异{angle_diff:.1f}度")
        
        if self.verbose:
            escape_heading = np.degrees(np.arctan2(escape_vector[1], escape_vector[0])) % 360
            print(f"4-40km逃逸方向: 导弹指向飞机方向={escape_heading:.1f}度")
        
        return escape_vector

    def _calculate_missile_to_aircraft_direction(self, missile, aircraft):
        """计算导弹指向飞机的方向（fallback方法）"""
        dx = aircraft["position"][0] - missile["position"][0]
        dz = aircraft["position"][1] - missile["position"][1]
        heading_rad = np.arctan2(dz, dx)
        heading_deg = np.degrees(heading_rad) % 360
        
        if self.verbose:
            print(f"使用fallback：导弹指向飞机方向 {heading_deg:.1f}度")
        
        return heading_deg

    def _calculate_evasion_vector(self, aircraft, missile, missile_type):
        """
        计算最优逃逸方向向量（修正版：根据导弹追踪状态决策）
        
        核心逻辑：
        1. 先判断导弹是否还在追踪飞机
        2. 如果被规避（不再追踪），完全指向敌机进攻
        3. 如果仍在追踪，按距离执行5种逃逸策略（含超近距离<5km策略）
        """
        # 计算导弹相对位置和距离
        dx = missile["position"][0] - aircraft["position"][0]
        dz = missile["position"][1] - aircraft["position"][1]
        distance = np.sqrt(dx**2 + dz**2)  # 单位：米
        distance_km = distance / 1000  # 转换为公里
        
        # 获取导弹真实heading（基于历史位置）
        missile_idx = 0  # 简化处理，实际使用时可通过上下文传递
        missile_heading = self._calculate_missile_heading(missile, missile_idx)
        if missile_heading is None:
            # 使用fallback：假设导弹指向飞机
            missile_heading = self._calculate_missile_to_aircraft_direction(missile, aircraft)
        
        # 关键：判断导弹是否还在追踪飞机
        is_missile_tracking = self._is_missile_tracking_aircraft(aircraft, missile, missile_heading)
        
        # 获取敌机方向向量
        enemy_direction = np.array([0.0, 0.0])
        if 'assigned_enemy' in aircraft and aircraft['assigned_enemy'] is not None:
            try:
                dx_enemy = aircraft['assigned_enemy'][0] - aircraft['position'][0]
                dz_enemy = aircraft['assigned_enemy'][1] - aircraft['position'][1]
                enemy_distance = np.sqrt(dx_enemy**2 + dz_enemy**2)
                if enemy_distance > 1e-6:
                    enemy_direction = np.array([dx_enemy, dz_enemy]) / enemy_distance
            except (TypeError, IndexError, KeyError):
                pass
        
        # 计算当前航向向量
        current_heading_rad = np.radians(aircraft["heading"])
        current_direction = np.array([
            np.cos(current_heading_rad),
            np.sin(current_heading_rad)
        ])
        
        if self.verbose:
            print(f"逃逸分析: 距离={distance_km:.1f}km, 导弹追踪={is_missile_tracking}, "
                  f"导弹heading={missile_heading:.1f}度")
        
        # 核心判断：导弹是否已被规避
        if not is_missile_tracking:
            # 导弹已被规避，转为攻击模式
            if np.linalg.norm(enemy_direction) > 0:
                if self.verbose:
                    print(f"导弹已被规避，完全指向敌机进攻")
                return enemy_direction
            else:
                if self.verbose:
                    print(f"导弹已被规避，无敌机目标，保持当前航向")
                return current_direction
        
        # 导弹仍在追踪，执行距离分段逃逸策略
        return self._execute_tracking_missile_evasion(aircraft, missile, missile_type, distance_km, 
                                                     enemy_direction, current_direction)

    def _execute_tracking_missile_evasion(self, aircraft, missile, missile_type, distance_km, 
                                        enemy_direction, current_direction):
        """
        执行针对追踪导弹的逃逸策略（含超近距离策略）
        
        参数:
            aircraft: 飞机状态
            missile: 导弹状态
            missile_type: 导弹类型
            distance_km: 弹目距离（公里）
            enemy_direction: 敌机方向向量
            current_direction: 当前航向向量
        
        返回:
            np.array: 逃逸方向向量
        """
        # 计算导弹相对位置
        dx = missile["position"][0] - aircraft["position"][0]
        dz = missile["position"][1] - aircraft["position"][1]
        
        # 计算飞机和导弹的速度向量和cos值
        aircraft_heading_rad = np.radians(aircraft["heading"])
        aircraft_velocity = np.array([
            np.cos(aircraft_heading_rad) * aircraft["speed"],
            np.sin(aircraft_heading_rad) * aircraft["speed"]
        ])
        
        # 获取导弹真实heading
        missile_idx = 0
        missile_heading = self._calculate_missile_heading(missile, missile_idx)
        if missile_heading is None:
            missile_heading = self._calculate_missile_to_aircraft_direction(missile, aircraft)
        
        missile_heading_rad = np.radians(missile_heading)
        missile_velocity = np.array([
            np.cos(missile_heading_rad) * missile["speed"],
            np.sin(missile_heading_rad) * missile["speed"]
        ])
        
        # 计算相对方向
        aircraft_direction = aircraft_velocity / np.linalg.norm(aircraft_velocity) if np.linalg.norm(aircraft_velocity) > 0 else np.array([1, 0])
        missile_direction = missile_velocity / np.linalg.norm(missile_velocity) if np.linalg.norm(missile_velocity) > 0 else np.array([1, 0])
        cos_value = np.dot(aircraft_direction, missile_direction)
        
        # 判断导弹相对位置
        missile_angle = np.degrees(np.arctan2(dz, dx)) % 360
        aircraft_heading = aircraft["heading"]
        angle_diff = (missile_angle - aircraft_heading + 360) % 360
        is_behind = 135 < angle_diff < 225  # 导弹在后方
        is_front = angle_diff <= 45 or angle_diff >= 315  # 导弹在前方
        
        # 判断敌机是否在前方
        is_enemy_front = False
        if np.linalg.norm(enemy_direction) > 0:
            try:
                dx_enemy = aircraft['assigned_enemy'][0] - aircraft['position'][0]
                dz_enemy = aircraft['assigned_enemy'][1] - aircraft['position'][1]
                enemy_angle = np.degrees(np.arctan2(dz_enemy, dx_enemy)) % 360
                enemy_angle_diff = (enemy_angle - aircraft_heading + 360) % 360
                is_enemy_front = enemy_angle_diff <= 90 or enemy_angle_diff >= 270
            except (TypeError, IndexError, KeyError):
                pass
        
        if self.verbose:
            print(f"追踪导弹逃逸: 距离={distance_km:.1f}km, cos值={cos_value:.3f}, "
                  f"导弹后方={is_behind}, 前方={is_front}, 敌机前方={is_enemy_front}")
        
        # 新增：超近距离策略 (<4km)
        if distance_km < 4:
            return self._handle_ultra_close_missile(aircraft, missile, is_front, is_behind, 
                                                  enemy_direction, current_direction)
        
        # 策略1: 远距离(80km外)或导弹速度低
        elif distance_km > 80 or missile["speed"] < aircraft["speed"]:
            if np.linalg.norm(enemy_direction) > 0:
                if self.verbose:
                    print(f"策略1: 导弹远距离或速度慢，完全指向敌机")
                return enemy_direction
            else:
                if self.verbose:
                    print(f"策略1: 导弹远距离或速度慢，无敌机目标，保持当前航向")
                return current_direction
        
        # 策略2: 中距离（40-80km）且头对头/侧向
        elif 40 <= distance_km <= 80 and cos_value < 0:
            # 使用导弹方向±90°进行逃逸
            missile_to_aircraft_rad = np.arctan2(dz, dx)
            evasion_angle1 = missile_to_aircraft_rad + np.pi/2  # +90°
            evasion_angle2 = missile_to_aircraft_rad - np.pi/2  # -90°
            
            evasion_vector1 = np.array([np.cos(evasion_angle1), np.sin(evasion_angle1)])
            evasion_vector2 = np.array([np.cos(evasion_angle2), np.sin(evasion_angle2)])
            
            # 选择更靠近敌机的方向
            if np.linalg.norm(enemy_direction) > 0:
                dot1 = np.dot(evasion_vector1, enemy_direction)
                dot2 = np.dot(evasion_vector2, enemy_direction)
                selected_vector = evasion_vector1 if dot1 > dot2 else evasion_vector2
                if self.verbose:
                    print(f"策略2: 中距离头对头/侧向，±90°逃逸靠近敌机")
                return selected_vector
            else:
                # 选择与当前航向更接近的方向
                dot1 = np.dot(evasion_vector1, current_direction)
                dot2 = np.dot(evasion_vector2, current_direction)
                selected_vector = evasion_vector1 if dot1 > dot2 else evasion_vector2
                if self.verbose:
                    print(f"策略2: 中距离头对头/侧向，±90°逃逸接近当前航向")
                return selected_vector
        
        # 策略3: 近距离（5-40km）且头对头/侧向 - 修正为反方向逃逸
        elif 4 <= distance_km <= 40 and cos_value < 0:
            # 导弹指向飞机方向的反方向逃逸
            escape_vector = self._calculate_missile_escape_direction(aircraft, missile)
            if self.verbose:
                print(f"策略3: 近距离头对头/侧向，导弹指向的反方向逃逸")
            return escape_vector
        
        # 策略4: 导弹在后方，敌机在前方
        elif is_behind and is_enemy_front:
            if np.linalg.norm(enemy_direction) > 0:
                if self.verbose:
                    print(f"策略4: 导弹在后方，敌机在前方，完全指向敌机")
                return enemy_direction
            else:
                if self.verbose:
                    print(f"策略4: 导弹在后方，无敌机目标，保持当前航向")
                return current_direction
        
        # 策略5: 导弹在后方，敌机也在后方但导弹速度低
        elif is_behind and not is_enemy_front and missile["speed"] < aircraft["speed"]:
            if np.linalg.norm(enemy_direction) > 0:
                if self.verbose:
                    print(f"策略5: 导弹在后方且速度慢，敌机也在后方，完全指向敌机进攻")
                return enemy_direction
            else:
                if self.verbose:
                    print(f"策略5: 导弹在后方且速度慢，无敌机目标，保持当前航向")
                return current_direction
        
        # 默认情况：使用原有逃逸逻辑
        else:
            if self.verbose:
                print(f"默认策略: 使用原有逃逸逻辑")
            return self._calculate_evasion_vector_original(aircraft, missile, missile_type)

    def _handle_ultra_close_missile(self, aircraft, missile, is_front, is_behind, 
                                  enemy_direction, current_direction):
        """
        处理超近距离(<4km)导弹的特殊逃逸策略
        
        参数:
            aircraft: 飞机状态
            missile: 导弹状态
            is_front: 导弹是否在前方
            is_behind: 导弹是否在后方
            enemy_direction: 敌机方向向量
            current_direction: 当前航向向量
        
        返回:
            np.array: 逃逸方向向量
        """
        # 计算高度差
        height_diff = missile["height"] - aircraft["height"]
        
        if is_front:
            # 前侧导弹：根据高度差决定升降机动
            if height_diff > 1000:
                # 导弹高于飞机1000m以上，转向导弹并降高
                missile_direction = self._calculate_turn_towards_missile(aircraft, missile)
                if self.verbose:
                    print(f"超近距离前侧导弹(高{height_diff:.0f}m)：转向导弹并准备降高")
                return missile_direction
            else:
                # 导弹不够高，升高从上方掠过
                if np.linalg.norm(enemy_direction) > 0:
                    # 升高的同时尽量指向敌机
                    combined_vector = 0.7 * enemy_direction + 0.3 * current_direction
                    if self.verbose:
                        print(f"超近距离前侧导弹(低{height_diff:.0f}m)：升高掠过并转向敌机")
                    return combined_vector
                else:
                    if self.verbose:
                        print(f"超近距离前侧导弹(低{height_diff:.0f}m)：升高掠过，保持航向")
                    return current_direction
        
        elif is_behind:
            # 后侧导弹：计算撞击时间，决定是否发射拦截弹
            impact_time = self._calculate_impact_time(aircraft, missile)
            
            if impact_time <= 3.0:
                # 3秒内撞击，需要发射拦截弹（通过修改_calculate_evasion_params实现）
                # 这里返回转向导弹方向，同时在evasion_params中设置发射标志
                missile_direction = self._calculate_turn_towards_missile(aircraft, missile)
                if self.verbose:
                    print(f"超近距离后侧导弹：{impact_time:.1f}秒内撞击，转向并准备发射拦截弹")
                return missile_direction
            else:
                # 时间充足，转向敌机攻击
                if np.linalg.norm(enemy_direction) > 0:
                    if self.verbose:
                        print(f"超近距离后侧导弹：{impact_time:.1f}秒后撞击，转向敌机")
                    return enemy_direction
                else:
                    if self.verbose:
                        print(f"超近距离后侧导弹：{impact_time:.1f}秒后撞击，保持航向")
                    return current_direction
        
        # 其他位置（侧方等）
        else:
            # 侧方导弹：尽量从垂直方向掠过
            if np.linalg.norm(enemy_direction) > 0:
                # 混合垂直逃逸和敌机方向
                perpendicular_vector = self._calculate_perpendicular_escape(aircraft, missile)
                combined_vector = 0.6 * perpendicular_vector + 0.4 * enemy_direction
                if self.verbose:
                    print(f"超近距离侧方导弹：垂直掠过并转向敌机")
                return combined_vector
            else:
                perpendicular_vector = self._calculate_perpendicular_escape(aircraft, missile)
                if self.verbose:
                    print(f"超近距离侧方导弹：垂直掠过")
                return perpendicular_vector

    def _calculate_turn_towards_missile(self, aircraft, missile):
        """
        计算转向导弹的方向向量
        
        参数:
            aircraft: 飞机状态
            missile: 导弹状态
        
        返回:
            np.array: 指向导弹的标准化方向向量
        """
        dx = missile["position"][0] - aircraft["position"][0]
        dz = missile["position"][1] - aircraft["position"][1]
        
        if np.sqrt(dx**2 + dz**2) < 1e-6:
            # 距离过近，返回当前航向
            heading_rad = np.radians(aircraft["heading"])
            return np.array([np.cos(heading_rad), np.sin(heading_rad)])
        
        # 指向导弹的方向向量
        direction_vector = np.array([dx, dz])
        direction_norm = np.linalg.norm(direction_vector)
        
        if direction_norm > 0:
            direction_vector = direction_vector / direction_norm
        
        return direction_vector

    def _calculate_impact_time(self, aircraft, missile):
        """
        计算导弹撞击飞机的预估时间
        
        参数:
            aircraft: 飞机状态
            missile: 导弹状态
        
        返回:
            float: 预估撞击时间（秒）
        """
        # 计算相对位置
        dx = missile["position"][0] - aircraft["position"][0]
        dz = missile["position"][1] - aircraft["position"][1]
        distance = np.sqrt(dx**2 + dz**2)
        
        if distance < 1e-6:
            return 0.0  # 距离极近
        
        # 计算相对速度向量
        aircraft_heading_rad = np.radians(aircraft["heading"])
        aircraft_vel = np.array([
            aircraft["speed"] * np.cos(aircraft_heading_rad),
            aircraft["speed"] * np.sin(aircraft_heading_rad)
        ])
        
        # 获取导弹heading
        missile_idx = 0
        missile_heading = self._calculate_missile_heading(missile, missile_idx)
        if missile_heading is None:
            missile_heading = self._calculate_missile_to_aircraft_direction(missile, aircraft)
        
        missile_heading_rad = np.radians(missile_heading)
        missile_vel = np.array([
            missile["speed"] * np.cos(missile_heading_rad),
            missile["speed"] * np.sin(missile_heading_rad)
        ])
        
        # 计算接近速度
        relative_vel = missile_vel - aircraft_vel
        direction_vector = np.array([dx, dz]) / distance
        closing_speed = np.dot(relative_vel, direction_vector)
        
        if closing_speed <= 0:
            return float('inf')  # 导弹未接近
        
        return distance / closing_speed

    def _calculate_perpendicular_escape(self, aircraft, missile):
        """
        计算垂直于导弹-飞机连线的逃逸方向
        
        参数:
            aircraft: 飞机状态
            missile: 导弹状态
        
        返回:
            np.array: 垂直逃逸方向向量
        """
        dx = missile["position"][0] - aircraft["position"][0]
        dz = missile["position"][1] - aircraft["position"][1]
        
        if np.sqrt(dx**2 + dz**2) < 1e-6:
            # 距离过近，返回当前航向
            heading_rad = np.radians(aircraft["heading"])
            return np.array([np.cos(heading_rad), np.sin(heading_rad)])
        
        # 计算垂直向量（逆时针旋转90度）
        perpendicular_vector = np.array([-dz, dx])
        perpendicular_norm = np.linalg.norm(perpendicular_vector)
        
        if perpendicular_norm > 0:
            perpendicular_vector = perpendicular_vector / perpendicular_norm
        
        if self.verbose:
            perp_heading = np.degrees(np.arctan2(perpendicular_vector[1], perpendicular_vector[0])) % 360
            print(f"计算垂直逃逸方向: {perp_heading:.1f}度")
        
        return perpendicular_vector

    def _calculate_evasion_vector_original(self, aircraft, missile, missile_type):
        """保留原有的逃逸向量计算逻辑作为备用"""
        # 计算导弹相对位置
        dx = missile["position"][0] - aircraft["position"][0]
        dz = missile["position"][1] - aircraft["position"][1]
        
        # 新增：计算指向敌机的方向向量
        enemy_direction = np.array([0.0, 0.0])  # 默认零向量
        if 'assigned_enemy' in aircraft and aircraft['assigned_enemy'] is not None:
            try:
                # 计算指向分配敌机的方向
                dx_enemy = aircraft['assigned_enemy'][0] - aircraft['position'][0]
                dz_enemy = aircraft['assigned_enemy'][1] - aircraft['position'][1]
                enemy_direction = np.array([dx_enemy, dz_enemy])
                enemy_direction_norm = np.linalg.norm(enemy_direction)
                if enemy_direction_norm > 0:
                    enemy_direction = enemy_direction / enemy_direction_norm
                else:
                    enemy_direction = np.array([0.0, 0.0])
            except (TypeError, IndexError, KeyError) as e:
                # 处理敌机位置数据异常
                enemy_direction = np.array([0.0, 0.0])
        
        # 考虑当前航向
        current_heading_rad = np.deg2rad(aircraft["heading"])
        current_direction = np.array([
            np.cos(current_heading_rad),
            np.sin(current_heading_rad)
        ])
        
        # 保留原有的后方导弹特殊处理逻辑
        # 检查导弹是否在飞机后方（90-270度范围）
        missile_angle = np.degrees(np.arctan2(dz, dx)) % 360
        aircraft_heading = aircraft["heading"]
        angle_diff = (missile_angle - aircraft_heading + 360) % 360
        is_behind = 90 < angle_diff < 270
        
        # 检查敌机是否在前方（270-360度或0-90度范围）
        is_enemy_front = False
        if np.linalg.norm(enemy_direction) > 0:
            dx_enemy = aircraft['assigned_enemy'][0] - aircraft['position'][0]
            dz_enemy = aircraft['assigned_enemy'][1] - aircraft['position'][1]
            enemy_angle = np.degrees(np.arctan2(dz_enemy, dx_enemy)) % 360
            enemy_angle_diff = (enemy_angle - aircraft_heading + 360) % 360
            is_enemy_front = enemy_angle_diff <= 90 or enemy_angle_diff >= 270
        
        # 特殊处理条件：导弹在后方 && 导弹速度 < 飞机速度 && 敌机在前方
        if (is_behind and 
            missile["speed"] < aircraft["speed"] and 
            is_enemy_front):
            # 不进行逃逸，80%指向敌机 + 20%当前航向
            combined_vector = 0.8 * enemy_direction + 0.2 * current_direction
            if self.verbose:
                print(f"后方导弹特殊处理: 导弹在后方且速度慢，敌机在前方，优先攻击敌机")
            return combined_vector
        
        # 检查导弹威胁程度
        is_threatening = self._is_missile_threatening(aircraft, missile)
        
        if is_threatening:
            # 有威胁：使用优化的±90°逃逸策略
            evasion_direction_deg = self._calculate_evasion_direction(aircraft, missile)
            evasion_angle_rad = np.radians(evasion_direction_deg)
            evasion_vector = np.array([
                np.cos(evasion_angle_rad),
                np.sin(evasion_angle_rad)
            ])
            
            if self.verbose:
                print(f"导弹威胁较高，执行±90°逃逸策略，方向: {evasion_direction_deg:.1f}度")
            
            # 如果有敌机目标，适度混合逃逸和攻击方向
            if np.linalg.norm(enemy_direction) > 0:
                combined_vector = 0.85 * evasion_vector + 0.15 * enemy_direction
                if self.verbose:
                    print(f"混合逃逸和攻击方向 (85%逃逸 + 15%敌机)")
            else:
                combined_vector = evasion_vector
        else:
            # 导弹威胁较低：转向分配目标
            if np.linalg.norm(enemy_direction) > 0:
                # 有敌机目标：80%指向敌机 + 20%当前航向
                combined_vector = 0.8 * enemy_direction + 0.2 * current_direction
                if self.verbose:
                    print(f"导弹威胁较低，转向敌机目标")
            else:
                # 无敌机目标：保持当前航向
                combined_vector = current_direction
                if self.verbose:
                    print(f"导弹威胁较低且无敌机目标，保持当前航向")
        
        return combined_vector

    
    def _create_explanation(self, context):
        """
        行为树节点：创建决策解释
        
        参数:
            context: 行为树执行上下文
        
        返回:
            bool: 执行是否成功
        """
        try:
            max_threat_idx = context["max_threat_idx"]
            threat_value = context["threat_value"]
            evasion_distance = context.get("evasion_distance", 0)
            aircraft = context["aircraft"]
            evasion_action = context["evasion_action"]
            
            # 获取威胁等级
            threat_levels = self.threat_model.get_threat_levels(
                aircraft, 
                context["missiles"]
            )
            
            threat_level = "未知"
            if max_threat_idx < len(threat_levels):
                threat_level = threat_levels[max_threat_idx][2]
            
            # 根据动作类型构建不同的解释文本
            action_type = context.get("action_type", "evasion")
            
            if action_type == "intercept":
                # 拦截动作解释
                intercept_sequence = context.get("intercept_sequence", [])
                if intercept_sequence:
                    turn_heading = intercept_sequence[0]['heading']
                    explanation = (
                        f"检测到高威胁导弹！目标导弹#{max_threat_idx}威胁值为{threat_value:.3f}（等级：{threat_level}），"
                        f"威胁度持续增加，满足拦截条件。"
                        f"执行拦截序列：首先调转机头至{turn_heading:.1f}度指向导弹，"
                        f"然后发射拦截弹进行主动防御。"
                        f"动作标志：机动标志={evasion_action.get('do_maneuver', 0)}，"
                        f"发射标志={evasion_action.get('launch_missile', 0)}。"
                    )
                else:
                    explanation = (
                        f"检测到高威胁导弹#{max_threat_idx}威胁值为{threat_value:.3f}，"
                        f"执行拦截动作，调转机头指向导弹并发射拦截弹。"
                    )
            else:
                # 逃逸动作解释
                explanation = (
                    f"检测到导弹威胁！目标导弹#{max_threat_idx}威胁值为{threat_value:.3f}（等级：{threat_level}），"
                    f"威胁度未达到拦截条件，采用行为树决策执行逃逸机动。"
                    f"执行垂直方向逃逸机动，移动距离{evasion_distance:.1f}单位，"
                    f"同时提升高度至{evasion_action['height']:.0f}米，"
                    f"增速至{evasion_action['speed']:.0f}节以快速脱离威胁区域。"
                    f"动作标志：机动标志={evasion_action.get('do_maneuver', 0)}，"
                    f"发射标志={evasion_action.get('launch_missile', 0)}。"
                )
            
            context["explanation"] = explanation
            
            if self.verbose:
                print(f"决策解释: {explanation}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"创建解释失败: {e}")
            
            # 提供备用解释
            context["explanation"] = f"选择威胁度最高的导弹#{max_threat_idx}进行逃逸机动"
            return True
    
    def get_behavior_explanation(self, state):
        """
        获取行为树决策解释（便捷接口）
        
        参数:
            state: 当前环境状态
        
        返回:
            str: 决策解释文本
        """
        if self.mode != "bt":
            return "当前非行为树模式，无法提供行为解释"
        
        result = self.process(state)
        return result.getprocess("explanation", "无法生成解释")
    
    def get_action_only(self, state):
        """
        仅获取动作结果（便捷接口）
        
        参数:
            state: 当前环境状态
        
        返回:
            dict: 飞机动作状态
        """
        result = self.process(state)
        if isinstance(result, dict) and "action" in result:
            return result["action"]
        return result
    
    def get_optimization_explanation(self):
        """
        获取最近一次优化的解释
        :return: 优化解释字典
        """
        if self.mode == "de":
            return getattr(self, 'last_optimization_explanation', {
                'text_explanation': '暂无优化解释',
                'optimization_type': 'Unknown'
            })
        else:
            return {
                'text_explanation': '当前模式不是DE优化模式',
                'optimization_type': 'N/A'
            }
    
    def get_comprehensive_explanation(self, state):
        """
        获取综合解释（包含决策过程和优化结果）
        :param state: 当前环境状态（位置单位：千米）
        :return: 综合解释字典
        """
        if self.mode == "de":
            # DE优化模式的综合解释
            result = self.process(state)
            
            # 获取威胁分析（需要转换为米制单位）
            converted_state = convert_state_units_to_meters(state)
            aircraft = converted_state['aircraft']
            missiles = converted_state['missiles']
            threat_analysis = self.get_threat_analysis(aircraft, missiles)
            
            # 获取优化解释
            optimization_explanation = self.get_optimization_explanation()
            
            # 计算优化效果（使用米制单位）
            original_threat = self.evaluate_current_threat(aircraft, missiles)
            
            # 将result转换为米制单位进行威胁评估
            if isinstance(result, dict) and 'position' in result:
                converted_result = convert_state_units_to_meters({'aircraft': result, 'missiles': []})['aircraft']
                result_different = True  # 如果process返回了结果，说明肯定有变化
                optimized_threat = self.evaluate_current_threat(converted_result, missiles)
            else:
                result_different = False
                optimized_threat = original_threat
            
            threat_reduction = ((original_threat - optimized_threat) / original_threat * 100) if original_threat > 0 else 0
            
            return {
                'mode': 'de_optimization',
                'threat_analysis': threat_analysis,
                'optimization_explanation': optimization_explanation,
                'effectiveness': {
                    'original_threat': original_threat,
                    'optimized_threat': optimized_threat,
                    'threat_reduction_percent': threat_reduction,
                    'selection_strategy': self.selection_strategy
                },
                'optimized_action': result.tolist() if hasattr(result, 'tolist') else result,
                'comprehensive_text': f"{optimization_explanation.get('text_explanation', '')} "
                                   f"威胁度从{original_threat:.4f}降低到{optimized_threat:.4f}，"
                                   f"降低幅度为{threat_reduction:.1f}%。采用{self.selection_strategy}策略选择最终解决方案。"
            }
        elif self.mode == "bt":
            # 行为树模式的综合解释
            result = self.process(state)
            
            # 获取威胁分析（需要转换为米制单位）
            converted_state = convert_state_units_to_meters(state)
            aircraft = converted_state['aircraft']
            missiles = converted_state['missiles']
            threat_analysis = self.get_threat_analysis(aircraft, missiles)
            
            # 计算行为树效果（使用米制单位）
            original_threat = self.evaluate_current_threat(aircraft, missiles)
            
            # 将action_result转换为米制单位进行威胁评估
            action_result = result.get('action', state['aircraft']) if isinstance(result, dict) else result
            if isinstance(action_result, dict) and 'position' in action_result:
                converted_action = convert_state_units_to_meters({'aircraft': action_result, 'missiles': []})['aircraft']
                bt_threat = self.evaluate_current_threat(converted_action, missiles)
            else:
                bt_threat = original_threat
            
            threat_reduction = ((original_threat - bt_threat) / original_threat * 100) if original_threat > 0 else 0
            
            return {
                'mode': 'behavior_tree',
                'threat_analysis': threat_analysis,
                'behavior_explanation': result.get('explanation', '无解释信息') if isinstance(result, dict) else '行为树决策',
                'effectiveness': {
                    'original_threat': original_threat,
                    'bt_threat': bt_threat,
                    'threat_reduction_percent': threat_reduction
                },
                'action_result': action_result.tolist() if hasattr(action_result, 'tolist') else action_result,
                'comprehensive_text': f"行为树决策系统识别出最高威胁导弹并生成逃逸策略。"
                                   f"威胁度从{original_threat:.4f}降低到{bt_threat:.4f}，"
                                   f"降低幅度为{threat_reduction:.1f}%。{result.get('explanation', '') if isinstance(result, dict) else ''}"
            }
        else:
            return {
                'mode': self.mode,
                'error': f'未知的运行模式: {self.mode}'
            }


def create_defend_system_from_config(config_path: str):
    """
    从配置文件创建防御专家系统
    
    参数:
        config_path: 配置文件路径
    
    返回:
        DefendExpertSystem实例
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 构建威胁评估模型
    threat_model = None
    if 'threat_model' in config:
        threat_model = build(config, 'threat_model')
    
    # 构建优化模块
    optimization_module = None
    if 'optimization_module' in config:
        optimization_module = build(config, 'optimization_module')
    
    # 创建系统
    system = DefendExpertSystem(
        threat_model=threat_model,
        optimization_module=optimization_module,
        **config.get('system_config', {})
    )
    
    return system


# 保持向后兼容
def create_de_system_from_config(config_path: str):
    """
    从配置文件创建DE专家系统（兼容性函数）
    """
    return create_defend_system_from_config(config_path)


# 便捷函数
def optimize_aircraft_position(aircraft, missiles, **de_config):
    """
    便捷函数：优化飞机位置以最小化威胁度
    
    参数:
        aircraft: 飞机状态字典
        missiles: 导弹列表
        de_config: DE优化配置
    
    返回:
        优化后的飞机状态
    """
    system = DefendExpertSystem(mode="de", de_config=de_config)
    state = {'aircraft': aircraft, 'missiles': missiles}
    return system.get_action_only(state)


def evade_with_behavior_tree(aircraft, missiles, **bt_config):
    """
    便捷函数：使用行为树模式进行威胁规避
    
    参数:
        aircraft: 飞机状态字典
        missiles: 导弹列表
        bt_config: 行为树配置
    
    返回:
        包含action和explanation的字典
    """
    system = DefendExpertSystem(mode="bt", bt_config=bt_config)
    state = {'aircraft': aircraft, 'missiles': missiles}
    return system.process(state)


if __name__ == "__main__":
    # 示例使用
    print("防御专家系统示例（支持DE优化和行为树模式）:")
    print("注意：输入数据位置单位为千米，高度为米，速度为米/秒")
    
    # 示例数据（位置单位：千米）
    aircraft = {
        "position": [-59.357, -92.518],     # x, z坐标（千米）
        "height": 15002,                     # 高度（米）
        "speed": 390.5,                     # 速度（米/秒）
        "heading": 173                      # 朝向（度）
    }
    
    missiles = [
        {
            "position": [-19.77, 68.34],    # x, z坐标（千米）
            "height": 9798,                 # 高度（米）
            "speed": 259.6                  # 速度（米/秒）
        },
        {
            "position": [-48.065, -33.342], # x, z坐标（千米）
            "height": 19858,                # 高度（米）
            "speed": 657.0                  # 速度（米/秒）
        }
    ]
    
    print("\n" + "="*60)
    print("模式1: DE优化模式")
    print("="*60)
    
    # 创建DE专家系统
    de_config = {
        'pop_size': 30,
        'max_iter': 100,  # 减少迭代次数以便演示
        'F': 0.8,
        'CR': 0.7
    }
    
    de_system = DefendExpertSystem(mode="de", de_config=de_config, verbose=True)
    
    # 威胁分析
    print("\n=== 威胁分析 ===")
    threat_analysis = de_system.get_threat_analysis(aircraft, missiles)
    print(f"导弹总数: {threat_analysis['total_missiles']}")
    print(f"最大威胁度: {threat_analysis['max_threat']:.4f}")
    print(f"威胁排名: {threat_analysis['threat_rankings']}")
    print(f"高威胁导弹: {threat_analysis['high_threat_missiles']}")
    
    # 执行优化
    print("\n=== 执行DE优化 ===")
    state = {'aircraft': aircraft, 'missiles': missiles}
    optimized_aircraft = de_system.process(state)
    
    print(f"原始飞机状态: {aircraft}")
    print(f"优化后状态: {optimized_aircraft}")
    
    # 对比威胁度
    original_threat = de_system.evaluate_current_threat(aircraft, missiles)
    optimized_threat = de_system.evaluate_current_threat(optimized_aircraft, missiles)
    
    print(f"\n=== DE优化效果 ===")
    print(f"原始威胁度: {original_threat:.4f}")
    print(f"优化后威胁度: {optimized_threat:.4f}")
    print(f"威胁度降低: {((original_threat - optimized_threat) / original_threat * 100):.2f}%")
    
    print("\n" + "="*60)
    print("模式2: 行为树模式")
    print("="*60)
    
    # 创建行为树专家系统
    bt_config = {
        'evasion_distance': 20.0,
        'height_gain': 1000.0,
        'speed_boost': 1.4,
        'max_height': 15000.0,
        'max_speed': 400.0
    }
    
    bt_system = DefendExpertSystem(mode="bt", bt_config=bt_config, verbose=True)
    
    # 执行行为树决策
    print("\n=== 执行行为树决策 ===")
    bt_result = bt_system.process(state)
    
    print(f"原始飞机状态: {aircraft}")
    print(f"行为树动作结果: {bt_result['action']}")
    print(f"决策解释: {bt_result['explanation']}")
    
    # 对比威胁度
    bt_threat = bt_system.evaluate_current_threat(bt_result['action'], missiles)
    
    print(f"\n=== 行为树效果 ===")
    print(f"原始威胁度: {original_threat:.4f}")
    print(f"行为树后威胁度: {bt_threat:.4f}")
    print(f"威胁度降低: {((original_threat - bt_threat) / original_threat * 100):.2f}%")
    
    print("\n" + "="*60)
    print("便捷函数演示")
    print("="*60)
    
    # 使用便捷函数
    print("\n=== 使用便捷函数 ===")
    
    # DE优化便捷函数
    quick_de_result = optimize_aircraft_position(aircraft, missiles, pop_size=20, max_iter=50)
    print(f"便捷DE优化结果: {quick_de_result}")
    
    # 行为树便捷函数
    quick_bt_result = evade_with_behavior_tree(aircraft, missiles, evasion_distance=25.0)
    print(f"便捷行为树结果: {quick_bt_result}")
    
    print("\n" + "="*60)
    print("演示完成")
    print("="*60)
