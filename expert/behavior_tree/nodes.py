"""
行为树节点定义
包含各种节点类型：选择节点、序列节点、条件节点、动作节点等
"""

import numpy as np
from enum import Enum
from abc import ABC, abstractmethod

# 默认决策标志字典
DEFAULT_DECISION_FLAGS = {
    'use_defense_decision': 0,
    'use_attack_decision': 0,
    'use_support_decision': 0,
    'use_radar_decision': 1  # 雷达决策默认开启
}


class NodeStatus(Enum):
    """节点执行状态"""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class BaseNode(ABC):
    """行为树节点基类"""
    
    def __init__(self, name="BaseNode"):
        self.name = name
        self.children = []
        self.parent = None
        
    def add_child(self, child):
        """添加子节点"""
        child.parent = self
        self.children.append(child)
        
    @abstractmethod
    def execute(self, context):
        """执行节点逻辑"""
        pass


class SelectorNode(BaseNode):
    """选择节点：依次执行子节点，直到有一个返回SUCCESS或RUNNING"""
    
    def execute(self, context):
        for child in self.children:
            status = child.execute(context)
            if status != NodeStatus.FAILURE:
                return status
        return NodeStatus.FAILURE


class SequenceNode(BaseNode):
    """序列节点：依次执行子节点，直到有一个返回FAILURE或RUNNING"""
    
    def execute(self, context):
        for child in self.children:
            status = child.execute(context)
            if status != NodeStatus.SUCCESS:
                return status
        return NodeStatus.SUCCESS


class ParallelNode(BaseNode):
    """并行节点：同时执行所有子节点，只要有一个成功就返回SUCCESS"""
    
    def execute(self, context):
        success_count = 0
        failure_count = 0
        
        for child in self.children:
            status = child.execute(context)
            if status == NodeStatus.SUCCESS:
                success_count += 1
            elif status == NodeStatus.FAILURE:
                failure_count += 1
        
        # 只要有一个成功就算成功
        if success_count > 0:
            return NodeStatus.SUCCESS
        elif failure_count == len(self.children):
            return NodeStatus.FAILURE
        else:
            return NodeStatus.RUNNING


class ConditionNode(BaseNode):
    """条件节点：检查特定条件"""
    
    def __init__(self, name, condition_func):
        super().__init__(name)
        self.condition_func = condition_func
        
    def execute(self, context):
        if self.condition_func(context):
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE


class ActionNode(BaseNode):
    """动作节点：执行具体动作"""
    
    def __init__(self, name, action_func):
        super().__init__(name)
        self.action_func = action_func
        
    def execute(self, context):
        try:
            result = self.action_func(context)
            if result:
                return NodeStatus.SUCCESS
            return NodeStatus.FAILURE
        except Exception as e:
            print(f"动作节点 {self.name} 执行失败: {e}")
            return NodeStatus.FAILURE


# ===== 特定的条件节点 =====

class IsAdvantageCondition(ConditionNode):
    """检查友机是否处于优势态势"""
    
    def __init__(self, aircraft_id):
        super().__init__(f"IsAdvantage_Aircraft_{aircraft_id}", None)
        self.aircraft_id = aircraft_id
        
    def execute(self, context):
        situations = context.get('friendly_situations', [])
        if self.aircraft_id < len(situations):
            return NodeStatus.SUCCESS if situations[self.aircraft_id] == "advantage" else NodeStatus.FAILURE
        return NodeStatus.FAILURE


class IsDisadvantageCondition(ConditionNode):
    """检查友机是否处于劣势态势"""
    
    def __init__(self, aircraft_id):
        super().__init__(f"IsDisadvantage_Aircraft_{aircraft_id}", None)
        self.aircraft_id = aircraft_id
        
    def execute(self, context):
        situations = context.get('friendly_situations', [])
        if self.aircraft_id < len(situations):
            return NodeStatus.SUCCESS if situations[self.aircraft_id] == "disadvantage" else NodeStatus.FAILURE
        return NodeStatus.FAILURE


class HasOptimalTargetCondition(ConditionNode):
    """检查友机是否有最佳攻击目标"""
    
    def __init__(self, aircraft_id):
        super().__init__(f"HasOptimalTarget_Aircraft_{aircraft_id}", None)
        self.aircraft_id = aircraft_id
        
    def execute(self, context):
        optimal_targets = context.get('optimal_targets', [])
        if self.aircraft_id < len(optimal_targets):
            return NodeStatus.SUCCESS if optimal_targets[self.aircraft_id] is not None else NodeStatus.FAILURE
        return NodeStatus.FAILURE


class ShouldStrikeCondition(ConditionNode):
    """检查友机是否应该执行打击任务"""
    
    def __init__(self, aircraft_id):
        super().__init__(f"ShouldStrike_Aircraft_{aircraft_id}", None)
        self.aircraft_id = aircraft_id
        
    def execute(self, context):
        # 获取态势信息
        situations = context.get('friendly_situations', [])
        optimal_targets = context.get('optimal_targets', [])
        engine = context.get('engine')
        
        if self.aircraft_id >= len(situations) or self.aircraft_id >= len(optimal_targets):
            return NodeStatus.FAILURE
        
        # 条件1: 处于优势态势
        if situations[self.aircraft_id] != "advantage":
            return NodeStatus.FAILURE
        
        # 条件2: 有可用的最优目标
        target_id = optimal_targets[self.aircraft_id]
        if target_id is None:
            return NodeStatus.FAILURE
        
        # 条件3: 目标未被过度锁定（锁定数量小于阈值）
        if engine:
            # 从配置获取目标锁定阈值
            config = context.get('config', {})
            behavior_config = config.get('behavior_tree', {})
            lock_threshold = behavior_config.get('target_lock_threshold', 2)
            
            lock_count = engine.get_target_lock_count(target_id)
            if lock_count >= lock_threshold:
                return NodeStatus.FAILURE
        
        return NodeStatus.SUCCESS


# ===== 特定的动作节点 =====

class AttackOptimalTargetAction(ActionNode):
    """攻击最佳目标动作"""
    
    def __init__(self, aircraft_id):
        super().__init__(f"AttackOptimalTarget_Aircraft_{aircraft_id}", None)
        self.aircraft_id = aircraft_id
        
    def execute(self, context):
        try:
            optimal_targets = context.get('optimal_targets', [])
            situation_details = context.get('situation_details', [])
            state = context.get('state')
            
            if self.aircraft_id >= len(optimal_targets) or self.aircraft_id >= len(situation_details):
                return NodeStatus.FAILURE
            
            
            target_id = optimal_targets[self.aircraft_id]
            aircraft_detail = situation_details[self.aircraft_id]
            engine = context.get('engine')
            
            # 更新共享打击链表
            if engine:
                engine.update_target_locks(self.aircraft_id, target_id)
            
            # 生成攻击动作
            action = self._generate_attack_action(state, self.aircraft_id, target_id)
            # 生成解释
            advantage_value = aircraft_detail.get('optimal_target_advantage', 0.0)
            explanation = (f"{state["our_aircrafts"][self.aircraft_id]["name"]}处于优势态势(态势值:{advantage_value:.3f})，"
                         f"选择攻击最佳目标敌机{target_id}。"
                         f"动作标志：机动={action.get('do_maneuver', 0)}，"
                         f"发射={action.get('launch_missile', 0)}")
            
            # 保存结果到上下文
            if 'actions' not in context:
                context['actions'] = {}
            if 'explanations' not in context:
                context['explanations'] = {}
                
            context['actions'][self.aircraft_id] = action
            context['explanations'][self.aircraft_id] = explanation
            
            return NodeStatus.SUCCESS
            
        except Exception as e:
            print(f"攻击动作节点执行失败: {e}")
            return NodeStatus.FAILURE
    
    def _generate_attack_action(self, state, aircraft_id, target_id):
        """生成优化的攻击动作
        改进点：
        1. 增加预测拦截点计算（考虑目标运动）
        2. 优化高度/速度策略（能量管理）
        3. 引入射击条件判断
        4. 位置计算与航向协同优化
        5. 机动策略增强（保持优势位置）
        """
        # 安全获取飞机状态
        our_aircrafts = state.get('our_aircrafts', [])
        enemies = state.get('enemies', [])
        
        if aircraft_id >= len(our_aircrafts) or target_id >= len(enemies):
            return self._get_default_action(aircraft_id, our_aircrafts)
        
        our_aircraft = our_aircrafts[aircraft_id]
        target_enemy = enemies[target_id]
        # 计算预测拦截点
        intercept_point = self._calculate_intercept_point(our_aircraft, target_enemy)
        # 计算攻击航向（指向拦截点）
        dx = intercept_point[0] - our_aircraft['position'][0]
        dy = intercept_point[1] - our_aircraft['position'][1]
        attack_heading = np.arctan2(dy, dx) * 180 / np.pi
        # 优化高度策略（获取高度优势）
        height_adjustment = self._calculate_attack_height_adjustment(our_aircraft, target_enemy)
        # 优化速度策略（根据距离调整）
        speed_adjustment = self._calculate_attack_speed_adjustment(our_aircraft, target_enemy)
        # 计算攻击位置（沿攻击航向移动）
        attack_distance = min(our_aircraft['speed'] * 5, 10000)/1000  # 5秒飞行距离
        attack_heading_rad = np.deg2rad(attack_heading)
        attack_position = our_aircraft['position'] + np.array([
            np.cos(attack_heading_rad) * attack_distance,
            np.sin(attack_heading_rad) * attack_distance
        ])
        # 判断是否满足发射条件
        should_launch = self._should_launch_missile(our_aircraft, target_enemy)
        # 构建优化的攻击动作
        action = {
            'speed': np.clip(our_aircraft['speed'] + speed_adjustment, 200, 450),
            'height': np.clip(our_aircraft['height'] + height_adjustment, 3000, 15000),
            'heading': attack_heading,
            'position': attack_position,
            'do_maneuver': 1,  # 攻击动作：执行机动
            'launch_missile': 1 if should_launch else 0,  # 智能发射决策
            # 添加决策标志：攻击决策
            'use_defense_decision': 0,
            'use_attack_decision': 1,
            'use_support_decision': 0,
            'use_radar_decision': 1
        }
        return action

    def _calculate_intercept_point(self, our_aircraft, target_enemy):
        """计算预测拦截点（考虑目标运动）"""
        # 将位置从km转换为m
        our_pos_m = our_aircraft['position'] * 1000
        target_pos_m = target_enemy['position'] * 1000
        
        # 计算目标速度向量（单位：m/s）
        target_speed = target_enemy.get('speed', 300)
        target_heading = target_enemy.get('heading', 0)
        target_heading_rad = np.deg2rad(target_heading)
        target_velocity = np.array([
            np.cos(target_heading_rad) * target_speed,
            np.sin(target_heading_rad) * target_speed
        ])
        
        # 计算己方速度向量（单位：m/s）
        our_heading_rad = np.deg2rad(our_aircraft['heading'])
        our_velocity = np.array([
            np.cos(our_heading_rad) * our_aircraft['speed'],
            np.sin(our_heading_rad) * our_aircraft['speed']
        ])
        
        # 计算相对位置（单位：m）和相对速度（单位：m/s）
        relative_pos_m = target_pos_m - our_pos_m
        relative_vel = target_velocity - our_velocity
        
        # 计算拦截时间（单位：秒）
        distance_m = np.linalg.norm(relative_pos_m)
        if distance_m > 0:
            # 计算接近速度（标量，单位：m/s）
            closing_speed = -np.dot(relative_pos_m, relative_vel) / distance_m
            closing_speed = max(closing_speed, 100)  # 确保最小接近速度
            intercept_time = min(distance_m / closing_speed, 15)  # 上限15秒
        else:
            intercept_time = 0
        
        # 计算预测位置（单位：m）并转换回km
        predicted_pos_m = target_pos_m + target_velocity * intercept_time
        return predicted_pos_m / 1000  # 转换回km

    def _calculate_attack_height_adjustment(self, our_aircraft, target_enemy):
        """计算攻击高度调整策略"""
        height_diff = target_enemy.get('height') - our_aircraft['height']
        
        # 高度优势策略
        if height_diff > 2000:  # 目标高出2000米以上
            return 800  # 紧急爬升
        elif height_diff > 0:   # 目标略高
            return 500
        elif height_diff < -1000:  # 我们有高度优势
            return 0  # 保持或略微下降
        
        # 默认小幅爬升
        return 200

    def _calculate_attack_speed_adjustment(self, our_aircraft, target_enemy):
        """计算攻击速度调整策略"""
        distance = np.linalg.norm(target_enemy['position'] - our_aircraft['position'])
        
        if distance > 200:  # 远距离（200km）
            return 50  # 加速接近
        elif distance > 100:  # 中距离
            return 30
        elif distance > 5:   # 近距离
            return 0 if our_aircraft['speed'] > 350 else 10
        else:  # 极近距离
            return -20  # 减速保持机动性

    def _should_launch_missile(self, our_aircraft, target_enemy):
        """判断是否满足导弹发射条件"""
        distance = np.linalg.norm(target_enemy['position'] - our_aircraft['position'])
        
        # 计算目标相对角度（导弹导引头视角）
        dx = target_enemy['position'][0] - our_aircraft['position'][0]
        dy = target_enemy['position'][1] - our_aircraft['position'][1]
        target_bearing = np.arctan2(dy, dx) * 180 / np.pi
        angle_diff = abs((target_bearing - our_aircraft['heading']) % 360)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # 发射条件判断
        return (
            distance < 200 and  # 在射程内（200km ）
            distance > 0 and    # 最小射程
            angle_diff < 60 and    # 目标在导引头视角内
            our_aircraft['height'] > 3000 # 安全高度
        )

    def _get_default_action(self, aircraft_id, our_aircrafts):
        """安全获取默认动作（保持状态）"""
        if aircraft_id < len(our_aircrafts):
            ac = our_aircrafts[aircraft_id]
            return {
                'speed': ac['speed'],
                'height': ac['height'],
                'heading': ac['heading'],
                'position': np.copy(ac['position'])
            }
        return {
            'speed': 350,
            'height': 8000,
            'heading': 0,
            'position': np.array([0, 0])
        }

class EvadeMaxThreatAction(ActionNode):
    """规避最大威胁动作"""
    
    def __init__(self, aircraft_id):
        super().__init__(f"EvadeMaxThreat_Aircraft_{aircraft_id}", None)
        self.aircraft_id = aircraft_id
        
    def execute(self, context):
        try:
            max_threat_enemies = context.get('max_threat_enemies', [])
            situation_details = context.get('situation_details', [])
            state = context.get('state')
            
            if self.aircraft_id >= len(max_threat_enemies) or self.aircraft_id >= len(situation_details):
                return NodeStatus.FAILURE
                
            threat_enemy_id = max_threat_enemies[self.aircraft_id]
            aircraft_detail = situation_details[self.aircraft_id]
            
            # 生成规避动作
            action = self._generate_evasion_action(state, self.aircraft_id, threat_enemy_id)
            
            # 生成解释
            threat_value = aircraft_detail.get('max_threat_value', 0.0)
            explanation = (f"{state["our_aircrafts"][self.aircraft_id]["name"]}处于劣势态势，面临敌机{threat_enemy_id}的高威胁"
                         f"(威胁值:{threat_value:.3f})，执行规避机动提升态势。"
                         f"动作标志：机动={action.get('do_maneuver', 0)}，"
                         f"发射={action.get('launch_missile', 0)}")
            
            # 保存结果到上下文
            if 'actions' not in context:
                context['actions'] = {}
            if 'explanations' not in context:
                context['explanations'] = {}
                
            context['actions'][self.aircraft_id] = action
            context['explanations'][self.aircraft_id] = explanation
            
            return NodeStatus.SUCCESS
            
        except Exception as e:
            print(f"规避动作节点执行失败: {e}")
            return NodeStatus.FAILURE
    
    def _generate_evasion_action(self, state, aircraft_id, threat_enemy_id):
        """生成规避动作
        关键改进：
        1. 位置参数改为沿规避航向的前进位置（而非绝对偏移）
        2. 航向计算考虑当前航向与目标航向的差值
        3. 速度变化与航向变化协同优化
        4. 增加规避距离的动态计算
        """
        # 安全获取飞机状态
        our_aircrafts = state.get('our_aircrafts', [])
        enemies = state.get('enemies', [])
        
        if aircraft_id >= len(our_aircrafts) or threat_enemy_id >= len(enemies):
            return self._get_default_action(aircraft_id, our_aircrafts)
        
        our_aircraft = our_aircrafts[aircraft_id]
        threat_enemy = enemies[threat_enemy_id]
        
        # 计算威胁向量和方向
        threat_vector = threat_enemy['position'] - our_aircraft['position']
        threat_distance = np.linalg.norm(threat_vector)
        threat_direction = np.arctan2(threat_vector[1], threat_vector[0]) * 180 / np.pi
        
        # 动态计算规避参数
        speed_change = self._calculate_speed_change(threat_distance)
        height_change = self._calculate_height_change(threat_distance)
        evasion_distance = self._calculate_evasion_distance(threat_distance, our_aircraft['speed'])
        
        # 随机选择规避方向（左转或右转）
        evasion_direction = np.random.choice([-1, 1])
        evasion_heading = (threat_direction + 90 * evasion_direction) % 360
        
        # 计算沿规避航向前进的位置（关键改进）
        evasion_heading_rad = np.deg2rad(evasion_heading)
        new_position = our_aircraft['position'] + np.array([
            np.cos(evasion_heading_rad) * evasion_distance,
            np.sin(evasion_heading_rad) * evasion_distance
        ])
        
        # 构建与仿真引擎匹配的动作
        action = {
            'speed': np.clip(our_aircraft['speed'] + speed_change, 200, 400),
            'height': np.clip(our_aircraft['height'] + height_change, 3000, 15000),
            'heading': evasion_heading,  # 绝对目标航向
            'position': new_position,    # 沿规避航向的目标位置
            'do_maneuver': 1,  # 攻击动作：执行机动
            'launch_missile': 1,  # 攻击动作：发射导弹
            # 添加决策标志：规避动作（归类为防御决策）
            'use_defense_decision': 1,
            'use_attack_decision': 0,
            'use_support_decision': 0,
            'use_radar_decision': 1
        }
        return action

    def _calculate_evasion_distance(self, threat_distance, current_speed):
        """动态计算规避移动距离（基于威胁距离和速度）"""
        # 基础距离：5秒飞行距离
        base_distance = (current_speed * 5)/1000
        
        if threat_distance < 5:  # 紧急规避
            return base_distance * 1.5
        elif threat_distance < 15:  # 中等规避
            return base_distance
        return base_distance * 0.7  # 远距离规避

    def _calculate_speed_change(self, threat_distance):
        """速度变化计算（与航向变化协同）"""
        if threat_distance < 5:  # 紧急规避
            return 80  # 大幅加速
        elif threat_distance < 15:
            return 40
        return 20  # 远距离小幅加速

    def _calculate_height_change(self, threat_distance):
        """高度变化计算（考虑爬升/俯冲性能）"""
        if threat_distance < 5:  # 紧急规避
            return np.random.choice([1500, -1000])  # 随机爬升或俯冲
        return 800  # 常规爬升

    def _get_default_action(self, aircraft_id, our_aircrafts):
        """安全获取默认动作（保持状态）"""
        if aircraft_id < len(our_aircrafts):
            ac = our_aircrafts[aircraft_id]
            return {
                'speed': ac['speed'],
                'height': ac['height'],
                'heading': ac['heading'],
                'position': np.copy(ac['position']),
                'do_maneuver': 1,  # 攻击动作：执行机动
                'launch_missile': 1  # 攻击动作：发射导弹
            }
        return {
            'speed': 350,
            'height': 8000,
            'heading': 0,
            'position': np.array([0, 0]),
            'do_maneuver': 1,  # 攻击动作：执行机动
            'launch_missile': 1  # 攻击动作：发射导弹
        }


# ===== 支援相关的动作节点 =====

class SupportTargetSelectionAction(ActionNode):
    """支援目标选择动作"""
    
    def __init__(self, aircraft_id):
        super().__init__(f"SupportTargetSelection_Aircraft_{aircraft_id}", None)
        self.aircraft_id = aircraft_id
        
    def execute(self, context):
        try:
            engine = context.get('engine')
            if not engine:
                return NodeStatus.FAILURE
                
            # 获取可用的支援目标
            support_targets = engine.get_available_support_targets(self.aircraft_id)
            
            if not support_targets:
                return NodeStatus.FAILURE
                
            # 选择优先级最高的支援目标
            selected_target_id = support_targets[0][0]
            
            # 将支援目标信息存储到上下文中，供后续支援动作使用
            if 'support_info' not in context:
                context['support_info'] = {}
            context['support_info'][self.aircraft_id] = {
                'target_aircraft_id': selected_target_id,
                'priority': support_targets[0][1]
            }
            
            return NodeStatus.SUCCESS
            
        except Exception as e:
            print(f"支援目标选择动作节点执行失败: {e}")
            return NodeStatus.FAILURE


class AttackSupportTargetBestTargetAction(ActionNode):
    """攻击支援目标的最佳攻击目标"""
    
    def __init__(self, aircraft_id):
        super().__init__(f"AttackSupportTargetBestTarget_Aircraft_{aircraft_id}", None)
        self.aircraft_id = aircraft_id
        
    def execute(self, context):
        try:
            # 获取支援信息
            support_info = context.get('support_info', {}).get(self.aircraft_id)
            if not support_info:
                return NodeStatus.FAILURE
                
            target_aircraft_id = support_info['target_aircraft_id']
            optimal_targets = context.get('optimal_targets', [])
            state = context.get('state')
            engine = context.get('engine')
            
            # 获取支援目标的最佳攻击目标
            if target_aircraft_id >= len(optimal_targets):
                return NodeStatus.FAILURE
                
            support_target_optimal_enemy = optimal_targets[target_aircraft_id]
            if support_target_optimal_enemy is None:
                return NodeStatus.FAILURE
                
            # 检查该目标是否已被过度锁定
            if engine:
                # 从配置获取目标锁定阈值
                config = context.get('config', {})
                behavior_config = config.get('behavior_tree', {})
                lock_threshold = behavior_config.get('target_lock_threshold', 2)
                
                lock_count = engine.get_target_lock_count(support_target_optimal_enemy)
                if lock_count >= lock_threshold:
                    return NodeStatus.FAILURE
                    
                # 更新目标锁定
                engine.update_target_locks(self.aircraft_id, support_target_optimal_enemy)
            
            # 生成攻击动作
            action = self._generate_support_attack_action(state, self.aircraft_id, support_target_optimal_enemy)
            
            # 生成解释
            support_aircraft_name = state["our_aircrafts"][target_aircraft_id]["name"]
            explanation = (f"{state['our_aircrafts'][self.aircraft_id]['name']}支援{support_aircraft_name}，"
                         f"攻击其最佳目标敌机{support_target_optimal_enemy}。"
                         f"动作标志：机动={action.get('do_maneuver', 0)}，"
                         f"发射={action.get('launch_missile', 0)}")
            
            # 保存结果到上下文
            if 'actions' not in context:
                context['actions'] = {}
            if 'explanations' not in context:
                context['explanations'] = {}
                
            context['actions'][self.aircraft_id] = action
            context['explanations'][self.aircraft_id] = explanation
            
            return NodeStatus.SUCCESS
            
        except Exception as e:
            print(f"攻击支援目标最佳目标动作节点执行失败: {e}")
            return NodeStatus.FAILURE
    
    def _generate_support_attack_action(self, state, aircraft_id, target_id):
        """生成支援攻击动作"""
        our_aircrafts = state.get('our_aircrafts', [])
        enemies = state.get('enemies', [])
        
        if aircraft_id >= len(our_aircrafts) or target_id >= len(enemies):
            return self._get_default_support_action(aircraft_id, our_aircrafts)
        
        our_aircraft = our_aircrafts[aircraft_id]
        target_enemy = enemies[target_id]
        
        # 计算攻击航向
        dx = target_enemy['position'][0] - our_aircraft['position'][0]
        dy = target_enemy['position'][1] - our_aircraft['position'][1]
        attack_heading = np.arctan2(dy, dx) * 180 / np.pi
        
        # 计算攻击位置
        attack_distance = min(our_aircraft['speed'] * 5, 10000) / 1000
        attack_heading_rad = np.deg2rad(attack_heading)
        attack_position = our_aircraft['position'] + np.array([
            np.cos(attack_heading_rad) * attack_distance,
            np.sin(attack_heading_rad) * attack_distance
        ])
        
        # 判断是否发射导弹
        distance = np.linalg.norm(target_enemy['position'] - our_aircraft['position'])
        should_launch = distance < 150 and distance > 5
        
        action = {
            'speed': np.clip(our_aircraft['speed'] + 30, 200, 450),
            'height': np.clip(our_aircraft['height'] + 500, 3000, 15000),
            'heading': attack_heading,
            'position': attack_position,
            'do_maneuver': 1,
            'launch_missile': 1 if should_launch else 0,
            'use_defense_decision': 0,
            'use_attack_decision': 0,
            'use_support_decision': 1,
            'use_radar_decision': 1
        }
        return action
    
    def _get_default_support_action(self, aircraft_id, our_aircrafts):
        """获取默认支援动作"""
        if aircraft_id < len(our_aircrafts):
            ac = our_aircrafts[aircraft_id]
            return {
                'speed': ac['speed'],
                'height': ac['height'],
                'heading': ac['heading'],
                'position': np.copy(ac['position']),
                'do_maneuver': 1,
                'launch_missile': 0
            }
        return {
            'speed': 350,
            'height': 8000,
            'heading': 0,
            'position': np.array([0, 0]),
            'do_maneuver': 1,
            'launch_missile': 0
        }


class AttackSupportTargetMaxThreatAction(ActionNode):
    """攻击支援目标的最大威胁目标"""
    
    def __init__(self, aircraft_id):
        super().__init__(f"AttackSupportTargetMaxThreat_Aircraft_{aircraft_id}", None)
        self.aircraft_id = aircraft_id
        
    def execute(self, context):
        try:
            # 获取支援信息
            support_info = context.get('support_info', {}).get(self.aircraft_id)
            if not support_info:
                return NodeStatus.FAILURE
                
            target_aircraft_id = support_info['target_aircraft_id']
            max_threat_enemies = context.get('max_threat_enemies', [])
            state = context.get('state')
            engine = context.get('engine')
            
            # 获取支援目标的最大威胁
            if target_aircraft_id >= len(max_threat_enemies):
                return NodeStatus.FAILURE
                
            threat_enemy_id = max_threat_enemies[target_aircraft_id]
            if threat_enemy_id is None:
                return NodeStatus.FAILURE
            
            # 检查该威胁目标是否已被过度锁定
            if engine:
                # 从配置获取目标锁定阈值
                config = context.get('config', {})
                behavior_config = config.get('behavior_tree', {})
                lock_threshold = behavior_config.get('target_lock_threshold', 2)
                
                lock_count = engine.get_target_lock_count(threat_enemy_id)
                if lock_count >= lock_threshold:
                    return NodeStatus.FAILURE
                    
                # 更新目标锁定
                engine.update_target_locks(self.aircraft_id, threat_enemy_id)
            
            # 生成攻击威胁目标的动作
            action = self._generate_threat_attack_action(state, self.aircraft_id, threat_enemy_id)
            
            # 生成解释
            support_aircraft_name = state["our_aircrafts"][target_aircraft_id]["name"]
            explanation = (f"{state['our_aircrafts'][self.aircraft_id]['name']}支援{support_aircraft_name}，"
                         f"攻击威胁其的敌机{threat_enemy_id}。"
                         f"动作标志：机动={action.get('do_maneuver', 0)}，"
                         f"发射={action.get('launch_missile', 0)}")
            
            # 保存结果到上下文
            if 'actions' not in context:
                context['actions'] = {}
            if 'explanations' not in context:
                context['explanations'] = {}
                
            context['actions'][self.aircraft_id] = action
            context['explanations'][self.aircraft_id] = explanation
            
            return NodeStatus.SUCCESS
            
        except Exception as e:
            print(f"攻击支援目标最大威胁动作节点执行失败: {e}")
            return NodeStatus.FAILURE
    
    def _generate_threat_attack_action(self, state, aircraft_id, threat_enemy_id):
        """生成威胁攻击动作"""
        our_aircrafts = state.get('our_aircrafts', [])
        enemies = state.get('enemies', [])
        
        if aircraft_id >= len(our_aircrafts) or threat_enemy_id >= len(enemies):
            return self._get_default_support_action(aircraft_id, our_aircrafts)
        
        our_aircraft = our_aircrafts[aircraft_id]
        threat_enemy = enemies[threat_enemy_id]
        
        # 计算攻击航向
        dx = threat_enemy['position'][0] - our_aircraft['position'][0]
        dy = threat_enemy['position'][1] - our_aircraft['position'][1]
        attack_heading = np.arctan2(dy, dx) * 180 / np.pi
        
        # 计算攻击位置
        attack_distance = min(our_aircraft['speed'] * 4, 8000) / 1000
        attack_heading_rad = np.deg2rad(attack_heading)
        attack_position = our_aircraft['position'] + np.array([
            np.cos(attack_heading_rad) * attack_distance,
            np.sin(attack_heading_rad) * attack_distance
        ])
        
        # 判断是否发射导弹（威胁目标优先级高）
        distance = np.linalg.norm(threat_enemy['position'] - our_aircraft['position'])
        should_launch = distance < 180 and distance > 3
        
        action = {
            'speed': np.clip(our_aircraft['speed'] + 40, 200, 450),
            'height': np.clip(our_aircraft['height'] + 600, 3000, 15000),
            'heading': attack_heading,
            'position': attack_position,
            'do_maneuver': 1,
            'launch_missile': 1 if should_launch else 0,
            'use_defense_decision': 0,
            'use_attack_decision': 1,
            'use_support_decision': 1,
            'use_radar_decision': 1
        }
        return action
    
    def _get_default_support_action(self, aircraft_id, our_aircrafts):
        """获取默认支援动作"""
        if aircraft_id < len(our_aircrafts):
            ac = our_aircrafts[aircraft_id]
            return {
                'speed': ac['speed'],
                'height': ac['height'],
                'heading': ac['heading'],
                'position': np.copy(ac['position']),
                'do_maneuver': 1,
                'launch_missile': 0
            }
        return {
            'speed': 350,
            'height': 8000,
            'heading': 0,
            'position': np.array([0, 0]),
            'do_maneuver': 1,
            'launch_missile': 0
        }


class ImprovePositionAction(ActionNode):
    """改善位置动作（中性态势）"""
    
    def __init__(self, aircraft_id):
        super().__init__(f"ImprovePosition_Aircraft_{aircraft_id}", None)
        self.aircraft_id = aircraft_id
        
    def execute(self, context):
        try:
            situation_details = context.get('situation_details', [])
            state = context.get('state')
            
            if self.aircraft_id >= len(situation_details):
                return NodeStatus.FAILURE
                
            aircraft_detail = situation_details[self.aircraft_id]
            
            # 生成位置改善动作
            action = self._generate_improvement_action(state, self.aircraft_id)
            
            # 生成解释
            avg_situation = aircraft_detail.get('average_situation_value', 0.0)
            explanation = (f"{state["our_aircrafts"][self.aircraft_id]["name"]}处于中性态势(态势值:{avg_situation:.3f})，"
                         f"执行位置调整以寻求更佳态势。"
                         f"动作标志：机动={action.get('do_maneuver', 0)},"
                         f"发射={action.get('launch_missile', 0)}")
            
            # 保存结果到上下文
            if 'actions' not in context:
                context['actions'] = {}
            if 'explanations' not in context:
                context['explanations'] = {}
                
            context['actions'][self.aircraft_id] = action
            context['explanations'][self.aircraft_id] = explanation
            
            return NodeStatus.SUCCESS
            
        except Exception as e:
            print(f"位置改善动作节点执行失败: {e}")
            return NodeStatus.FAILURE
    
    def _generate_improvement_action(self, state, aircraft_id):
        """生成优化的位置改善动作
        改进点：
        1. 基于战术态势的智能位置调整
        2. 航向优化朝向有利位置
        3. 速度/高度协同变化
        4. 移除武器发射决策
        5. 增加随机性避免模式化
        """
        our_aircrafts = state.get('our_aircrafts', [])
        
        if aircraft_id >= len(our_aircrafts):
            return self._get_default_action(aircraft_id, our_aircrafts)
        
        our_aircraft = our_aircrafts[aircraft_id]
        current_pos = our_aircraft['position']
        current_heading = our_aircraft['heading']
        
        # 计算战术优化位置（关键改进）
        target_position = self._calculate_optimal_position(state, aircraft_id)
        
        # 计算指向目标位置的航向
        dx = target_position[0] - current_pos[0]
        dy = target_position[1] - current_pos[1]
        target_heading = np.arctan2(dy, dx) * 180 / np.pi
        
        # 计算航向变化量（限制在合理范围内）
        heading_diff = (target_heading - current_heading) % 360
        if heading_diff > 180:
            heading_diff -= 360
        
        # 限制最大转向角度（避免过度机动）
        max_turn = 30  # 最大30度调整
        heading_adjustment = np.clip(heading_diff, -max_turn, max_turn)
        new_heading = (current_heading + heading_adjustment) % 360
        
        # 计算速度和高度变化（战术优化）
        speed_change = self._calculate_speed_adjustment(state, aircraft_id)
        height_change = self._calculate_height_adjustment(state, aircraft_id)
        
        # 构建优化的位置改善动作
        action = {
            'speed': np.clip(our_aircraft['speed'] + speed_change, 200, 400),
            'height': np.clip(our_aircraft['height'] + height_change, 3000, 15000),
            'heading': new_heading,
            'position': target_position,
            'do_maneuver': 1,  # 攻击动作：执行机动
            'launch_missile': 0,  # 攻击动作：发射导弹
            # 添加决策标志：攻击和支援决策
            'use_defense_decision': 0,
            'use_attack_decision': 1,
            'use_support_decision': 1,
            'use_radar_decision': 1
        }
        return action

    def _calculate_optimal_position(self, state, aircraft_id):
        """计算战术最优位置
        策略：
        1. 优先占据高度优势位置
        2. 保持与友机的战术编队
        3. 远离已知威胁区域
        4. 朝向任务目标方向推进
        """
        our_aircraft = state['our_aircrafts'][aircraft_id]
        current_pos = our_aircraft['position']
        
        # 策略1：高度优势位置（默认向前推进）
        forward_distance = (our_aircraft['speed'] * 10)/1000  # 10秒飞行距离
        heading_rad = np.deg2rad(our_aircraft['heading'])
        base_position = current_pos + np.array([
            np.cos(heading_rad) * forward_distance,
            np.sin(heading_rad) * forward_distance
        ])
        
        # 策略2：编队位置调整（如果有友机）
        if len(state['our_aircrafts']) > 1:
            wingman_positions = [ac['position'] for i, ac in enumerate(state['our_aircrafts']) if i != aircraft_id]
            formation_offset = self._calculate_formation_offset(aircraft_id, state['our_aircrafts'])
            avg_wingman_position = np.mean(wingman_positions, axis=0) if wingman_positions else current_pos
            formation_position = avg_wingman_position + formation_offset
            # 混合基础位置和编队位置
            base_position = 0.7 * base_position + 0.3 * formation_position
        
        # # 策略3：威胁回避调整
        # if 'enemies' in state and state['enemies']:
        #     closest_threat = min(state['enemies'], key=lambda e: np.linalg.norm(e['position'] - current_pos))
        #     threat_vector = closest_threat['position'] - current_pos
        #     threat_direction = np.arctan2(threat_vector[1], threat_vector[0])
        #     # 向威胁反方向微调
        #     avoid_direction = threat_direction + np.pi  # 180度方向
        #     avoid_distance = min(20, np.linalg.norm(threat_vector))  # 动态调整距离
        #     avoid_offset = np.array([
        #         np.cos(avoid_direction) * avoid_distance,
        #         np.sin(avoid_direction) * avoid_distance
        #     ])
        #     base_position += avoid_offset
        
        # 策略4：目标导向调整
        if 'optimal_targets' in state:
            target_vector = state['optimal_targets'][aircraft_id] - current_pos
            target_direction = np.arctan2(target_vector[1], target_vector[0])
            # 向目标方向微调
            target_offset = np.array([
                np.cos(target_direction) * 1,
                np.sin(target_direction) * 1
            ])
            base_position += target_offset
        
        return base_position

    def _calculate_formation_offset(self, aircraft_id, our_aircrafts):
        """计算编队位置偏移量
        根据飞机在编队中的角色分配位置：
        - 长机(0号)：中心位置
        - 僚机(1号)：右后侧
        - 其他：左后侧
        """
        if aircraft_id == 0:  # 长机
            return np.array([0, 0])
        
        # 基础偏移量（约10000米距离）
        base_offset = np.array([-10, 0])  # 后方
        
        if aircraft_id == 1:  # 2号僚机（右后侧）
            return base_offset + np.array([-7.5, 10])
        
        # 其他飞机（左后侧）
        return base_offset + np.array([-7.5, -10])

    def _calculate_speed_adjustment(self, state, aircraft_id):
        """计算速度调整量
        策略：
        - 接近目标时减速
        - 需要快速机动时加速
        - 保持编队速度
        """
        our_aircraft = state['our_aircrafts'][aircraft_id]
        
        # 编队速度匹配
        if len(state['our_aircrafts']) > 1:
            avg_speed = np.mean([ac['speed'] for ac in state['our_aircrafts']])
            speed_diff = avg_speed - our_aircraft['speed']
            return np.clip(speed_diff, -20, 20)  # 温和调整
        
        # 目标接近减速
        if 'optimal_targets' in state:
            target_dist = np.linalg.norm(state['optimal_targets'][aircraft_id] - our_aircraft['position'])
            if target_dist < 20:  # 20km内开始减速
                return -min(30, (20 - target_dist) / 0.5)
        
        # 默认小幅加速
        return 5 if our_aircraft['speed'] < 350 else 0

    def _calculate_height_adjustment(self, state, aircraft_id):
        """计算高度调整量
        策略：
        - 接近目标时降低高度（减少暴露）
        - 巡航时保持最佳高度
        - 威胁存在时增加高度
        """
        our_aircraft = state['our_aircrafts'][aircraft_id]
        
        # 威胁存在时爬升
        if 'enemies' in state and state['enemies']:
            closest_threat = min(state['enemies'], key=lambda e: np.linalg.norm(e['position'] - our_aircraft['position']))
            threat_dist = np.linalg.norm(closest_threat['position'] - our_aircraft['position'])
            if threat_dist < 30:  # 30km内有威胁
                return 800 if our_aircraft['height'] < 10000 else 400
        
        # 目标接近时降低高度
        if 'optimal_targets' in state:
            target_dist = np.linalg.norm(state['optimal_targets'][aircraft_id] - our_aircraft['position'])
            if target_dist < 10:  # 10km内
                return -min(500, (10 - target_dist) / 20)
        
        # 保持最佳巡航高度
        optimal_height = 9000
        height_diff = optimal_height - our_aircraft['height']
        return np.clip(height_diff, -300, 300)

    def _get_default_action(self, aircraft_id, our_aircrafts):
        """安全获取默认动作（保持状态）"""
        if aircraft_id < len(our_aircrafts):
            ac = our_aircrafts[aircraft_id]
            return {
                'speed': ac['speed'],
                'height': ac['height'],
                'heading': ac['heading'],
                'position': np.copy(ac['position']),
                'do_maneuver': 1,  # 攻击动作：执行机动
                'launch_missile': 1  # 攻击动作：发射导弹
            }
        return {
            'speed': 350,
            'height': 8000,
            'heading': 0,
            'position': np.array([0, 0]),
            'do_maneuver': 1,  # 攻击动作：执行机动
            'launch_missile': 1  # 攻击动作：发射导弹
        }
