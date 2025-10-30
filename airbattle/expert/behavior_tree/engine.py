"""
行为树引擎
基于态势感知的行为树决策引擎实现
"""

import numpy as np
from .nodes import *


class BehaviorTreeEngine:
    """行为树决策引擎"""
    
    def __init__(self, config=None):
        self.root = None
        self.context = {}
        self.shared_target_locks = {}  # 共享打击链表 {target_id: [aircraft_id1, aircraft_id2, ...]}
        self.config = config or {}  # 配置参数
        
    def build_tree(self, state, threat_model):
        """
        根据当前状态和威胁模型构建行为树
        :param state: 当前环境状态
        :param threat_model: 威胁评估模型
        """
        # 每次进入行为树前清空共享打击链表
        self.shared_target_locks = {}
        
        # 获取态势数据
        threat_model.process_state(state)
        
        # 从配置获取态势阈值
        behavior_config = self.config.get('behavior_tree', {})
        situation_thresholds = behavior_config.get('situation_thresholds', {})
        advantage_threshold = situation_thresholds.get('advantage_threshold', 0.2)
        disadvantage_threshold = situation_thresholds.get('disadvantage_threshold', 0.2)
        
        # 收集所有需要的态势信息
        friendly_situations = threat_model.get_friendly_situations(
            advantage_threshold=advantage_threshold, 
            disadvantage_threshold=disadvantage_threshold
        )
        optimal_targets = threat_model.get_optimal_target_per_friendly()
        max_threat_enemies = threat_model.get_max_threat_enemy_per_friendly()
        situation_details = threat_model.get_friendly_situation_details()
        
        # 设置上下文
        self.context = {
            'state': state,
            'friendly_situations': friendly_situations,
            'optimal_targets': optimal_targets,
            'max_threat_enemies': max_threat_enemies,
            'situation_details': situation_details,
            'actions': {},
            'explanations': {},
            'engine': self,  # 将引擎实例传入上下文，供节点使用
            'config': self.config  # 将配置传入上下文，供节点使用
        }
        
        # 构建根节点（并行节点，确保所有友机都被处理）
        self.root = ParallelNode("RootParallel")
        
        # 为每架友机创建行为子树
        num_aircrafts = len(state.get('our_aircrafts', []))
        for aircraft_id in range(num_aircrafts):
            aircraft_tree = self._build_aircraft_tree(aircraft_id)
            self.root.add_child(aircraft_tree)
            
    def _build_aircraft_tree(self, aircraft_id):
        """
        为单架友机构建行为子树 - 新的二分支结构
        :param aircraft_id: 友机ID
        :return: 行为子树根节点
        """
        # 创建友机行为选择器
        aircraft_selector = SelectorNode(f"Aircraft_{aircraft_id}_Selector")
        
        # 打击分支：直接攻击最优目标
        strike_sequence = SequenceNode(f"Aircraft_{aircraft_id}_Strike")
        strike_sequence.add_child(ShouldStrikeCondition(aircraft_id))
        strike_sequence.add_child(AttackOptimalTargetAction(aircraft_id))
        
        # 支援分支：支援友机
        support_sequence = SequenceNode(f"Aircraft_{aircraft_id}_Support")
        support_sequence.add_child(SupportTargetSelectionAction(aircraft_id))
        # 支援类型选择器（简化为两种支援类型）
        support_type_selector = SelectorNode(f"Aircraft_{aircraft_id}_SupportTypeSelector")
        support_type_selector.add_child(AttackSupportTargetBestTargetAction(aircraft_id))
        support_type_selector.add_child(AttackSupportTargetMaxThreatAction(aircraft_id))
        support_sequence.add_child(support_type_selector)
        
        # 按优先级添加分支：优先打击，然后支援
        aircraft_selector.add_child(strike_sequence)
        aircraft_selector.add_child(support_sequence)
        
        return aircraft_selector
        
    def execute(self):
        """
        执行行为树
        :return: (actions_dict, explanations_dict)
        """
        if self.root is None:
            return {}, {}
            
        # 执行行为树
        status = self.root.execute(self.context)
        
        # 返回生成的动作和解释
        actions = self.context.get('actions', {})
        explanations = self.context.get('explanations', {})
        
        return actions, explanations
        
    def get_all_actions(self, state, threat_model):
        """
        获取所有友方飞机的下一步动作
        :param state: 当前环境状态
        :param threat_model: 威胁评估模型
        :return: (actions_list, explanations_list)
        """
        self.build_tree(state, threat_model)
        actions_dict, explanations_dict = self.execute()
        
        num_aircrafts = len(state.get('our_aircrafts', []))
        
        # 转换为列表格式
        actions_list = []
        explanations_list = []
        
        for aircraft_id in range(num_aircrafts):
            if aircraft_id in actions_dict:
                actions_list.append(actions_dict[aircraft_id])
                explanations_list.append(explanations_dict.get(aircraft_id, "无行为解释"))
            else:
                # 默认动作（字典格式）
                our_aircrafts = state.get('our_aircrafts', [])
                if aircraft_id < len(our_aircrafts):
                    our_aircraft = our_aircrafts[aircraft_id]
                    default_action = {
                        'speed': our_aircraft['speed'],
                        'height': our_aircraft['height'],
                        'heading': our_aircraft['heading'],
                        'position': np.array(our_aircraft['position']),
                        'do_maneuver': 1,  # 攻击系统默认动作：执行机动
                        'launch_interceptor': 1  # 攻击系统默认动作：发射导弹
                    }
                else:
                    default_action = {
                        'speed': 250,
                        'height': 8000,
                        'heading': 0,
                        'position': np.array([0, 0]),
                        'do_maneuver': 1,  # 攻击系统默认动作：执行机动
                        'launch_interceptor': 1  # 攻击系统默认动作：发射导弹
                    }
                actions_list.append(default_action)
                explanations_list.append(f"友机{aircraft_id}保持当前状态")
                
        return actions_list, explanations_list
        
    def get_actions_for_friendly(self, aircraft_id, state, threat_model):
        """
        获取指定友方飞机的下一步动作
        :param aircraft_id: 友机ID
        :param state: 当前环境状态
        :param threat_model: 威胁评估模型
        :return: (action, explanation)
        """
        actions_list, explanations_list = self.get_all_actions(state, threat_model)
        
        if aircraft_id < len(actions_list):
            return actions_list[aircraft_id], explanations_list[aircraft_id]
        else:
            # 返回默认动作（字典格式）
            default_action = {
                'speed': 250,
                'height': 8000,
                'heading': 0,
                'position': np.array([0, 0]),
                'do_maneuver': 1,  # 攻击系统默认动作：执行机动
                'launch_interceptor': 1  # 攻击系统默认动作：发射导弹
            }
            return default_action, f"友机{aircraft_id}不存在"
            
    def print_tree_structure(self, node=None, depth=0):
        """
        打印行为树结构（调试用）
        :param node: 当前节点，默认为根节点
        :param depth: 当前深度
        """
        if node is None:
            node = self.root
            
        if node is None:
            print("行为树为空")
            return
            
        indent = "  " * depth
        print(f"{indent}{node.__class__.__name__}: {node.name}")
        
        for child in node.children:
            self.print_tree_structure(child, depth + 1)
            
    def get_decision_summary(self, state, threat_model):
        """
        获取决策总结
        :param state: 当前环境状态
        :param threat_model: 威胁评估模型
        :return: 决策总结字典
        """
        actions_list, explanations_list = self.get_all_actions(state, threat_model)
        
        # 获取整体态势
        team_advantage, team_risk = threat_model.get_overall_situation()
        team_situation = threat_model.evaluate_situation(team_advantage, team_risk)
        
        summary = {
            'team_situation': {
                'status': team_situation,
                'advantage': team_advantage,
                'risk': team_risk
            },
            'individual_decisions': [],
            'total_aircrafts': len(actions_list)
        }
        
        # 添加每架飞机的决策信息
        friendly_situations = threat_model.get_friendly_situations()
        for i in range(len(actions_list)):
            decision = {
                'aircraft_id': i,
                'situation': friendly_situations[i] if i < len(friendly_situations) else 'unknown',
                'action': actions_list[i],
                'explanation': explanations_list[i]
            }
            summary['individual_decisions'].append(decision)
            
        return summary

    def update_target_locks(self, aircraft_id, target_id):
        """
        更新目标锁定记录
        :param aircraft_id: 友机ID
        :param target_id: 目标ID
        """
        if target_id not in self.shared_target_locks:
            self.shared_target_locks[target_id] = []
        if aircraft_id not in self.shared_target_locks[target_id]:
            self.shared_target_locks[target_id].append(aircraft_id)

    def get_target_lock_count(self, target_id):
        """
        获取目标被锁定的数量
        :param target_id: 目标ID
        :return: 锁定数量
        """
        return len(self.shared_target_locks.get(target_id, []))

    def get_available_support_targets(self, current_aircraft_id):
        """
        获取可用的支援目标（需要支援的友机）
        :param current_aircraft_id: 当前友机ID
        :return: [(友机ID, 支援优先级), ...]
        """
        support_candidates = []
        state = self.context.get('state', {})
        friendly_situations = self.context.get('friendly_situations', [])
        situation_details = self.context.get('situation_details', [])
        
        # 从配置获取支援选择参数
        behavior_config = self.config.get('behavior_tree', {})
        support_config = behavior_config.get('support_selection', {})
        weights = support_config.get('priority_weights', {})
        disadvantage_factor = weights.get('disadvantage_factor', 3.0)
        neutral_factor = weights.get('neutral_factor', 1.0)
        threat_factor = weights.get('threat_factor', 1.0)
        distance_factor = weights.get('distance_factor', 1.0)
        min_support_priority = support_config.get('min_support_priority', 0.5)
        distance_unit = support_config.get('distance_unit', 100.0)
        
        our_aircrafts = state.get('our_aircrafts', [])
        
        for aircraft_id, aircraft in enumerate(our_aircrafts):
            if aircraft_id == current_aircraft_id:
                continue  # 不支援自己
                
            # 计算支援优先级
            priority = 0.0
            
            # 态势因子：劣势态势优先级更高
            if aircraft_id < len(friendly_situations):
                situation = friendly_situations[aircraft_id]
                if situation == "disadvantage":
                    priority += disadvantage_factor
                elif situation == "neutral":
                    priority += neutral_factor
            
            # 威胁因子：受威胁程度
            if aircraft_id < len(situation_details):
                detail = situation_details[aircraft_id]
                threat_value = detail.get('max_threat_value', 0.0)
                priority += threat_value * threat_factor
            
            # 距离因子：距离越近支援效果越好
            current_pos = our_aircrafts[current_aircraft_id]['position']
            target_pos = aircraft['position']
            distance = np.linalg.norm(target_pos - current_pos)
            if distance > 0:
                priority += distance_factor / (distance / distance_unit)
            
            if priority > min_support_priority:  # 只考虑优先级较高的支援目标
                support_candidates.append((aircraft_id, priority))
        
        # 按优先级排序
        support_candidates.sort(key=lambda x: x[1], reverse=True)
        return support_candidates


# ===== 便捷函数 =====

def create_and_execute_behavior_tree(state, threat_model):
    """
    便捷函数：创建并执行行为树
    :param state: 当前环境状态
    :param threat_model: 威胁评估模型
    :return: (actions_list, explanations_list, summary)
    """
    engine = BehaviorTreeEngine()
    actions_list, explanations_list = engine.get_all_actions(state, threat_model)
    summary = engine.get_decision_summary(state, threat_model)
    
    return actions_list, explanations_list, summary


def get_behavior_tree_decision_for_aircraft(aircraft_id, state, threat_model):
    """
    便捷函数：获取指定友机的行为树决策
    :param aircraft_id: 友机ID
    :param state: 当前环境状态
    :param threat_model: 威胁评估模型
    :return: (action, explanation)
    """
    engine = BehaviorTreeEngine()
    return engine.get_actions_for_friendly(aircraft_id, state, threat_model)


# ===== 示例使用 =====
if __name__ == "__main__":
    # 这里可以添加一些测试代码
    print("行为树引擎模块已加载")
