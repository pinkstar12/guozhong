import yaml
import numpy as np
from common.registry import build
from .behavior_tree import BehaviorTreeEngine


# 默认决策标志字典
DEFAULT_DECISION_FLAGS = {
    'use_defense_decision': 0,
    'use_attack_decision': 0,
    'use_support_decision': 0,
    'use_radar_decision': 1
}

# 攻击系统决策标志（攻击和支援决策置为1）
ATTACK_DECISION_FLAGS = {
    'use_defense_decision': 0,
    'use_attack_decision': 1,
    'use_support_decision': 1,
    'use_radar_decision': 1
}

class ExpertSystem:
    def __init__(self, threat_model, optimization_module, selection=None, mode='optimization', behavior_tree=None, **kwargs):
        """
        :param threat_model: 已实例化的 threat_model 对象（由 registry/build 递归注入）
        :param optimization_module: 已实例化的优化模块对象（同上）
        :param selection: 配置中 selection 字段的内容（如有）
        :param mode: 运行模式，'optimization' 或 'behavior_tree'
        :param behavior_tree: 行为树配置参数
        :param kwargs: 兜底其余配置，防止报错
        """
        print("init optimization_module:", optimization_module, type(optimization_module))
        self.threat_model = threat_model
        self.optimization_module = optimization_module
        self.selection_config = selection or {}
        self.selection_strategy = self.selection_config.get('strategy', 'utopia')
        
        # 新增：模式配置
        self.mode = mode
        
        # 保存完整配置以便传递给行为树引擎
        self.config = {
            'behavior_tree': behavior_tree or {},
            'selection': selection or {},
            'mode': mode,
            **kwargs
        }
        
        # 初始化行为树引擎，传递配置参数
        self.behavior_tree_engine = BehaviorTreeEngine(config=self.config)
        
        print(f"ExpertSystem initialized in {self.mode} mode")


    def Process(self, state):
        """
        处理状态并生成动作
        :param state: 当前环境状态
        :return: 优化后的动作 或 行为树决策动作
        """
        # 新增：根据模式选择处理方式
        if self.mode == 'behavior_tree':
            return self._process_behavior_tree(state)
        elif self.mode == 'optimization':
            return self._process_optimization(state)
        else:
            raise ValueError(f"未知的运行模式: {self.mode}")
    
    def _process_behavior_tree(self, state):
        """
        使用行为树模式处理状态
        :param state: 当前环境状态
        :return: 行为树决策的动作字典
        """
        actions_list, explanations_list = self.behavior_tree_engine.get_all_actions(state, self.threat_model)
        
        # 为每个动作添加攻击决策标志
        enhanced_actions = []
        for action in actions_list:
            if isinstance(action, dict):
                action.update(ATTACK_DECISION_FLAGS.copy())
            else:
                # 如果动作不是字典格式，创建新的字典格式动作
                enhanced_action = {
                    'original_action': action,
                    **ATTACK_DECISION_FLAGS.copy()
                }
                enhanced_actions.append(enhanced_action)
                continue
            enhanced_actions.append(action)
        
        # 存储决策解释供后续查询
        self.last_explanations = explanations_list
        self.last_decision_summary = self.behavior_tree_engine.get_decision_summary(state, self.threat_model)
        
        return {
            "actions": enhanced_actions,
            "explanations": explanations_list
        }
        
    def _process_optimization(self, state):
        """
        使用优化模式处理状态
        :param state: 当前环境状态
        :return: 优化算法的动作
        """
        # 获取帕累托最优解集
        pareto_set = self.optimization_module.process_state(state)

        # 生成优化解释
        if hasattr(self.optimization_module, 'generate_explanation'):
            self.last_optimization_explanation = self.optimization_module.generate_explanation(
                pareto_set, state
            )
        else:
            self.last_optimization_explanation = {
                'text_explanation': '优化模块不支持解释生成',
                'optimization_type': 'Unknown'
            }

        # 态势感知策略选择
        if self.selection_strategy == 'dynamic' and hasattr(self.threat_model, 'evaluate_situation'):
            # 获取当前态势评估
            advantage, risk = self.threat_model.process_state(state)
            situation = self.threat_model.evaluate_situation(advantage, risk)

            # 根据态势选择策略
            if situation == "advantage":
                # 优势时采用保守策略（最小风险）
                actions = self._select_conservative(pareto_set)
            elif situation == "disadvantage":
                # 劣势时采用激进策略（最大优势）
                actions = self._select_aggressive(pareto_set)
            else:
                # 均势时使用平衡策略
                actions = self._select_balanced(pareto_set)
        else:
            # 非动态策略选择
            if self.selection_strategy == 'utopia':
                actions = self._select_by_utopia(pareto_set)
            elif self.selection_strategy == 'first':
                # 默认返回第一个解
                actions = pareto_set[0][0]
            elif self.selection_strategy == 'min_risk':
                actions = self._select_conservative(pareto_set)
            elif self.selection_strategy == 'max_advantage':
                actions = self._select_aggressive(pareto_set)
            else:
                raise ValueError(f"未知的选择策略: {self.selection_strategy}")
        
        # 为优化生成的动作添加攻击决策标志
        enhanced_actions = self._add_decision_flags_to_actions(actions)
        
        # 返回统一的字典格式
        return {"actions": enhanced_actions}

    def _select_by_utopia(self, pareto_solutions):
        """
        使用理想点法从帕累托前沿选择最优解
        :param pareto_solutions: 帕累托解集 [(actions, objectives), ...]
        :return: 选中的动作列表
        """
        # 提取目标函数值
        objectives = np.array([obj for _, obj in pareto_solutions])

        # 计算理想点（每个目标的最小值）
        utopia_point = np.min(objectives, axis=0)

        # 归一化目标值（0-1范围）
        min_vals = np.min(objectives, axis=0)
        max_vals = np.max(objectives, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals < 1e-5] = 1  # 避免除以零

        norm_objectives = (objectives - min_vals) / range_vals

        # 计算到理想点的欧氏距离
        norm_utopia = (utopia_point - min_vals) / range_vals
        distances = np.linalg.norm(norm_objectives - norm_utopia, axis=1)

        # 选择距离最小的解
        best_idx = np.argmin(distances)
        return pareto_solutions[best_idx][0]

    def _select_conservative(self, pareto_solutions):
        """选择最小风险解"""
        objectives = np.array([obj for _, obj in pareto_solutions])
        min_risk_idx = np.argmin(objectives[:, 1])  # 第二个目标是风险
        return pareto_solutions[min_risk_idx][0]

    def _select_aggressive(self, pareto_solutions):
        """选择最大优势解（注意：第一个目标是-advantage，所以最小化第一个目标就是最大化advantage）"""
        objectives = np.array([obj for _, obj in pareto_solutions])
        max_advantage_idx = np.argmin(objectives[:, 0])  # 第一个目标是-advantage
        return pareto_solutions[max_advantage_idx][0]

    def _select_balanced(self, pareto_solutions):
        """平衡选择策略（使用加权平均）"""
        objectives = np.array([obj for _, obj in pareto_solutions])

        # 归一化目标值
        min_vals = np.min(objectives, axis=0)
        max_vals = np.max(objectives, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals < 1e-5] = 1

        norm_objectives = (objectives - min_vals) / range_vals

        # 计算加权分数（优势权重60%，风险权重40%）
        weights = np.array([0.6, 0.4])
        scores = np.dot(norm_objectives, weights)

        # 选择加权分数最高的解
        best_idx = np.argmin(scores)  # 注意：目标值都是最小化，所以分数越低越好
        return pareto_solutions[best_idx][0]
    
    def _add_decision_flags_to_actions(self, actions):
        """
        为动作添加攻击决策标志
        :param actions: 动作列表或单个动作
        :return: 包含决策标志的动作列表
        """
        if isinstance(actions, list):
            enhanced_actions = []
            for action in actions:
                if isinstance(action, dict):
                    action.update(ATTACK_DECISION_FLAGS.copy())
                    enhanced_actions.append(action)
                else:
                    # 如果动作不是字典格式，创建新的字典格式动作
                    enhanced_action = {
                        'original_action': action,
                        **ATTACK_DECISION_FLAGS.copy()
                    }
                    enhanced_actions.append(enhanced_action)
            return enhanced_actions
        else:
            # 单个动作
            if isinstance(actions, dict):
                actions.update(ATTACK_DECISION_FLAGS.copy())
                return actions
            else:
                # 如果动作不是字典格式，创建新的字典格式动作
                return {
                    'original_action': actions,
                    **ATTACK_DECISION_FLAGS.copy()
                }
    
    # ===== 新增功能：模式切换和便捷输出接口 =====
    
    def set_mode(self, mode):
        """
        切换运行模式
        :param mode: 'optimization' 或 'behavior_tree'
        """
        if mode not in ['optimization', 'behavior_tree']:
            raise ValueError(f"无效的模式: {mode}，支持的模式: ['optimization', 'behavior_tree']")
        
        old_mode = self.mode
        self.mode = mode
        print(f"模式已从 {old_mode} 切换到 {mode}")
    
    def get_current_mode(self):
        """
        获取当前运行模式
        :return: 当前模式字符串
        """
        return self.mode
    
    def get_all_friendly_actions(self, state):
        """
        获取所有友方飞机的下一步动作
        :param state: 当前环境状态
        :return: 根据模式返回不同格式的结果
        """
        if self.mode == 'behavior_tree':
            actions_list, explanations_list = self.behavior_tree_engine.get_all_actions(state, self.threat_model)
            return {
                'actions': actions_list,
                'explanations': explanations_list,
                'mode': 'behavior_tree'
            }
        elif self.mode == 'optimization':
            actions = self._process_optimization(state)
            return {
                'actions': actions,
                'explanations': ['优化算法决策'] * len(actions) if isinstance(actions, list) else ['优化算法决策'],
                'mode': 'optimization'
            }
        else:
            raise ValueError(f"未知的运行模式: {self.mode}")
    
    def get_friendly_action(self, aircraft_id, state):
        """
        获取指定友方飞机的下一步动作
        :param aircraft_id: 友方飞机ID
        :param state: 当前环境状态
        :return: 指定飞机的动作和解释
        """
        if self.mode == 'behavior_tree':
            action, explanation = self.behavior_tree_engine.get_actions_for_friendly(aircraft_id, state, self.threat_model)
            return {
                'aircraft_id': aircraft_id,
                'action': action,
                'explanation': explanation,
                'mode': 'behavior_tree'
            }
        elif self.mode == 'optimization':
            all_actions = self._process_optimization(state)
            if isinstance(all_actions, list) and aircraft_id < len(all_actions):
                return {
                    'aircraft_id': aircraft_id,
                    'action': all_actions[aircraft_id],
                    'explanation': '优化算法决策',
                    'mode': 'optimization'
                }
            else:
                return {
                    'aircraft_id': aircraft_id,
                    'action': np.zeros(8),
                    'explanation': f'友机{aircraft_id}不存在或优化结果格式不匹配',
                    'mode': 'optimization'
                }
        else:
            raise ValueError(f"未知的运行模式: {self.mode}")
    
    def get_decision_explanations(self, state):
        """
        获取决策解释（仅在行为树模式下有效）
        :param state: 当前环境状态
        :return: 决策解释列表
        """
        if self.mode == 'behavior_tree':
            _, explanations_list = self.behavior_tree_engine.get_all_actions(state, self.threat_model)
            return explanations_list
        else:
            return ['优化模式不提供详细决策解释']
    
    def get_decision_summary(self, state):
        """
        获取决策总结
        :param state: 当前环境状态
        :return: 决策总结字典
        """
        if self.mode == 'behavior_tree':
            return self.behavior_tree_engine.get_decision_summary(state, self.threat_model)
        elif self.mode == 'optimization':
            # 为优化模式提供基本的决策总结
            actions = self._process_optimization(state)
            
            # 获取态势信息
            team_advantage, team_risk = self.threat_model.process_state(state)
            team_situation = self.threat_model.evaluate_situation(team_advantage, team_risk)
            
            return {
                'mode': 'optimization',
                'team_situation': {
                    'status': team_situation,
                    'advantage': team_advantage,
                    'risk': team_risk
                },
                'selection_strategy': self.selection_strategy,
                'actions': actions.tolist() if hasattr(actions, 'tolist') else actions,
                'total_aircrafts': len(state.get('our_aircrafts', []))
            }
        else:
            raise ValueError(f"未知的运行模式: {self.mode}")
    
    def print_behavior_tree_structure(self):
        """
        打印行为树结构（仅在行为树模式下有效）
        """
        if self.mode == 'behavior_tree':
            print("=== 行为树结构 ===")
            self.behavior_tree_engine.print_tree_structure()
        else:
            print("当前模式不是行为树模式，无法打印行为树结构")
    
    def get_mode_info(self):
        """
        获取模式信息
        :return: 模式详细信息字典
        """
        info = {
            'current_mode': self.mode,
            'available_modes': ['optimization', 'behavior_tree'],
            'threat_model': str(type(self.threat_model).__name__),
            'optimization_module': str(type(self.optimization_module).__name__) if self.optimization_module else None,
            'selection_strategy': self.selection_strategy
        }
        
        if self.mode == 'behavior_tree':
            info['behavior_tree_engine'] = str(type(self.behavior_tree_engine).__name__)
        
        return info
    
    def get_optimization_explanation(self):
        """
        获取最近一次优化的解释
        :return: 优化解释字典
        """
        if self.mode == 'optimization':
            return getattr(self, 'last_optimization_explanation', {
                'text_explanation': '暂无优化解释',
                'optimization_type': 'Unknown'
            })
        else:
            return {
                'text_explanation': '当前模式不是优化模式',
                'optimization_type': 'N/A'
            }
    
    def get_comprehensive_explanation(self, state):
        """
        获取综合解释（包含决策过程和优化结果）
        :param state: 当前环境状态
        :return: 综合解释字典
        """
        if self.mode == 'optimization':
            # 执行优化以获取最新解释
            actions = self._process_optimization(state)
            
            # 获取态势信息
            team_advantage, team_risk = self.threat_model.process_state(state)
            team_situation = self.threat_model.evaluate_situation(team_advantage, team_risk)
            
            # 获取优化解释
            optimization_explanation = self.get_optimization_explanation()
            
            return {
                'mode': 'optimization',
                'decision_summary': {
                    'team_situation': {
                        'status': team_situation,
                        'advantage': team_advantage,
                        'risk': team_risk
                    },
                    'selection_strategy': self.selection_strategy,
                    'selected_actions': actions.tolist() if hasattr(actions, 'tolist') else actions
                },
                'optimization_explanation': optimization_explanation,
                'comprehensive_text': f"{optimization_explanation.get('text_explanation', '')} "
                                   f"基于当前{team_situation}态势，采用{self.selection_strategy}策略进行最终决策选择。"
            }
        elif self.mode == 'behavior_tree':
            # 行为树模式的综合解释
            _, explanations_list = self.behavior_tree_engine.get_all_actions(state, self.threat_model)
            decision_summary = self.behavior_tree_engine.get_decision_summary(state, self.threat_model)
            
            return {
                'mode': 'behavior_tree',
                'decision_summary': decision_summary,
                'detailed_explanations': explanations_list,
                'comprehensive_text': f"行为树决策系统基于当前态势执行了{len(explanations_list)}个决策节点，"
                                   f"为{decision_summary.get('total_aircrafts', 0)}架友方飞机生成了相应的战术动作。"
            }
        else:
            return {
                'mode': self.mode,
                'error': f'未知的运行模式: {self.mode}'
            }
