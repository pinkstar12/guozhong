"""
优化结果解释生成器模块
为NSGA-II和DE优化算法提供人类可读的优化结果解释
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod

class BaseExplanationGenerator(ABC):
    """解释生成器基类"""
    
    def __init__(self, optimization_params: Dict = None):
        """
        初始化解释生成器
        
        参数:
            optimization_params: 优化参数配置
        """
        self.optimization_params = optimization_params or {}
        self.parameter_names = [
            'speed', 'height', 'heading', 'position_x', 'position_y'
        ]
    
    @abstractmethod
    def generate_explanation(self, optimization_result: Any, initial_state: Dict = None) -> Dict[str, Any]:
        """
        生成优化结果解释
        
        参数:
            optimization_result: 优化结果
            initial_state: 初始状态（可选）
        
        返回:
            解释字典
        """
        pass
    
    def _analyze_parameter_changes(self, initial_params: List[float], 
                                 optimized_params: List[float]) -> Dict[str, Any]:
        """
        分析参数变化
        
        参数:
            initial_params: 初始参数
            optimized_params: 优化后参数
        
        返回:
            参数变化分析结果
        """
        changes = {}
        
        for i, param_name in enumerate(self.parameter_names[:len(initial_params)]):
            if i < len(optimized_params):
                initial_val = initial_params[i]
                optimized_val = optimized_params[i]
                change = optimized_val - initial_val
                change_percent = (change / initial_val * 100) if initial_val != 0 else 0
                
                changes[param_name] = {
                    'initial': initial_val,
                    'optimized': optimized_val,
                    'change': change,
                    'change_percent': change_percent,
                    'significant': abs(change_percent) > 5.0  # 5%以上变化视为显著
                }
        
        return changes
    
    def _categorize_parameter_importance(self, changes: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        根据变化幅度对参数重要性进行分类
        
        参数:
            changes: 参数变化分析结果
        
        返回:
            参数重要性分类
        """
        high_impact = []  # 变化>20%
        medium_impact = []  # 变化5-20%
        low_impact = []  # 变化<5%
        
        for param_name, change_info in changes.items():
            change_percent = abs(change_info['change_percent'])
            
            if change_percent > 20:
                high_impact.append(param_name)
            elif change_percent > 5:
                medium_impact.append(param_name)
            else:
                low_impact.append(param_name)
        
        return {
            'high_impact': high_impact,
            'medium_impact': medium_impact,
            'low_impact': low_impact
        }
    
    def _generate_constraint_analysis(self, bounds: List[Tuple[float, float]], 
                                    optimized_params: List[float]) -> Dict[str, Any]:
        """
        生成约束分析
        
        参数:
            bounds: 参数边界列表
            optimized_params: 优化后参数
        
        返回:
            约束分析结果
        """
        constraint_analysis = {
            'boundary_violations': [],
            'near_boundaries': [],
            'constraint_utilization': []
        }
        
        for i, (param_val, (lower, upper)) in enumerate(zip(optimized_params, bounds)):
            param_name = self.parameter_names[i] if i < len(self.parameter_names) else f'param_{i}'
            
            # 检查边界违反
            if param_val < lower or param_val > upper:
                constraint_analysis['boundary_violations'].append({
                    'parameter': param_name,
                    'value': param_val,
                    'bounds': (lower, upper)
                })
            
            # 检查接近边界（在边界5%范围内）
            range_val = upper - lower
            if range_val > 0:
                lower_threshold = lower + 0.05 * range_val
                upper_threshold = upper - 0.05 * range_val
                
                if param_val <= lower_threshold or param_val >= upper_threshold:
                    constraint_analysis['near_boundaries'].append({
                        'parameter': param_name,
                        'value': param_val,
                        'bounds': (lower, upper),
                        'utilization': min(
                            (param_val - lower) / range_val,
                            (upper - param_val) / range_val
                        )
                    })
                
                # 约束利用率
                utilization = (param_val - lower) / range_val
                constraint_analysis['constraint_utilization'].append({
                    'parameter': param_name,
                    'utilization': utilization
                })
        
        return constraint_analysis


class NSGAExplanationGenerator(BaseExplanationGenerator):
    """NSGA-II多目标优化解释生成器"""
    
    def generate_explanation(self, optimization_result: List[Tuple], 
                           initial_state: Dict = None) -> Dict[str, Any]:
        """
        生成NSGA-II优化结果解释
        
        参数:
            optimization_result: [(actions_list, objectives_list), ...] 格式的Pareto解集
            initial_state: 初始状态
        
        返回:
            包含解释信息的字典
        """
        if not optimization_result:
            return {'error': '优化结果为空'}
        
        # 提取目标函数值和动作
        all_objectives = np.array([obj for _, obj in optimization_result])
        all_actions = [actions for actions, _ in optimization_result]
        
        # Pareto前沿分析
        pareto_analysis = self._analyze_pareto_front(all_objectives)
        
        # 解集多样性分析
        diversity_analysis = self._analyze_solution_diversity(all_actions, all_objectives)
        
        # 目标冲突分析
        conflict_analysis = self._analyze_objective_conflicts(all_objectives)
        
        # 推荐解分析
        recommended_solutions = self._recommend_solutions(optimization_result)
        
        # 生成文本解释
        text_explanation = self._generate_nsga_text_explanation(
            pareto_analysis, diversity_analysis, conflict_analysis, recommended_solutions
        )
        
        explanation = {
            'optimization_type': 'NSGA-II Multi-objective',
            'total_solutions': len(optimization_result),
            'pareto_analysis': pareto_analysis,
            'diversity_analysis': diversity_analysis,
            'conflict_analysis': conflict_analysis,
            'recommended_solutions': recommended_solutions,
            'text_explanation': text_explanation,
            'objectives_description': {
                'objective_1': 'Team Advantage (负值，越小越好)',
                'objective_2': 'Team Risk (正值，越小越好)'
            }
        }
        
        return explanation
    
    def _analyze_pareto_front(self, objectives: np.ndarray) -> Dict[str, Any]:
        """分析Pareto前沿特征"""
        if len(objectives) == 0:
            return {}
        
        analysis = {
            'front_size': len(objectives),
            'objective_ranges': {},
            'extreme_points': {},
            'center_point': {}
        }
        
        # 目标函数范围
        for i in range(objectives.shape[1]):
            obj_values = objectives[:, i]
            analysis['objective_ranges'][f'objective_{i+1}'] = {
                'min': float(np.min(obj_values)),
                'max': float(np.max(obj_values)),
                'range': float(np.max(obj_values) - np.min(obj_values)),
                'mean': float(np.mean(obj_values)),
                'std': float(np.std(obj_values))
            }
        
        # 极值点
        analysis['extreme_points'] = {
            'min_advantage_solution': int(np.argmin(objectives[:, 0])),  # 最大优势（最小-advantage）
            'min_risk_solution': int(np.argmin(objectives[:, 1])),      # 最小风险
            'balanced_solution': int(np.argmin(np.sum(objectives, axis=1)))  # 平衡解
        }
        
        # 中心点
        analysis['center_point'] = {
            'advantage': float(np.mean(objectives[:, 0])),
            'risk': float(np.mean(objectives[:, 1]))
        }
        
        return analysis
    
    def _analyze_solution_diversity(self, actions_list: List, objectives: np.ndarray) -> Dict[str, Any]:
        """分析解的多样性"""
        if len(actions_list) <= 1:
            return {'diversity_score': 0.0, 'analysis': '解集过小，无法分析多样性'}
        
        # 计算目标空间的多样性
        obj_distances = []
        for i in range(len(objectives)):
            for j in range(i+1, len(objectives)):
                distance = np.linalg.norm(objectives[i] - objectives[j])
                obj_distances.append(distance)
        
        diversity_score = np.mean(obj_distances) if obj_distances else 0.0
        
        return {
            'diversity_score': float(diversity_score),
            'min_distance': float(np.min(obj_distances)) if obj_distances else 0.0,
            'max_distance': float(np.max(obj_distances)) if obj_distances else 0.0,
            'analysis': f'解集多样性得分: {diversity_score:.4f}'
        }
    
    def _analyze_objective_conflicts(self, objectives: np.ndarray) -> Dict[str, Any]:
        """分析目标冲突程度"""
        if len(objectives) <= 1:
            return {'conflict_analysis': '解集过小，无法分析目标冲突'}
        
        # 计算目标间相关性
        correlation = np.corrcoef(objectives.T)
        
        conflict_analysis = {
            'objective_correlation': float(correlation[0, 1]) if correlation.shape == (2, 2) else 0.0,
            'conflict_level': 'unknown'
        }
        
        # 判断冲突程度
        corr_val = conflict_analysis['objective_correlation']
        if corr_val > 0.7:
            conflict_analysis['conflict_level'] = 'low'  # 目标一致性高
            conflict_analysis['description'] = '目标间冲突较小，优势和风险趋势相似'
        elif corr_val > 0.3:
            conflict_analysis['conflict_level'] = 'medium'
            conflict_analysis['description'] = '目标间存在中等程度冲突'
        elif corr_val > -0.3:
            conflict_analysis['conflict_level'] = 'medium'
            conflict_analysis['description'] = '目标间冲突程度适中'
        else:
            conflict_analysis['conflict_level'] = 'high'  # 高度冲突
            conflict_analysis['description'] = '目标间存在显著冲突，需要权衡优势和风险'
        
        return conflict_analysis
    
    def _recommend_solutions(self, optimization_result: List[Tuple]) -> Dict[str, Any]:
        """推荐解决方案"""
        if not optimization_result:
            return {}
        
        objectives = np.array([obj for _, obj in optimization_result])
        
        recommendations = {}
        
        # 1. 最大优势解（攻击性策略）
        max_advantage_idx = np.argmin(objectives[:, 0])
        recommendations['aggressive'] = {
            'index': int(max_advantage_idx),
            'objectives': objectives[max_advantage_idx].tolist(),
            'description': '最大化团队优势的攻击性策略',
            'suitable_for': '优势局面或需要快速决战'
        }
        
        # 2. 最小风险解（保守策略）
        min_risk_idx = np.argmin(objectives[:, 1])
        recommendations['conservative'] = {
            'index': int(min_risk_idx),
            'objectives': objectives[min_risk_idx].tolist(),
            'description': '最小化团队风险的保守策略',
            'suitable_for': '劣势局面或需要稳妥推进'
        }
        
        # 3. 平衡解（中庸策略）
        # 使用归一化后的欧氏距离到理想点
        normalized_obj = (objectives - np.min(objectives, axis=0)) / (
            np.max(objectives, axis=0) - np.min(objectives, axis=0) + 1e-8
        )
        ideal_point = np.zeros(2)  # 理想点为(0,0)
        distances = np.linalg.norm(normalized_obj - ideal_point, axis=1)
        balanced_idx = np.argmin(distances)
        
        recommendations['balanced'] = {
            'index': int(balanced_idx),
            'objectives': objectives[balanced_idx].tolist(),
            'description': '平衡优势和风险的中庸策略',
            'suitable_for': '均势局面或不确定情况'
        }
        
        return recommendations
    
    def _generate_nsga_text_explanation(self, pareto_analysis: Dict, diversity_analysis: Dict,
                                       conflict_analysis: Dict, recommended_solutions: Dict) -> str:
        """生成NSGA-II优化的文本解释"""
        explanation_parts = []
        
        # 基本信息
        front_size = pareto_analysis.get('front_size', 0)
        explanation_parts.append(
            f"NSGA-II多目标优化生成了{front_size}个Pareto最优解，"
            f"同时优化团队优势（最大化）和团队风险（最小化）两个目标。"
        )
        
        # 目标冲突分析
        conflict_desc = conflict_analysis.get('description', '')
        if conflict_desc:
            explanation_parts.append(conflict_desc)
        
        # 解集多样性
        diversity_score = diversity_analysis.get('diversity_score', 0)
        if diversity_score > 0:
            explanation_parts.append(
                f"解集多样性得分为{diversity_score:.3f}，"
                f"{'提供了丰富的战术选择' if diversity_score > 1.0 else '解的分布相对集中'}。"
            )
        
        # 推荐策略
        if recommended_solutions:
            explanation_parts.append("基于当前态势，推荐以下策略选择：")
            
            for strategy_name, strategy_info in recommended_solutions.items():
                strategy_desc = strategy_info.get('description', '')
                suitable_for = strategy_info.get('suitable_for', '')
                explanation_parts.append(f"• {strategy_desc}，适用于{suitable_for}")
        
        return ' '.join(explanation_parts)


class DEExplanationGenerator(BaseExplanationGenerator):
    """DE单目标优化解释生成器"""
    
    def generate_explanation(self, optimization_result: Dict, 
                           initial_state: Dict = None) -> Dict[str, Any]:
        """
        生成DE优化结果解释
        
        参数:
            optimization_result: DE优化结果字典
            initial_state: 初始状态
        
        返回:
            包含解释信息的字典
        """
        if not optimization_result.get('success', False):
            return {
                'error': '优化失败',
                'message': optimization_result.get('message', '未知错误')
            }
        
        # 提取优化结果
        best_solution = optimization_result.get('best_solution', [])
        best_fitness = optimization_result.get('best_fitness', float('inf'))
        history = optimization_result.get('history', {})
        
        # 收敛性分析
        convergence_analysis = self._analyze_convergence(history)
        
        # 参数优化分析
        parameter_analysis = self._analyze_parameter_optimization(
            best_solution, initial_state
        )
        
        # 优化效果评估
        effectiveness_analysis = self._analyze_optimization_effectiveness(
            best_fitness, history, initial_state
        )
        
        # 生成文本解释
        text_explanation = self._generate_de_text_explanation(
            convergence_analysis, parameter_analysis, effectiveness_analysis
        )
        
        explanation = {
            'optimization_type': 'Differential Evolution',
            'success': True,
            'best_fitness': float(best_fitness),
            'convergence_analysis': convergence_analysis,
            'parameter_analysis': parameter_analysis,
            'effectiveness_analysis': effectiveness_analysis,
            'text_explanation': text_explanation,
            'optimization_config': self.optimization_params
        }
        
        return explanation
    
    def _analyze_convergence(self, history: Dict) -> Dict[str, Any]:
        """分析收敛性"""
        if not history or 'best_fitness' not in history:
            return {'analysis': '无收敛历史数据'}
        
        best_fitness_history = history['best_fitness']
        
        analysis = {
            'total_generations': len(best_fitness_history),
            'convergence_generation': history.get('convergence_iter', None),
            'final_fitness': float(best_fitness_history[-1]) if best_fitness_history else 0.0,
            'improvement_rate': 0.0,
            'convergence_speed': 'unknown'
        }
        
        if len(best_fitness_history) > 1:
            initial_fitness = best_fitness_history[0]
            final_fitness = best_fitness_history[-1]
            
            # 改进率
            if initial_fitness != 0:
                analysis['improvement_rate'] = float(
                    (initial_fitness - final_fitness) / initial_fitness * 100
                )
            
            # 收敛速度分析
            if analysis['convergence_generation']:
                conv_gen = analysis['convergence_generation']
                total_gen = analysis['total_generations']
                
                if conv_gen < total_gen * 0.3:
                    analysis['convergence_speed'] = 'fast'
                elif conv_gen < total_gen * 0.7:
                    analysis['convergence_speed'] = 'medium'
                else:
                    analysis['convergence_speed'] = 'slow'
            
            # 收敛趋势
            if len(best_fitness_history) >= 10:
                recent_improvement = (
                    best_fitness_history[-10] - best_fitness_history[-1]
                ) / best_fitness_history[-10] if best_fitness_history[-10] != 0 else 0
                
                analysis['recent_improvement'] = float(recent_improvement * 100)
        
        return analysis
    
    def _analyze_parameter_optimization(self, best_solution: List[float], 
                                      initial_state: Dict = None) -> Dict[str, Any]:
        """分析参数优化结果"""
        if not best_solution:
            return {'analysis': '无最优解数据'}
        
        analysis = {
            'optimized_parameters': {},
            'key_adjustments': [],
            'parameter_summary': ''
        }
        
        # 参数值分析
        param_names = ['position_x', 'position_z', 'height', 'speed', 'heading']
        for i, value in enumerate(best_solution):
            param_name = param_names[i] if i < len(param_names) else f'param_{i}'
            # 处理不同类型的值
            try:
                if isinstance(value, (list, tuple)):
                    # 处理列表或元组，递归转换每个元素
                    analysis['optimized_parameters'][param_name] = [
                        float(v) if not isinstance(v, (dict, list, tuple)) else str(v) 
                        for v in value
                    ]
                elif isinstance(value, dict):
                    # 处理字典类型，转换为字符串表示
                    analysis['optimized_parameters'][param_name] = str(value)
                elif isinstance(value, (int, float)):
                    # 处理数值类型
                    analysis['optimized_parameters'][param_name] = float(value)
                else:
                    # 其他类型转换为字符串
                    analysis['optimized_parameters'][param_name] = str(value)
            except (ValueError, TypeError) as e:
                # 如果转换失败，使用字符串表示
                analysis['optimized_parameters'][param_name] = str(value)
        
        # 如果有初始状态，计算变化
        if initial_state and 'aircraft' in initial_state:
            aircraft = initial_state['aircraft']
            initial_params = [
                aircraft['position'][0], aircraft['position'][1],
                aircraft['height'], aircraft['speed'],
                aircraft.get('heading', 0)
            ]
            
            changes = self._analyze_parameter_changes(initial_params, best_solution)
            analysis['parameter_changes'] = changes
            
            # 识别关键调整
            importance = self._categorize_parameter_importance(changes)
            analysis['parameter_importance'] = importance
            
            # 生成关键调整描述
            key_adjustments = []
            for param in importance['high_impact']:
                change_info = changes[param]
                key_adjustments.append(f"{param}: {change_info['change']:+.2f} ({change_info['change_percent']:+.1f}%)")
            
            analysis['key_adjustments'] = key_adjustments
        
        return analysis
    
    def _analyze_optimization_effectiveness(self, best_fitness: float, 
                                          history: Dict, initial_state: Dict = None) -> Dict[str, Any]:
        """分析优化效果"""
        analysis = {
            'final_threat_level': float(best_fitness),
            'threat_reduction': 0.0,
            'effectiveness_rating': 'unknown'
        }
        
        # 威胁等级评估
        if best_fitness < 0.2:
            analysis['threat_level_description'] = '威胁很低'
            analysis['effectiveness_rating'] = 'excellent'
        elif best_fitness < 0.4:
            analysis['threat_level_description'] = '威胁较低'
            analysis['effectiveness_rating'] = 'good'
        elif best_fitness < 0.6:
            analysis['threat_level_description'] = '威胁中等'
            analysis['effectiveness_rating'] = 'moderate'
        elif best_fitness < 0.8:
            analysis['threat_level_description'] = '威胁较高'
            analysis['effectiveness_rating'] = 'limited'
        else:
            analysis['threat_level_description'] = '威胁很高'
            analysis['effectiveness_rating'] = 'poor'
        
        # 如果有历史记录，计算威胁降低幅度
        if history and 'best_fitness' in history and len(history['best_fitness']) > 0:
            initial_fitness = history['best_fitness'][0]
            if initial_fitness > 0:
                analysis['threat_reduction'] = float(
                    (initial_fitness - best_fitness) / initial_fitness * 100
                )
        
        return analysis
    
    def _generate_de_text_explanation(self, convergence_analysis: Dict, 
                                    parameter_analysis: Dict, 
                                    effectiveness_analysis: Dict) -> str:
        """生成DE优化的文本解释"""
        explanation_parts = []
        
        # 基本优化结果
        final_threat = effectiveness_analysis.get('final_threat_level', 0)
        threat_desc = effectiveness_analysis.get('threat_level_description', '未知')
        explanation_parts.append(
            f"差分进化算法优化完成，最终威胁度为{final_threat:.4f}（{threat_desc}）。"
        )
        
        # 威胁降低效果
        threat_reduction = effectiveness_analysis.get('threat_reduction', 0)
        if threat_reduction > 0:
            explanation_parts.append(f"相比初始状态，威胁度降低了{threat_reduction:.1f}%。")
        
        # 收敛性描述
        total_gen = convergence_analysis.get('total_generations', 0)
        conv_speed = convergence_analysis.get('convergence_speed', 'unknown')
        if total_gen > 0:
            speed_desc = {'fast': '快速', 'medium': '中等速度', 'slow': '缓慢'}.get(conv_speed, '')
            explanation_parts.append(f"算法经过{total_gen}代进化，{speed_desc}收敛到最优解。")
        
        # 关键参数调整
        key_adjustments = parameter_analysis.get('key_adjustments', [])
        if key_adjustments:
            explanation_parts.append(f"主要参数调整包括：{', '.join(key_adjustments[:3])}。")
        
        # 效果评估
        effectiveness = effectiveness_analysis.get('effectiveness_rating', 'unknown')
        rating_desc = {
            'excellent': '优秀',
            'good': '良好', 
            'moderate': '一般',
            'limited': '有限',
            'poor': '较差'
        }.get(effectiveness, '未知')
        
        explanation_parts.append(f"整体优化效果评定为{rating_desc}。")
        
        return ' '.join(explanation_parts)


def create_explanation_generator(optimization_type: str, **kwargs) -> BaseExplanationGenerator:
    """
    创建解释生成器工厂函数
    
    参数:
        optimization_type: 'nsga' 或 'de'
        kwargs: 传递给生成器的参数
    
    返回:
        解释生成器实例
    """
    if optimization_type.lower() == 'nsga':
        return NSGAExplanationGenerator(**kwargs)
    elif optimization_type.lower() == 'de':
        return DEExplanationGenerator(**kwargs)
    else:
        raise ValueError(f"不支持的优化类型: {optimization_type}")
