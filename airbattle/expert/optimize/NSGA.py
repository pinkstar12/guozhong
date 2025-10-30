import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from .explanation_generator import NSGAExplanationGenerator

class NSGAOptimization:
    def __init__(self, situation_module=None, **config):
        self.situation_module = situation_module
        self.bounds = None
        self.optimization_params = None
        self.initialize(self.situation_module, config)

    def initialize(self, situation_module, config):
        """
        初始化优化决策模块
        :param situation_module: 态势评估模块实例
        :param config: 优化配置参数
        """
        self.situation_module = situation_module
        self.bounds = {
            'delta_speed': 20,
            'delta_height': 1000,
            'delta_heading': 30,
            'delta_pos': 10,
            'abs_speed': (200, 300),
            'abs_height': (5000, 10000),
            'abs_heading': (0, 360),
            'abs_pos_x': (-180,180),  
            'abs_pos_y': (-90, 90),  
        }

        self.bounds.update(config.get('bounds', {}))
        self.optimization_params = config.get('optimization_params', {
            'pop_size': 50,
            'n_gen': 100
        })

    def process_state(self, state):
        """
        处理状态数据并生成优化决策
        :param state: 全局状态字典
        :return: 优化后的动作（飞机参数列表）
        形状为List[List[Dict]],外层长度等于Pareto解数量，内层长度等于飞机数量，每个Dict存储单个飞机的五个动作参数
        """
        # 从状态中提取我方飞机
        our_aircrafts = state['our_aircrafts']
        num_our = len(our_aircrafts)

        # 1. 针对每架飞机生成各自的bound
        xl = []
        xu = []
        for aircraft in our_aircrafts:
            # 速度
            v = aircraft['speed']
            xl.append(max(self.bounds['abs_speed'][0], v - self.bounds['delta_speed']))
            xu.append(min(self.bounds['abs_speed'][1], v + self.bounds['delta_speed']))
            # 高度
            h = aircraft['height']
            xl.append(max(self.bounds['abs_height'][0], h - self.bounds['delta_height']))
            xu.append(min(self.bounds['abs_height'][1], h + self.bounds['delta_height']))
            # 航向
            heading = aircraft['heading']
            xl.append(max(self.bounds['abs_heading'][0], heading - self.bounds['delta_heading']))
            xu.append(min(self.bounds['abs_heading'][1], heading + self.bounds['delta_heading']))
            # x 坐标（北）
            x = aircraft['position'][0]
            xl.append(max(self.bounds['abs_pos_x'][0], x - self.bounds['delta_pos']))
            xu.append(min(self.bounds['abs_pos_x'][1], x + self.bounds['delta_pos']))
            # y 坐标（东）
            y = aircraft['position'][1]
            xl.append(max(self.bounds['abs_pos_y'][0], y - self.bounds['delta_pos']))
            xu.append(min(self.bounds['abs_pos_y'][1], y + self.bounds['delta_pos']))

        xl = np.array(xl)
        xu = np.array(xu)

        class GuidanceProblem(ElementwiseProblem):
            def __init__(self, outer):
                super().__init__(
                    n_var=5 * num_our,
                    n_obj=2,
                    n_constr=1,
                    xl=xl,
                    xu=xu,
                )
                self.outer = outer

            def _evaluate(self, x, out, *args, **kwargs):
                # 解析优化变量为多架飞机的参数
                params_list = []
                for i in range(num_our):
                    params = {
                        'speed': x[i],
                        'height': x[num_our + i],
                        'heading': x[2 * num_our + i],
                        'position': np.array([
                            x[3 * num_our + i],
                            x[4 * num_our + i]
                        ])
                    }
                    params_list.append(params)

                # 创建临时状态
                temp_state = {
                    'our_aircrafts': params_list,
                    'enemies': state['enemies']
                }

                # 使用态势评估模块计算团队优势值和风险值
                advantage, risk = self.outer.situation_module.process_state(temp_state)

                # 计算约束：飞机间最小距离（避免碰撞）
                min_distance = float('inf')
                positions = [p['position'] for p in params_list]
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        dist = np.linalg.norm(positions[i] - positions[j])
                        if dist < min_distance:
                            min_distance = dist

                # 设置目标函数和约束
                out["F"] = [-advantage, risk]  # 最小化 -advantage 和 risk
                out["G"] = [5.0 - min_distance]  # 约束：最小距离 >= 5km

        # 使用NSGA-II多目标优化算法
        algorithm = NSGA2(
            pop_size=self.optimization_params['pop_size']
        )

        # 执行优化
        problem = GuidanceProblem(outer=self)
        res = minimize(
            problem,
            algorithm,
            ('n_gen', self.optimization_params['n_gen']),
            verbose=False
        )

        # 解析优化结果
        optimal_actions = []
        objectives_list = []
        num_vars_per_aircraft = 5

        # 同时返回动作和目标函数值
        for sol, obj in zip(res.X, res.F):
            aircraft_actions = []
            for i in range(num_our):
                start_idx = i * num_vars_per_aircraft
                aircraft_sol = sol[start_idx:start_idx + num_vars_per_aircraft]
                action = {
                    'speed': aircraft_sol[0],
                    'height': aircraft_sol[1],
                    'heading': aircraft_sol[2],
                    'position': np.array([aircraft_sol[3], aircraft_sol[4]])
                }
                aircraft_actions.append(action)
            optimal_actions.append(aircraft_actions)
            objectives_list.append(obj)  # 保存目标函数值

        # 返回带目标值的解集
        return list(zip(optimal_actions, objectives_list))
    
    def generate_explanation(self, optimization_result, initial_state=None):
        """
        生成NSGA-II优化结果解释
        
        参数:
            optimization_result: 优化结果 [(actions, objectives), ...]
            initial_state: 初始状态（可选）
        
        返回:
            解释字典
        """
        # 创建解释生成器
        explanation_generator = NSGAExplanationGenerator(self.optimization_params)
        
        # 生成解释
        explanation = explanation_generator.generate_explanation(
            optimization_result, initial_state
        )
        
        return explanation
