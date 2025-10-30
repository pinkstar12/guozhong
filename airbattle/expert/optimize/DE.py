import numpy as np
from typing import Callable, List, Tuple, Optional, Dict, Any
from .explanation_generator import DEExplanationGenerator

class DifferentialEvolution:
    """
    差分进化算法优化器
    用于优化飞机位置和朝向，最小化导弹威胁度
    """
    
    def __init__(self, 
                 obj_func: Callable,
                 bounds: List[Tuple[float, float]],
                 pop_size: int = 50,
                 F: float = 0.8,
                 CR: float = 0.7,
                 strategy: str = 'rand/1/bin',
                 max_iter: int = 1000,
                 tolerance: float = 1e-6,
                 seed: Optional[int] = None):
        """
        初始化DE优化器
        
        参数:
            obj_func: 目标函数，接收参数向量，返回目标值
            bounds: 参数边界列表 [(min1, max1), (min2, max2), ...]
            pop_size: 种群大小
            F: 差分权重因子 (0.4-1.0)
            CR: 交叉概率 (0.0-1.0)
            strategy: 变异策略
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            seed: 随机种子
        """
        self.obj_func = obj_func
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.strategy = strategy
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        
        # 问题维度
        self.dim = len(bounds)
        self.lower_bounds = self.bounds[:, 0]
        self.upper_bounds = self.bounds[:, 1]
        
        # 历史记录
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'best_solution': None,
            'convergence_iter': None
        }
        
        # 初始化种群
        self.population = None
        self.fitness = None
        self.best_idx = None
        self.best_fitness = float('inf')
        self.best_solution = None
    
    def _initialize_population(self) -> np.ndarray:
        """初始化种群"""
        population = np.random.uniform(
            low=self.lower_bounds,
            high=self.upper_bounds,
            size=(self.pop_size, self.dim)
        )
        return population
    
    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """评估种群适应度"""
        fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            fitness[i] = self.obj_func(population[i])
        return fitness
    
    def _mutation(self, population: np.ndarray, target_idx: int) -> np.ndarray:
        """变异操作"""
        # 选择三个不同的个体（不包括目标个体）
        candidates = list(range(self.pop_size))
        candidates.remove(target_idx)
        
        if self.strategy == 'rand/1/bin':
            # 随机选择三个个体
            a, b, c = np.random.choice(candidates, 3, replace=False)
            # 变异向量: x_a + F * (x_b - x_c)
            mutant = population[a] + self.F * (population[b] - population[c])
        
        elif self.strategy == 'best/1/bin':
            # 基于最优个体的变异
            a, b = np.random.choice(candidates, 2, replace=False)
            mutant = self.best_solution + self.F * (population[a] - population[b])
        
        elif self.strategy == 'rand/2/bin':
            # 使用五个个体的变异
            if len(candidates) >= 5:
                a, b, c, d, e = np.random.choice(candidates, 5, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c]) + self.F * (population[d] - population[e])
            else:
                # 回退到rand/1/bin策略
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
        
        else:
            raise ValueError(f"未知的变异策略: {self.strategy}")
        
        return mutant
    
    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """交叉操作"""
        trial = np.copy(target)
        
        # 二项交叉
        crossover_mask = np.random.random(self.dim) < self.CR
        
        # 确保至少有一个参数被交叉
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        
        trial[crossover_mask] = mutant[crossover_mask]
        
        return trial
    
    def _bound_constraint(self, individual: np.ndarray) -> np.ndarray:
        """边界约束处理"""
        # 镜像边界处理
        bounded = np.copy(individual)
        
        for i in range(self.dim):
            while bounded[i] < self.lower_bounds[i] or bounded[i] > self.upper_bounds[i]:
                if bounded[i] < self.lower_bounds[i]:
                    bounded[i] = 2 * self.lower_bounds[i] - bounded[i]
                elif bounded[i] > self.upper_bounds[i]:
                    bounded[i] = 2 * self.upper_bounds[i] - bounded[i]
        
        return bounded
    
    def _update_best(self, population: np.ndarray, fitness: np.ndarray) -> None:
        """更新最优解"""
        current_best_idx = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]
        
        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_solution = np.copy(population[current_best_idx])
            self.best_idx = current_best_idx
    
    def _check_convergence(self, fitness: np.ndarray) -> bool:
        """检查收敛条件"""
        # 检查适应度方差是否足够小
        fitness_std = np.std(fitness)
        # 确保tolerance是数值类型
        tolerance_val = float(self.tolerance) if isinstance(self.tolerance, str) else self.tolerance
        return fitness_std < tolerance_val
    
    def optimize(self, verbose: bool = False) -> Dict[str, Any]:
        """
        执行差分进化优化
        
        参数:
            verbose: 是否打印详细信息
        
        返回:
            优化结果字典
        """
        # 初始化种群
        self.population = self._initialize_population()
        self.fitness = self._evaluate_population(self.population)
        self._update_best(self.population, self.fitness)
        
        # 记录初始状态
        self.history['best_fitness'].append(self.best_fitness)
        self.history['mean_fitness'].append(np.mean(self.fitness))
        
        if verbose:
            print(f"DE优化开始...")
            print(f"初始最优适应度: {self.best_fitness:.6f}")
        
        # 主优化循环
        for iteration in range(self.max_iter):
            new_population = np.zeros_like(self.population)
            
            # 对每个个体执行变异、交叉和选择
            for i in range(self.pop_size):
                # 变异
                mutant = self._mutation(self.population, i)
                
                # 边界约束
                mutant = self._bound_constraint(mutant)
                
                # 交叉
                trial = self._crossover(self.population[i], mutant)
                
                # 边界约束
                trial = self._bound_constraint(trial)
                
                # 选择
                trial_fitness = self.obj_func(trial)
                
                if trial_fitness < self.fitness[i]:
                    new_population[i] = trial
                    self.fitness[i] = trial_fitness
                else:
                    new_population[i] = self.population[i]
            
            # 更新种群
            self.population = new_population
            
            # 更新最优解
            self._update_best(self.population, self.fitness)
            
            # 记录历史
            self.history['best_fitness'].append(self.best_fitness)
            self.history['mean_fitness'].append(np.mean(self.fitness))
            
            # 检查收敛
            if self._check_convergence(self.fitness):
                if verbose:
                    print(f"在第 {iteration + 1} 代收敛")
                self.history['convergence_iter'] = iteration + 1
                break
            
            # 打印进度
            if verbose and (iteration + 1) % 50 == 0:
                print(f"第 {iteration + 1} 代: 最优适应度 = {self.best_fitness:.6f}, "
                      f"平均适应度 = {np.mean(self.fitness):.6f}")
        
        if verbose:
            final_iter = self.history['convergence_iter'] or self.max_iter
            print(f"优化完成! 总迭代次数: {final_iter}")
            print(f"最优适应度: {self.best_fitness:.6f}")
            print(f"最优解: {self.best_solution}")
        
        # 返回结果
        result = {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'history': self.history,
            'success': True,
            'message': 'Optimization completed successfully'
        }
        
        return result
    
    def get_optimization_history(self) -> Dict[str, List]:
        """获取优化历史记录"""
        return self.history


class DEOptimization:
    """
    DE优化模块，用于替代NSGA-II
    与现有系统架构兼容
    """
    
    def __init__(self, threat_evaluator=None, **config):
        """
        初始化DE优化模块
        
        参数:
            threat_evaluator: 威胁评估器实例
            config: 优化配置参数
        """
        self.threat_evaluator = threat_evaluator
        self.bounds = None
        self.optimization_params = None
        self.initialize(threat_evaluator, config)
    
    def initialize(self, threat_evaluator, config):
        """
        初始化优化参数
        
        参数:
            threat_evaluator: 威胁评估器实例
            config: 优化配置参数
        """
        self.threat_evaluator = threat_evaluator
        
        # 设置参数边界
        self.bounds = {
            'delta_speed': 20,
            'delta_height': 1000,
            'delta_heading': 30,
            'delta_pos': 10,
            'abs_speed': (200, 300),
            'abs_height': (5000, 10000),
            'abs_heading': (0, 360),
            'abs_pos_x': (-180, 180),  # 经度范围
            'abs_pos_z': (-90, 90),    # 纬度范围
        }
        
        self.bounds.update(config.get('bounds', {}))
        
        # 设置DE优化参数
        self.optimization_params = config.get('optimization_params', {
            'pop_size': 50,
            'F': 0.8,
            'CR': 0.7,
            'max_iter': 1000,
            'strategy': 'rand/1/bin',
            'tolerance': 1e-6
        })
    
    def _create_objective_function(self, missiles: List[Dict]) -> Callable:
        """
        创建目标函数
        
        参数:
            missiles: 导弹列表
        
        返回:
            目标函数
        """
        def objective(params):
            """
            目标函数：最小化威胁度
            
            参数:
                params: [x, z, height, speed, heading] 飞机参数
            
            返回:
                威胁度值
            """
            # 构建飞机状态
            plane = {
                'position': [params[0], params[1]],  # [x, z]
                'height': params[2],
                'speed': params[3],
                'heading': params[4] if len(params) > 4 else 0  # 可选朝向参数
            }
            
            # 计算威胁度
            if hasattr(self.threat_evaluator, 'evaluate_threats'):
                threats = self.threat_evaluator.evaluate_threats(plane, missiles)
                # 返回最大威胁度或平均威胁度
                return max(threats) if threats else 0.0
            else:
                # 备用方案：直接调用威胁评估函数
                from ..evalthreat.SingleAir_to_MulitMissile import evaluate_air_to_missiles
                threats = evaluate_air_to_missiles(plane, missiles)
                return max(threats) if threats else 0.0
        
        return objective
    
    def _setup_bounds(self, initial_plane: Dict) -> List[Tuple[float, float]]:
        """
        设置优化边界
        
        参数:
            initial_plane: 初始飞机状态
        
        返回:
            边界列表
        """
        bounds = []
        
        # 位置边界 (x)
        x = initial_plane['position'][0]
        bounds.append((
            max(self.bounds['abs_pos_x'][0], x - self.bounds['delta_pos']),
            min(self.bounds['abs_pos_x'][1], x + self.bounds['delta_pos'])
        ))
        
        # 位置边界 (z)
        z = initial_plane['position'][1]
        bounds.append((
            max(self.bounds['abs_pos_z'][0], z - self.bounds['delta_pos']),
            min(self.bounds['abs_pos_z'][1], z + self.bounds['delta_pos'])
        ))
        
        # 高度边界
        h = initial_plane['height']
        bounds.append((
            max(self.bounds['abs_height'][0], h - self.bounds['delta_height']),
            min(self.bounds['abs_height'][1], h + self.bounds['delta_height'])
        ))
        
        # 速度边界
        v = initial_plane['speed']
        bounds.append((
            max(self.bounds['abs_speed'][0], v - self.bounds['delta_speed']),
            min(self.bounds['abs_speed'][1], v + self.bounds['delta_speed'])
        ))
        
        # 朝向边界（可选）
        if 'heading' in initial_plane:
            heading = initial_plane['heading']
            bounds.append((
                max(self.bounds['abs_heading'][0], heading - self.bounds['delta_heading']),
                min(self.bounds['abs_heading'][1], heading + self.bounds['delta_heading'])
            ))
        
        return bounds
    
    def process_state(self, state: Dict) -> List[Tuple[List[Dict], List[float]]]:
        """
        处理状态数据并生成优化决策
        
        参数:
            state: 状态字典，包含飞机和导弹信息
        
        返回:
            优化结果列表 [(actions, objectives), ...]
        """
        # 提取飞机和导弹信息
        our_aircrafts = state.get('our_aircrafts', [])
        missiles = state.get('missiles', [])
        
        if not our_aircrafts or not missiles:
            return []
        
        # 目前只处理单架飞机的情况
        aircraft = our_aircrafts[0]
        
        # 创建目标函数
        objective_func = self._create_objective_function(missiles)
        
        # 设置边界
        bounds = self._setup_bounds(aircraft)
        
        # 创建DE优化器
        de_optimizer = DifferentialEvolution(
            obj_func=objective_func,
            bounds=bounds,
            **self.optimization_params
        )
        
        # 执行优化
        result = de_optimizer.optimize(verbose=False)
        
        # 解析结果
        if result['success']:
            best_params = result['best_solution']
            
            # 构建优化后的飞机状态
            optimized_aircraft = {
                'position': [best_params[0], best_params[1]],
                'height': best_params[2],
                'speed': best_params[3],
                'heading': best_params[4] if len(best_params) > 4 else aircraft.get('heading', 0)
            }
            
            # 计算目标函数值
            threat_value = result['best_fitness']
            
            # 返回兼容格式
            return [([optimized_aircraft], [threat_value, 0.0])]  # 第二个目标设为0
        
        else:
            # 优化失败，返回原始状态
            return [([aircraft], [1.0, 1.0])]
    
    def generate_explanation(self, optimization_result, initial_state=None):
        """
        生成DE优化结果解释
        
        参数:
            optimization_result: DE优化结果字典
            initial_state: 初始状态（可选）
        
        返回:
            解释字典
        """
        # 创建解释生成器
        explanation_generator = DEExplanationGenerator(self.optimization_params)
        
        # 生成解释
        explanation = explanation_generator.generate_explanation(
            optimization_result, initial_state
        )
        
        return explanation
