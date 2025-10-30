import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .defend_expertsystem import DefendExpertSystem
from .attack_expertsystem import ExpertSystem as AttackExpertSystem


# 默认决策标志字典
DEFAULT_DECISION_FLAGS = {
    'use_defense_decision': 0,
    'use_attack_decision': 0,
    'use_support_decision': 0,
    'use_radar_decision': 1
}


class ExpertSystem:
    """
    集成专家系统：整合攻击和防御专家系统
    
    功能：
    1. 识别被导弹锁定的我方飞机
    2. 高威胁飞机使用防御专家系统
    3. 未锁定飞机使用攻击专家系统
    4. 提供统一的输出接口
    """
    
    def __init__(self, 
                 attack_mode: str = "optimization",
                 defend_mode: str = "de", 
                 high_threat_threshold: float = 0.5,
                 low_threat_threshold: float = 0.3,
                 attack_system: Optional[AttackExpertSystem] = None,
                 defend_system: Optional[DefendExpertSystem] = None,
                 **kwargs):
        """
        初始化集成专家系统
        
        参数:
            attack_mode: 攻击系统模式 ("optimization" 或 "behavior_tree")
            defend_mode: 防御系统模式 ("de" 或 "bt")
            high_threat_threshold: 高威胁阈值 (>此值触发防御)
            low_threat_threshold: 低威胁阈值 (<此值视为未锁定)
            attack_system: 外部传入的攻击系统实例
            defend_system: 外部传入的防御系统实例
            kwargs: 其他配置参数
        """
        print(f"初始化集成专家系统...")
        print(f"攻击模式: {attack_mode}, 防御模式: {defend_mode}")
        print(f"威胁阈值: 高={high_threat_threshold}, 低={low_threat_threshold}")
        
        # 威胁评估阈值
        self.high_threat_threshold = high_threat_threshold
        self.low_threat_threshold = low_threat_threshold
        
        # 创建或使用传入的子系统
        if defend_system is not None:
            self.defend_system = defend_system
        else:
            # 使用默认配置创建防御系统
            defend_config = kwargs.get('defend_config', {})
            self.defend_system = DefendExpertSystem(
                mode=defend_mode, 
                verbose=kwargs.get('verbose', False),
                **defend_config
            )
        
        if attack_system is not None:
            self.attack_system = attack_system
        else:
            # 攻击系统需要threat_model和optimization_module参数
            # 如果没有提供这些参数，则设置为None，后续可通过set_attack_system设置
            attack_config = kwargs.get('attack_config', {})
            if 'threat_model' in attack_config and 'optimization_module' in attack_config:
                self.attack_system = AttackExpertSystem(
                    threat_model=attack_config['threat_model'],
                    optimization_module=attack_config['optimization_module'],
                    mode=attack_mode,
                    verbose=kwargs.get('verbose', False),
                    **{k: v for k, v in attack_config.items() if k not in ['threat_model', 'optimization_module']}
                )
            else:
                print(f"警告: 缺少攻击系统必需参数(threat_model, optimization_module)，攻击系统将设置为None")
                print(f"可以稍后通过set_attack_system()方法设置攻击系统")
                self.attack_system = None
        
        # 存储决策历史
        self.last_decision_summary = {}
        self.last_explanations = []
        
        print("集成专家系统初始化完成")
    
    def set_attack_system(self, attack_system: AttackExpertSystem):
        """设置攻击专家系统实例"""
        self.attack_system = attack_system
        if hasattr(attack_system, 'set_mode'):
            print(f"攻击系统模式已设置")
    
    def set_defend_system(self, defend_system: DefendExpertSystem):
        """设置防御专家系统实例"""
        self.defend_system = defend_system
        print(f"防御系统已设置")
    
    def set_thresholds(self, high_threshold: float, low_threshold: float):
        """
        设置威胁阈值
        
        参数:
            high_threshold: 高威胁阈值
            low_threshold: 低威胁阈值
        """
        if high_threshold <= low_threshold:
            raise ValueError("高威胁阈值必须大于低威胁阈值")
        
        self.high_threat_threshold = high_threshold
        self.low_threat_threshold = low_threshold
        print(f"威胁阈值已更新: 高={high_threshold}, 低={low_threshold}")
    
    def set_attack_mode(self, mode: str):
        """设置攻击系统模式"""
        if self.attack_system and hasattr(self.attack_system, 'set_mode'):
            self.attack_system.set_mode(mode)
        else:
            print("攻击系统未设置或不支持模式切换")
    
    def set_defend_mode(self, mode: str):
        """设置防御系统模式"""
        if hasattr(self.defend_system, 'mode'):
            self.defend_system.mode = mode
            print(f"防御系统模式已切换到: {mode}")
        else:
            print("防御系统不支持模式切换")
    
    def process(self, state: Dict) -> Dict:
        """
        处理战场状态并生成所有飞机的动作
        
        参数:
            state: 战场状态，格式：
            {
                'our_aircrafts': [飞机状态列表],
                'enemies': [敌机状态列表], 
                'missiles': [导弹状态列表]
            }
        
        返回:
            {
                'actions': [所有飞机的动作],
                'explanations': [对应的解释],
                'decision_summary': {决策总结}
            }
        """
        # 验证输入
        if not self._validate_state(state):
            raise ValueError("输入状态格式不正确")
        
        # 1. 识别被导弹锁定的飞机
        locked_aircrafts = self._identify_locked_aircrafts(state)
        
        # 1.5. 获取敌机分配（为防御系统提供攻击目标信息）
        enemy_assignments = self._get_enemy_assignments(state)
        
        # 2. 处理被锁定的飞机（评估威胁并决策）
        defend_results = {}
        for aircraft_id, missiles in locked_aircrafts.items():
            aircraft = state['our_aircrafts'][aircraft_id]
            
            # 为飞机添加分配的敌机位置信息
            if enemy_assignments and aircraft_id < len(enemy_assignments):
                
                enemy_idx = enemy_assignments[aircraft_id]
                if enemy_idx is None:
                    aircraft['assigned_enemy'] = None
                elif 0 <= enemy_idx < len(state.get('enemies', [])):
                    aircraft['assigned_enemy'] = state['enemies'][enemy_idx]['position']
                else:
                    aircraft['assigned_enemy'] = None
            else:
                aircraft['assigned_enemy'] = None
            
            threat_value = self._evaluate_threat(aircraft, missiles)
            
            if threat_value > self.high_threat_threshold:
                # 高威胁：使用防御系统
                defend_state = {
                    'aircraft': aircraft,
                    'missiles': missiles
                }
                result = self.defend_system.process(defend_state)
                print("高威胁：使用防御系统")
                
                defend_results[aircraft_id] = {
                    'action': result['action'] if isinstance(result, dict) and 'action' in result else result,
                    'explanation': f"防御决策(威胁值:{threat_value:.3f}): {result.get('explanation', '使用防御系统规避威胁') if isinstance(result, dict) else '执行防御机动'}",
                    'source': 'defend',
                    'threat_value': threat_value
                }
            else:
                # 低威胁：标记为未锁定
                print("低威胁：标记为未锁定")
                defend_results[aircraft_id] = {
                    'source': 'unlocked_low_threat',
                    'threat_value': threat_value
                }
        
        # 3. 收集未锁定的飞机（包括低威胁飞机）
        unlocked_ids = []
        for i in range(len(state['our_aircrafts'])):
            if i not in locked_aircrafts:
                # 未被导弹锁定
                unlocked_ids.append(i)
            elif i in defend_results and defend_results[i]['source'] == 'unlocked_low_threat':
                # 被锁定但威胁值低
                unlocked_ids.append(i)
        
        # 4. 处理未锁定飞机（使用攻击系统）
        attack_results = {}
        if unlocked_ids and self.attack_system:
            # 构建攻击系统输入
            attack_state = {
                'our_aircrafts': [state['our_aircrafts'][i] for i in unlocked_ids],
                'enemies': state.get('enemies', [])
            }
            print("未锁定飞机（使用攻击系统）")
            try:
                # 调用攻击系统 - 修正方法调用
                if hasattr(self.attack_system, 'get_all_friendly_actions'):
                    attack_response = self.attack_system.get_all_friendly_actions(attack_state)
                elif hasattr(self.attack_system, 'process'):
                    attack_response = self.attack_system.process(attack_state)
                else:
                    # 尝试直接调用系统
                    attack_response = self.attack_system(attack_state)
                
                # 处理攻击系统输出
                if isinstance(attack_response, dict):
                    actions = attack_response.get('actions', [])
                    explanations = attack_response.get('explanations', [])
                elif isinstance(attack_response, list):
                    actions = attack_response
                    explanations = ['攻击决策'] * len(actions)
                else:
                    # 单个动作
                    actions = [attack_response] * len(unlocked_ids)
                    explanations = ['攻击决策'] * len(unlocked_ids)
                
                # 映射回原始ID
                for idx, aircraft_id in enumerate(unlocked_ids):
                    if idx < len(actions):
                        attack_results[aircraft_id] = {
                            'action': actions[idx],
                            'explanation': f"攻击决策: {explanations[idx] if idx < len(explanations) else '协同攻击'}",
                            'source': 'attack'
                        }
                    else:
                        attack_results[aircraft_id] = {
                            'action': state['our_aircrafts'][aircraft_id],
                            'explanation': "攻击系统输出不足，保持原状态",
                            'source': 'default'
                        }
                        
            except Exception as e:
                print(f"攻击系统调用失败: {e}")
                print(f"攻击系统类型: {type(self.attack_system)}")
                if hasattr(self.attack_system, '__dict__'):
                    print(f"攻击系统属性: {list(self.attack_system.__dict__.keys())}")
                
                # 攻击系统失败时，未锁定飞机保持原状态
                for aircraft_id in unlocked_ids:
                    attack_results[aircraft_id] = {
                        'action': state['our_aircrafts'][aircraft_id],
                        'explanation': f"攻击系统异常，保持原状态: {str(e)[:100]}",
                        'source': 'error'
                    }
        else:
            # 无攻击系统或无未锁定飞机
            for aircraft_id in unlocked_ids:
                attack_results[aircraft_id] = {
                    'action': state['our_aircrafts'][aircraft_id],
                    'explanation': "无攻击系统，保持原状态" if not self.attack_system else "无需攻击决策",
                    'source': 'default'
                }
        
        # 5. 整合所有结果
        all_results = {**defend_results, **attack_results}
        
        # 6. 格式化输出
        return self._format_output(all_results, state)
    
    def _identify_locked_aircrafts(self, state: Dict) -> Dict[int, List]:
        """
        识别被导弹锁定的飞机
        
        参数:
            state: 战场状态
        
        返回:
            字典，键为飞机ID，值为锁定该飞机的导弹列表
        """
        locked_aircrafts = {}
        
        for missile in state.get('missiles', []):
            if 'target' in missile:
                target_id = missile['target']
                if 0 <= target_id < len(state['our_aircrafts']):
                    if target_id not in locked_aircrafts:
                        locked_aircrafts[target_id] = []
                    locked_aircrafts[target_id].append(missile)
        
        return locked_aircrafts
    
    def _evaluate_threat(self, aircraft: Dict, missiles: List) -> float:
        """
        评估威胁值
        
        参数:
            aircraft: 飞机状态 (位置单位：千米)
            missiles: 导弹列表 (位置单位：千米)
        
        返回:
            威胁值 (0-1)
        """
        if not missiles:
            return 0.0
        
        try:
            # 检查输入数据单位并转换为米制
            from expert.defend_expertsystem import convert_state_units_to_meters
            
            # 构建临时状态用于单位转换
            temp_state = {
                'aircraft': aircraft,
                'missiles': missiles
            }
            converted_state = convert_state_units_to_meters(temp_state)
            
            # 使用转换后的数据进行威胁评估
            return self.defend_system.evaluate_current_threat(
                converted_state['aircraft'], 
                converted_state['missiles']
            )
        except Exception as e:
            print(f"威胁评估失败: {e}")
            return 0.5  # 默认中等威胁  """

    
    def _format_output(self, results: Dict, state: Dict) -> Dict:
        """
        格式化输出结果
        
        参数:
            results: 处理结果字典
            state: 原始状态
        
        返回:
            格式化的输出
        """
        actions = []
        explanations = []
        
        # 按飞机ID顺序整理结果
        for i in range(len(state['our_aircrafts'])):
            if i in results and 'action' in results[i]:
                action = results[i]['action']
                # 确保动作包含决策标志（如果子系统没有添加的话）
                if isinstance(action, dict) and not any(key in action for key in DEFAULT_DECISION_FLAGS.keys()):
                    action.update(DEFAULT_DECISION_FLAGS.copy())
                actions.append(action)
                explanations.append(results[i]['explanation'])
            else:
                # 未处理的飞机保持原状态并添加默认决策标志
                default_action = state['our_aircrafts'][i].copy()
                if isinstance(default_action, dict):
                    default_action.update(DEFAULT_DECISION_FLAGS.copy())
                actions.append(default_action)
                explanations.append("未受威胁，保持原状态")
        
        # 生成决策总结
        decision_summary = self._generate_decision_summary(results, state)
        
        # 保存历史
        self.last_decision_summary = decision_summary
        self.last_explanations = explanations
        
        return {
            'actions': actions,
            'explanations': explanations,
            'decision_summary': decision_summary
        }
    
    def _generate_decision_summary(self, results: Dict, state: Dict) -> Dict:
        """生成决策总结"""
        defend_count = sum(1 for r in results.values() if r.get('source') == 'defend')
        attack_count = sum(1 for r in results.values() if r.get('source') == 'attack')
        unlocked_count = sum(1 for r in results.values() if r.get('source', '').startswith('unlocked'))
        
        # 威胁统计
        threat_values = [r.get('threat_value', 0) for r in results.values() if 'threat_value' in r]
        avg_threat = np.mean(threat_values) if threat_values else 0.0
        max_threat = max(threat_values) if threat_values else 0.0
        
        return {
            'total_aircrafts': len(state['our_aircrafts']),
            'defend_aircrafts': defend_count,
            'attack_aircrafts': attack_count,
            'unlocked_aircrafts': unlocked_count,
            'missile_threats': len(state.get('missiles', [])),
            'average_threat': float(avg_threat),
            'max_threat': float(max_threat),
            'high_threat_threshold': self.high_threat_threshold,
            'low_threat_threshold': self.low_threat_threshold,
            'systems_status': {
                'defend_system': 'active' if hasattr(self, 'defend_system') else 'inactive',
                'attack_system': 'active' if self.attack_system else 'inactive'
            }
        }
    
    def _get_enemy_assignments(self, state: Dict) -> List[int]:
        """
        获取各友方飞机的最佳攻击目标敌机索引
        
        参数:
            state: 战场状态
        
        返回:
            敌机索引列表，每个元素对应一个友方飞机的最佳攻击目标
        """
        try:
            # 检查是否有敌机信息
            if 'enemies' not in state or not state['enemies']:
                return [None] * len(state['our_aircrafts'])
            
            # 优先使用攻击系统的威胁模型
            if self.attack_system and hasattr(self.attack_system, 'threat_model'):
                print("使用攻击系统的威胁模型获取敌机分配")
                threat_model = self.attack_system.threat_model
                
                # 处理状态数据获取最佳目标分配
                threat_model.process_state(state)
                enemy_assignments = threat_model.get_optimal_target_per_friendly()
                
                return enemy_assignments
            
            else:
                # 如果没有攻击系统，使用独立威胁模型
                print("使用独立威胁模型获取敌机分配")
                from expert.evalthreat.Mutilair_to_Mutilair import ThreatModel
                
                # 使用完整的参数配置创建威胁模型
                our_params = {
                    'angle_params': {'theta_Rmax': 80, 'theta_Max': 50, 'theta_Maxmin': 30},
                    'distance_params': {'D_Rmax': 120, 'D_Mmax': 60, 'D_Mmin': 5, 'D_NZmax': 30, 'D_NZmin': 5}
                }
                enemy_params = {
                    'angle_params': {'phi_Rmax': 70, 'phi_Max': 40, 'phi_Maxmin': 20},
                    'distance_params': {'D_Rmax': 100, 'D_Mmax': 40, 'D_Mmin': 5, 'D_NZmax': 20, 'D_NZmin': 2}
                }
                speed_params = {
                    'D_high': 80, 'V_high': 350, 'D_low': 30, 'V_low': 250, 'V_base': 300
                }
                
                threat_model = ThreatModel(
                    our_params=our_params,
                    enemy_params=enemy_params,
                    speed_params=speed_params
                )
                
                # 处理状态数据获取最佳目标分配
                threat_model.process_state(state)
                enemy_assignments = threat_model.get_optimal_target_per_friendly()
                
                return enemy_assignments
            
        except Exception as e:
            print(f"获取敌机分配失败: {e}")
            # 返回默认分配（按顺序分配）
            num_aircrafts = len(state['our_aircrafts'])
            num_enemies = len(state.get('enemies', []))
            if num_enemies == 0:
                return [None] * num_aircrafts
            return [i % num_enemies for i in range(num_aircrafts)]
    
    def _validate_state(self, state: Dict) -> bool:
        """验证输入状态格式"""
        required_keys = ['our_aircrafts']
        if not all(key in state for key in required_keys):
            return False
        
        if not isinstance(state['our_aircrafts'], list) or len(state['our_aircrafts']) == 0:
            return False
        
        return True
    
    # ========== 外部接口方法 ==========
    
    def get_all_actions(self, state: Dict) -> Dict:
        """
        获取我方全体飞机的指导动作和相应解释
        
        参数:
            state: 战场状态
        
        返回:
            包含actions和explanations的字典
        """
        return self.process(state)
    
    def get_aircraft_action(self, aircraft_id: int, state: Dict) -> Dict:
        """
        获得特定我方飞机的指导动作和相应解释
        
        参数:
            aircraft_id: 飞机ID (从0开始)
            state: 战场状态
        
        返回:
            包含单个飞机动作和解释的字典
        """
        if aircraft_id < 0 or aircraft_id >= len(state.get('our_aircrafts', [])):
            return {
                'aircraft_id': aircraft_id,
                'action': None,
                'explanation': f"飞机ID {aircraft_id} 超出范围",
                'error': True
            }
        
        # 处理整体状态
        result = self.process(state)
        
        return {
            'aircraft_id': aircraft_id,
            'action': result['actions'][aircraft_id],
            'explanation': result['explanations'][aircraft_id],
            'error': False
        }
    
    def get_decision_summary(self, state: Optional[Dict] = None) -> Dict:
        """
        获取决策总结
        
        参数:
            state: 可选的战场状态，如果提供则重新计算
        
        返回:
            决策总结字典
        """
        if state is not None:
            result = self.process(state)
            return result['decision_summary']
        else:
            return self.last_decision_summary
    
    def get_system_info(self) -> Dict:
        """
        获取系统信息
        
        返回:
            系统配置和状态信息
        """
        info = {
            'thresholds': {
                'high_threat': self.high_threat_threshold,
                'low_threat': self.low_threat_threshold
            },
            'systems': {
                'defend_system': {
                    'available': hasattr(self, 'defend_system'),
                    'mode': getattr(self.defend_system, 'mode', 'unknown') if hasattr(self, 'defend_system') else None
                },
                'attack_system': {
                    'available': self.attack_system is not None,
                    'mode': getattr(self.attack_system, 'mode', 'unknown') if self.attack_system else None
                }
            }
        }
        
        # 添加子系统详细信息
        if hasattr(self, 'defend_system') and hasattr(self.defend_system, 'get_mode_info'):
            info['defend_system_info'] = self.defend_system.get_mode_info()
        
        if self.attack_system and hasattr(self.attack_system, 'get_mode_info'):
            info['attack_system_info'] = self.attack_system.get_mode_info()
        
        return info
    
    def print_system_status(self):
        """打印系统状态"""
        print("=" * 50)
        print("集成专家系统状态")
        print("=" * 50)
        
        info = self.get_system_info()
        
        print(f"威胁阈值设置:")
        print(f"  高威胁阈值: {info['thresholds']['high_threat']}")
        print(f"  低威胁阈值: {info['thresholds']['low_threat']}")
        
        print(f"\n子系统状态:")
        print(f"  防御系统: {'可用' if info['systems']['defend_system']['available'] else '不可用'}")
        if info['systems']['defend_system']['available']:
            print(f"    模式: {info['systems']['defend_system']['mode']}")
        
        print(f"  攻击系统: {'可用' if info['systems']['attack_system']['available'] else '不可用'}")
        if info['systems']['attack_system']['available']:
            print(f"    模式: {info['systems']['attack_system']['mode']}")
        
        if hasattr(self, 'last_decision_summary') and self.last_decision_summary:
            summary = self.last_decision_summary
            print(f"\n最近决策总结:")
            print(f"  处理飞机数: {summary.get('total_aircrafts', 0)}")
            print(f"  防御飞机数: {summary.get('defend_aircrafts', 0)}")
            print(f"  攻击飞机数: {summary.get('attack_aircrafts', 0)}")
            print(f"  导弹威胁数: {summary.get('missile_threats', 0)}")
            print(f"  平均威胁值: {summary.get('average_threat', 0):.3f}")
            print(f"  最大威胁值: {summary.get('max_threat', 0):.3f}")
        
        print("=" * 50)


def create_expert_system_from_config(config_path: str) -> ExpertSystem:
    """
    从配置文件创建集成专家系统
    
    参数:
        config_path: 配置文件路径
    
    返回:
        ExpertSystem实例
    """
    import yaml
    from common.registry import build
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 构建攻击系统
    attack_system = None
    if 'attack_system' in config:
        attack_system = build(config, 'attack_system')
    
    # 构建防御系统
    defend_system = None
    if 'defend_system' in config:
        defend_system = build(config, 'defend_system')
    
    # 获取系统配置
    system_config = config.get('expert_system', {})
    
    # 创建集成系统
    expert_system = ExpertSystem(
        attack_system=attack_system,
        defend_system=defend_system,
        **system_config
    )
    
    return expert_system


if __name__ == "__main__":
    # 演示示例
    print("集成专家系统演示")
    print("=" * 60)
    
    # 创建系统实例
    expert = ExpertSystem(
        attack_mode="optimization",
        defend_mode="de",
        high_threat_threshold=0.7,
        low_threat_threshold=0.3,
        verbose=True
    )
    
    # 示例战场状态
    state = {
        'our_aircrafts': [
            {'speed': 250, 'height': 8000, 'heading': 45, 'position': np.array([15.0, 10.0])},
            {'speed': 245, 'height': 8100, 'heading': 50, 'position': np.array([18.0, 12.0])},
            {'speed': 255, 'height': 7900, 'heading': 42, 'position': np.array([20.0, 16.0])}
        ],
        'enemies': [
            {'speed': 230, 'height': 7600, 'heading': 210, 'position': np.array([50.0, 52.0])},
            {'speed': 235, 'height': 7700, 'heading': 205, 'position': np.array([53.0, 48.0])},
            {'speed': 238, 'height': 7500, 'heading': 212, 'position': np.array([49.0, 54.0])}
        ],
        'missiles': [
            {"position": [1000, 500], "height": 800, "speed": 250, "target": 0},
            {"position": [2000, 1000], "height": 900, "speed": 200, "target": 1},
            {"position": [500, 300], "height": 700, "speed": 280, "target": 0}
        ]
    }
    
    print("\n示例战场状态:")
    print(f"我方飞机数: {len(state['our_aircrafts'])}")
    print(f"敌方飞机数: {len(state['enemies'])}")
    print(f"导弹威胁数: {len(state['missiles'])}")
    
    # 打印系统状态
    expert.print_system_status()
    
    # 注意：实际使用需要正确配置攻击系统
    print(f"\n注意: 本演示仅展示系统结构，完整功能需要配置攻击和防御子系统")
    
    # 测试接口（仅防御功能）
    try:
        print(f"\n测试获取特定飞机动作:")
        aircraft_result = expert.get_aircraft_action(0, state)
        print(f"飞机0: {aircraft_result['explanation']}")
    except Exception as e:
        print(f"测试失败（预期的，需要完整配置）: {e}")
    
    print(f"\n演示完成")
