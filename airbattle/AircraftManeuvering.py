import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

# ====================== 机动参数配置 ======================
@dataclass
class ManeuverConfig:
    """机动参数配置类"""
    max_speed: float = 100.0           # 最大速度 (m/s)
    max_acceleration: float = 15.0     # 最大加速度 (m/s²)
    max_climb_rate: float = 20.0       # 最大爬升率 (m/s)
    max_dive_rate: float = 25.0        # 最大俯冲率 (m/s)
    max_turn_rate: float = 3.0         # 最大转弯率 (rad/s)
    min_altitude: float = 50.0         # 最小飞行高度 (m)
    max_altitude: float = 800.0        # 最大飞行高度 (m)
    combat_radius: float = 200.0       # 作战半径 (m)
    evasion_radius: float = 300.0      # 规避半径 (m)
    formation_distance: float = 150.0  # 编队距离 (m)
    energy_consumption_rate: float = 0.01  # 能量消耗率

class ManeuverType(Enum):
    """机动类型枚举"""
    ATTACK = "attack"
    EVADE = "evade"
    HOLD = "hold"
    RETREAT = "retreat"
    FLANK = "flank"
    CLIMB = "climb"
    DIVE = "dive"

# ====================== 核心机动模型 ======================
class AircraftManeuvering:
    """
    飞机机动模型 - 负责将决策指令转换为具体的机动动作
    """
    
    def __init__(self, config: ManeuverConfig = None):
        self.config = config or ManeuverConfig()
        self.dt = 0.1  # 时间步长 (秒)
        print(f"飞机机动模型初始化完成，时间步长: {self.dt}s")
    
    def execute_maneuvers(self, decisions: Set[Tuple], drone_states: Dict, 
                         enemy_states: Dict, time_step: float = 0.1) -> Dict:
        """
        执行机动指令，更新所有无人机的状态
        
        Args:
            decisions: 决策指令集合 {(drone_id, strategy, params), ...}
            drone_states: 当前无人机状态字典
            enemy_states: 敌方无人机状态字典
            time_step: 时间步长
            
        Returns:
            Dict: 更新后的无人机状态字典
        """
        self.dt = time_step
        updated_states = {}
        
        print(f"\n--- 开始执行机动指令 (时间步长: {self.dt}s) ---")
        
        for drone_id, drone_state in drone_states.items():
            # 查找该无人机的决策指令
            decision = self._find_decision_for_drone(decisions, drone_id)
            
            if decision:
                _, strategy, params = decision
                # 执行对应的机动动作
                new_state = self._execute_single_maneuver(
                    drone_state, strategy, params, drone_states, enemy_states
                )
                updated_states[drone_id] = new_state
                print(f"无人机 {drone_id} 执行 {strategy} 机动")
            else:
                # 如果没有找到决策，保持当前状态
                updated_states[drone_id] = self._maintain_current_state(drone_state)
                print(f"无人机 {drone_id} 保持当前状态")
        
        return updated_states
    
    def _find_decision_for_drone(self, decisions: Set[Tuple], drone_id: str) -> Optional[Tuple]:
        """查找指定无人机的决策指令"""
        for decision in decisions:
            if decision[0] == drone_id:
                return decision
        return None
    
    def _execute_single_maneuver(self, drone_state: Dict, strategy: str, params: any,
                               ally_states: Dict, enemy_states: Dict) -> Dict:
        """
        执行单个无人机的机动动作
        """
        # 复制当前状态
        new_state = drone_state.copy()
        
        # 根据策略类型执行不同的机动
        if strategy == ManeuverType.ATTACK.value:
            new_state = self._execute_attack_maneuver(new_state, params, enemy_states)
        elif strategy == ManeuverType.EVADE.value:
            new_state = self._execute_evasion_maneuver(new_state, enemy_states)
        elif strategy == ManeuverType.RETREAT.value:
            new_state = self._execute_retreat_maneuver(new_state, enemy_states)
        elif strategy == ManeuverType.FLANK.value:
            new_state = self._execute_flank_maneuver(new_state, params, enemy_states)
        elif strategy == ManeuverType.CLIMB.value:
            new_state = self._execute_climb_maneuver(new_state)
        elif strategy == ManeuverType.DIVE.value:
            new_state = self._execute_dive_maneuver(new_state)
        elif strategy == ManeuverType.HOLD.value:
            new_state = self._execute_hold_maneuver(new_state, ally_states)
        else:
            # 未知策略，保持当前状态
            new_state = self._maintain_current_state(new_state)
        
        # 更新物理状态
        new_state = self._update_physics(new_state)
        
        # 消耗能量
        new_state = self._consume_energy(new_state, strategy)
        
        return new_state
    
    def _execute_attack_maneuver(self, drone_state: Dict, target_id: str, 
                                enemy_states: Dict) -> Dict:
        """执行攻击机动"""
        if not target_id or target_id not in enemy_states:
            return self._maintain_current_state(drone_state)
        
        target_state = enemy_states[target_id]
        target_pos = np.array(target_state['position'])
        current_pos = np.array(drone_state['position'])
        
        # 计算攻击向量
        attack_vector = target_pos - current_pos
        distance = np.linalg.norm(attack_vector)
        
        if distance > 0:
            # 归一化攻击向量
            attack_direction = attack_vector / distance
            
            # 计算攻击速度 (根据距离调整)
            if distance < self.config.combat_radius:
                # 近距离：降低速度，提高机动性
                speed_factor = 0.7
            else:
                # 远距离：全速接近
                speed_factor = 1.0
            
            target_velocity = attack_direction * self.config.max_speed * speed_factor
            
            # 高度调整：尝试获得高度优势
            if target_pos[2] > current_pos[2]:
                # 目标在上方，爬升
                target_velocity[2] = self.config.max_climb_rate * 0.5
            
            drone_state['velocity'] = target_velocity
            drone_state['strategy'] = 'attack'
            drone_state['target'] = target_id
        
        return drone_state
    
    def _execute_evasion_maneuver(self, drone_state: Dict, enemy_states: Dict) -> Dict:
        """执行规避机动"""
        current_pos = np.array(drone_state['position'])
        evasion_vector = np.zeros(3)
        
        # 计算所有敌方威胁的合成规避向量
        for enemy_id, enemy_state in enemy_states.items():
            enemy_pos = np.array(enemy_state['position'])
            threat_vector = current_pos - enemy_pos
            distance = np.linalg.norm(threat_vector)
            
            if distance < self.config.evasion_radius and distance > 0:
                # 威胁权重与距离成反比
                threat_weight = 1.0 / (distance + 1.0)
                normalized_threat = threat_vector / distance
                evasion_vector += normalized_threat * threat_weight
        
        # 如果存在威胁，执行规避
        if np.linalg.norm(evasion_vector) > 0:
            evasion_vector = evasion_vector / np.linalg.norm(evasion_vector)
            
            # 添加随机扰动提高不可预测性
            random_factor = np.random.normal(0, 0.2, 3)
            evasion_vector += random_factor
            evasion_vector = evasion_vector / np.linalg.norm(evasion_vector)
            
            # 设置规避速度
            drone_state['velocity'] = evasion_vector * self.config.max_speed * 0.8
            
            # 高度机动：随机选择爬升或俯冲
            if np.random.random() > 0.5:
                drone_state['velocity'][2] = self.config.max_climb_rate * 0.6
            else:
                drone_state['velocity'][2] = -self.config.max_dive_rate * 0.4
        
        drone_state['strategy'] = 'evade'
        return drone_state
    
    def _execute_retreat_maneuver(self, drone_state: Dict, enemy_states: Dict) -> Dict:
        """执行撤退机动"""
        current_pos = np.array(drone_state['position'])
        retreat_vector = np.zeros(3)
        
        # 计算远离所有敌方的撤退向量
        for enemy_id, enemy_state in enemy_states.items():
            enemy_pos = np.array(enemy_state['position'])
            retreat_direction = current_pos - enemy_pos
            distance = np.linalg.norm(retreat_direction)
            
            if distance > 0:
                retreat_vector += retreat_direction / distance
        
        if np.linalg.norm(retreat_vector) > 0:
            retreat_vector = retreat_vector / np.linalg.norm(retreat_vector)
            
            # 撤退速度
            drone_state['velocity'] = retreat_vector * self.config.max_speed
            
            # 撤退时优先爬升获得高度优势
            drone_state['velocity'][2] = self.config.max_climb_rate * 0.8
        
        drone_state['strategy'] = 'retreat'
        return drone_state
    
    def _execute_flank_maneuver(self, drone_state: Dict, target_id: str, 
                               enemy_states: Dict) -> Dict:
        """执行侧翼机动"""
        if not target_id or target_id not in enemy_states:
            return self._maintain_current_state(drone_state)
        
        target_state = enemy_states[target_id]
        target_pos = np.array(target_state['position'])
        current_pos = np.array(drone_state['position'])
        
        # 计算侧翼向量
        to_target = target_pos - current_pos
        distance = np.linalg.norm(to_target)
        
        if distance > 0:
            # 创建垂直于目标方向的侧翼向量
            # 使用叉积创建垂直向量
            up_vector = np.array([0, 0, 1])
            flank_vector = np.cross(to_target, up_vector)
            
            if np.linalg.norm(flank_vector) > 0:
                flank_vector = flank_vector / np.linalg.norm(flank_vector)
                
                # 随机选择左翼或右翼
                if np.random.random() > 0.5:
                    flank_vector = -flank_vector
                
                # 设置侧翼速度
                drone_state['velocity'] = flank_vector * self.config.max_speed * 0.8
                
                # 保持与目标相似的高度
                height_diff = target_pos[2] - current_pos[2]
                if abs(height_diff) > 50:
                    drone_state['velocity'][2] = np.sign(height_diff) * self.config.max_climb_rate * 0.3
        
        drone_state['strategy'] = 'flank'
        drone_state['target'] = target_id
        return drone_state
    
    def _execute_climb_maneuver(self, drone_state: Dict) -> Dict:
        """执行爬升机动"""
        current_velocity = np.array(drone_state['velocity'])
        
        # 保持水平方向的速度，增加垂直爬升
        current_velocity[2] = self.config.max_climb_rate
        
        # 适当减少水平速度以保持能量
        horizontal_velocity = current_velocity[:2]
        horizontal_speed = np.linalg.norm(horizontal_velocity)
        
        if horizontal_speed > 0:
            # 减少水平速度
            current_velocity[:2] = horizontal_velocity * 0.7
        
        drone_state['velocity'] = current_velocity
        drone_state['strategy'] = 'climb'
        return drone_state
    
    def _execute_dive_maneuver(self, drone_state: Dict) -> Dict:
        """执行俯冲机动"""
        current_velocity = np.array(drone_state['velocity'])
        
        # 设置俯冲速度
        current_velocity[2] = -self.config.max_dive_rate
        
        # 俯冲时可以增加水平速度
        horizontal_velocity = current_velocity[:2]
        horizontal_speed = np.linalg.norm(horizontal_velocity)
        
        if horizontal_speed > 0:
            # 增加水平速度
            current_velocity[:2] = horizontal_velocity * 1.2
        
        drone_state['velocity'] = current_velocity
        drone_state['strategy'] = 'dive'
        return drone_state
    
    def _execute_hold_maneuver(self, drone_state: Dict, ally_states: Dict) -> Dict:
        """执行保持位置机动"""
        current_pos = np.array(drone_state['position'])
        
        # 计算编队中心
        ally_positions = [np.array(ally['position']) for ally in ally_states.values()]
        if ally_positions:
            formation_center = np.mean(ally_positions, axis=0)
            
            # 向编队中心移动
            to_center = formation_center - current_pos
            distance_to_center = np.linalg.norm(to_center)
            
            if distance_to_center > self.config.formation_distance:
                # 如果距离编队中心太远，向中心移动
                to_center_normalized = to_center / distance_to_center
                drone_state['velocity'] = to_center_normalized * self.config.max_speed * 0.5
            else:
                # 如果在编队范围内，保持相对静止
                drone_state['velocity'] = np.array([0, 0, 0])
        else:
            # 如果没有友军，保持当前位置
            drone_state['velocity'] = np.array([0, 0, 0])
        
        drone_state['strategy'] = 'hold'
        return drone_state
    
    def _maintain_current_state(self, drone_state: Dict) -> Dict:
        """维持当前状态"""
        # 逐渐减速
        current_velocity = np.array(drone_state['velocity'])
        drone_state['velocity'] = current_velocity * 0.9
        return drone_state
    
    def _update_physics(self, drone_state: Dict) -> Dict:
        """更新物理状态"""
        # 更新位置
        position = np.array(drone_state['position'])
        velocity = np.array(drone_state['velocity'])
        
        # 限制速度
        speed = np.linalg.norm(velocity)
        if speed > self.config.max_speed:
            velocity = velocity / speed * self.config.max_speed
            drone_state['velocity'] = velocity
        
        # 更新位置
        new_position = position + velocity * self.dt
        
        # 限制高度
        new_position[2] = np.clip(new_position[2], 
                                 self.config.min_altitude, 
                                 self.config.max_altitude)
        
        drone_state['position'] = new_position
        drone_state['altitude'] = new_position[2]
        
        return drone_state
    
    def _consume_energy(self, drone_state: Dict, strategy: str) -> Dict:
        """消耗能量"""
        current_energy = drone_state.get('energy', 1.0)
        
        # 不同策略的能量消耗率不同
        consumption_rates = {
            'attack': 1.5,
            'evade': 2.0,
            'retreat': 1.8,
            'flank': 1.3,
            'climb': 1.6,
            'dive': 1.2,
            'hold': 0.8
        }
        
        consumption_rate = consumption_rates.get(strategy, 1.0)
        energy_consumed = self.config.energy_consumption_rate * consumption_rate * self.dt
        
        new_energy = max(0.0, current_energy - energy_consumed)
        drone_state['energy'] = new_energy
        
        return drone_state
    
    def get_maneuver_status(self, drone_states: Dict) -> Dict:
        """获取机动状态报告"""
        status_report = {}
        
        for drone_id, state in drone_states.items():
            position = np.array(state['position'])
            velocity = np.array(state['velocity'])
            speed = np.linalg.norm(velocity)
            
            status_report[drone_id] = {
                'position': position.tolist(),
                'velocity': velocity.tolist(),
                'speed': speed,
                'altitude': state['altitude'],
                'energy': state['energy'],
                'strategy': state.get('strategy', 'unknown'),
                'target': state.get('target', None)
            }
        
        return status_report

# ====================== 机动系统集成接口 ======================
class ManeuverIntegrator:
    """机动系统集成器 - 连接决策系统和机动模型"""
    
    def __init__(self, maneuver_config: ManeuverConfig = None):
        self.maneuvering = AircraftManeuvering(maneuver_config)
        self.simulation_time = 0.0
        self.time_step = 0.1
        
    def integrate_with_combat_system(self, combat_system, simulation_duration: float = 10.0):
        """
        与作战系统集成，运行完整的决策-机动循环
        
        Args:
            combat_system: DroneCombatSystem实例
            simulation_duration: 仿真持续时间(秒)
        """
        print(f"\n=== 开始决策-机动集成仿真 (持续时间: {simulation_duration}s) ===")
        
        while self.simulation_time < simulation_duration:
            print(f"\n--- 仿真时间: {self.simulation_time:.1f}s ---")
            
            # 1. 获取决策指令
            decisions = combat_system.get_strategy_decisions()
            
            if not decisions:
                print("没有有效决策，仿真结束")
                break
            
            # 2. 执行机动
            blue_states = combat_system.environment.blue_drones
            red_states = combat_system.environment.red_drones
            
            updated_blue_states = self.maneuvering.execute_maneuvers(
                decisions, blue_states, red_states, self.time_step
            )
            
            # 3. 更新环境状态
            combat_system.environment.blue_drones = updated_blue_states
            
            # 4. 显示状态报告
            status_report = self.maneuvering.get_maneuver_status(updated_blue_states)
            self._print_status_report(status_report)
            
            # 5. 更新时间
            self.simulation_time += self.time_step
        
        print(f"\n=== 仿真完成，总用时: {self.simulation_time:.1f}s ===")
    
    def _print_status_report(self, status_report: Dict):
        """打印状态报告"""
        print("\n--- 机动状态报告 ---")
        for drone_id, status in status_report.items():
            pos = status['position']
            print(f"{drone_id}: 位置({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) "
                  f"速度:{status['speed']:.1f}m/s 能量:{status['energy']:.2f} "
                  f"策略:{status['strategy']}")