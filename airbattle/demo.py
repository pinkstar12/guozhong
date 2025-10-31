"""
分层强化学习多弹协同制导系统主程序
基于原有DroneCombatSystem，集成分层强化学习决策模块
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np

if __package__ is None or __package__ == "":
    package_root = os.path.dirname(__file__)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from DroneCombatSystem import DroneCombatSystem  # type: ignore
    from AircraftManeuvering import AircraftManeuvering, ManeuverConfig  # type: ignore
    from Hierarchical import (  # type: ignore
        HierarchicalRLConfig,
        HierarchicalDecisionSystem,
        HierarchicalManeuverIntegrator,
        ExpertKnowledgeSystem,
        TaskType,
    )
else:
    from .DroneCombatSystem import DroneCombatSystem
    from .AircraftManeuvering import AircraftManeuvering, ManeuverConfig
    from .Hierarchical import (
        HierarchicalRLConfig,
        HierarchicalDecisionSystem,
        HierarchicalManeuverIntegrator,
        ExpertKnowledgeSystem,
        TaskType
    )

class MultiMissileCooperativeGuidanceSystem:
    """多弹协同制导系统主控制器"""
    
    def __init__(self):
        print("=== 初始化多弹协同制导系统 ===")
        
        # 1. 创建基础作战环境
        self.combat_system = DroneCombatSystem(red_invasion_mode='concentrated')
        
        # 2. 配置机动参数
        self.maneuver_config = ManeuverConfig(
            max_speed=100.0,
            max_acceleration=20.0,
            max_climb_rate=25.0,
            max_dive_rate=30.0,
            max_turn_rate=4.0,
            combat_radius=180.0,
            evasion_radius=280.0,
            formation_distance=120.0
        )
        
        # 3. 配置分层强化学习参数
        self.rl_config = HierarchicalRLConfig(
            high_level_lr=0.001,
            low_level_lr=0.0008,
            high_level_epsilon=0.15,
            low_level_epsilon=0.25,
            high_level_gamma=0.95,
            low_level_gamma=0.9,
            state_dim=32,
            action_dim=16,
            max_cluster_size=6,
            leader_selection_threshold=0.65
        )
        
        # 4. 创建分层集成系统
        self.hierarchical_integrator = HierarchicalManeuverIntegrator(
            self.combat_system, 
            self.maneuver_config, 
            self.rl_config
        )
        
        # 5. 系统状态记录
        self.mission_history = []
        self.performance_metrics = {
            'total_missions': 0,
            'successful_missions': 0,
            'average_survival_rate': 0.0,
            'average_mission_time': 0.0,
            'leader_effectiveness': {}
        }
        
        print("多弹协同制导系统初始化完成")
    
    def run_training_missions(self, num_missions: int = 8, mission_duration: float = 25.0):
        """运行训练任务"""
        print(f"\n=== 开始执行 {num_missions} 个训练任务 ===")
        
        for mission_id in range(1, num_missions + 1):
            print(f"\n{'='*60}")
            print(f"任务 {mission_id}/{num_missions}: 多弹协同攻击训练")
            print(f"{'='*60}")
            
            # 重置任务环境
            self._reset_mission_environment()
            
            # 记录任务开始状态
            initial_blue_count = len(self.combat_system.environment.blue_drones)
            initial_red_count = len(self.combat_system.environment.red_drones)
            
            print(f"任务态势: 蓝方{initial_blue_count}弹 vs 红方{initial_red_count}弹")
            
            # 执行分层强化学习仿真
            mission_start_time = 0.0
            mission_result = self._execute_single_mission(mission_duration, mission_id)
            
            # 记录任务结果
            self._record_mission_result(mission_id, mission_result, initial_blue_count, initial_red_count)
            
            # 显示任务总结
            self._print_mission_summary(mission_id, mission_result)
        
        # 显示整体训练结果
        self._print_training_summary()
    
    def _execute_single_mission(self, duration: float, mission_id: int) -> Dict:
        """执行单个任务"""
        simulation_time = 0.0
        time_step = 0.1
        step_count = 0
        
        # 任务状态记录
        mission_log = {
            'decisions_made': 0,
            'maneuvers_executed': 0,
            'leader_changes': 0,
            'collective_rewards': [],
            'blue_casualties': 0,
            'red_casualties': 0
        }
        
        print(f"\n--- 任务 {mission_id} 开始执行 ---")
        
        while simulation_time < duration:
            step_count += 1
            
            # 记录状态变化前
            blue_states_before = self.combat_system.environment.blue_drones.copy()
            red_states_before = self.combat_system.environment.red_drones.copy()
            
            # 检查终止条件
            if not blue_states_before:
                print("任务失败: 蓝方全部损失")
                break
            if not red_states_before:
                print("任务成功: 红方全部消灭")
                break
            
            # 1. 分层决策
            try:
                hierarchical_decisions = self.hierarchical_integrator.hierarchical_decision_system.get_hierarchical_decisions()
                mission_log['decisions_made'] += 1
            except Exception as e:
                print(f"决策异常: {e}")
                hierarchical_decisions = set()
            
            if not hierarchical_decisions:
                print(f"步骤 {step_count}: 无有效决策，任务终止")
                break
            
            # 2. 执行机动
            try:
                updated_blue_states = self.hierarchical_integrator.maneuvering.execute_maneuvers(
                    hierarchical_decisions, 
                    blue_states_before, 
                    red_states_before, 
                    time_step
                )
                self.combat_system.environment.blue_drones = updated_blue_states
                mission_log['maneuvers_executed'] += 1
            except Exception as e:
                print(f"机动执行异常: {e}")
                updated_blue_states = blue_states_before
            
            # 3. 模拟敌方简单响应 (简化处理)
            self._simulate_enemy_response(red_states_before, updated_blue_states, time_step)
            
            # 4. 训练与奖励更新
            if step_count % 5 == 0:
                try:
                    self.hierarchical_integrator.hierarchical_decision_system.update_rewards_and_train(
                        blue_states_before, updated_blue_states
                    )
                except Exception as e:
                    print(f"训练更新异常: {e}")
            
            # 5. 状态监控
            if step_count % 20 == 0:
                self._print_mission_progress(mission_id, simulation_time, updated_blue_states, step_count)
            
            # 记录伤亡
            mission_log['blue_casualties'] += len(blue_states_before) - len(updated_blue_states)
            mission_log['red_casualties'] += len(red_states_before) - len(self.combat_system.environment.red_drones)
            
            simulation_time += time_step
        
        # 任务结束
        final_blue_count = len(self.combat_system.environment.blue_drones)
        final_red_count = len(self.combat_system.environment.red_drones)
        
        mission_result = {
            'duration': simulation_time,
            'steps': step_count,
            'blue_survivors': final_blue_count,
            'red_survivors': final_red_count,
            'mission_success': final_red_count == 0 and final_blue_count > 0,
            'log': mission_log
        }
        
        return mission_result
    
    def _simulate_enemy_response(self, red_states: Dict, blue_states: Dict, dt: float):
        """简化的敌方响应模拟"""
        if not red_states or not blue_states:
            return
        
        # 简单的敌方AI: 接近并攻击最近的蓝方单位
        for red_id, red_state in red_states.items():
            red_pos = np.array(red_state['position'])
            
            # 找到最近的蓝方单位
            nearest_blue = None
            min_distance = float('inf')
            
            for blue_id, blue_state in blue_states.items():
                blue_pos = np.array(blue_state['position'])
                distance = np.linalg.norm(red_pos - blue_pos)
                if distance < min_distance:
                    min_distance = distance
                    nearest_blue = (blue_id, blue_pos)
            
            if nearest_blue:
                # 向最近蓝方移动
                direction = nearest_blue[1] - red_pos
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    speed = 60.0  # 敌方速度
                    red_state['velocity'] = direction * speed
                    red_state['position'] = red_pos + direction * speed * dt
                
                # 简化的攻击判定
                if min_distance < 100:  # 攻击距离
                    blue_id = nearest_blue[0]
                    if blue_id in blue_states:
                        # 减少蓝方健康度
                        blue_states[blue_id]['health'] -= 0.1 * dt
                        if blue_states[blue_id]['health'] <= 0:
                            del blue_states[blue_id]  # 摧毁蓝方单位
        
        # 更新红方状态
        self.combat_system.environment.red_drones = red_states
    
    def _reset_mission_environment(self):
        """重置任务环境"""
        # 重新生成随机初始态势
        self.combat_system.environment.blue_drones = self.combat_system.environment._init_drones(5, 'blue')
        self.combat_system.environment.red_drones = self.combat_system.environment._init_drones(
            np.random.randint(3, 6), 'red'  # 红方数量随机化
        )
    
    def _print_mission_progress(self, mission_id: int, sim_time: float, blue_states: Dict, step: int):
        """打印任务进度"""
        red_count = len(self.combat_system.environment.red_drones)
        blue_count = len(blue_states)
        
        if blue_count > 0:
            avg_energy = sum(s['energy'] for s in blue_states.values()) / blue_count
            avg_health = sum(s['health'] for s in blue_states.values()) / blue_count
        else:
            avg_energy = avg_health = 0.0
        
        # 获取训练统计
        training_stats = self.hierarchical_integrator.hierarchical_decision_system.get_training_statistics()
        
        print(f"任务{mission_id} | 时间:{sim_time:.1f}s | 步骤:{step} | "
              f"蓝方:{blue_count} | 红方:{red_count} | "
              f"平均能量:{avg_energy:.2f} | 平均健康:{avg_health:.2f} | "
              f"奖励:{training_stats['avg_reward']:.3f}")
    
    def _record_mission_result(self, mission_id: int, result: Dict, initial_blue: int, initial_red: int):
        """记录任务结果"""
        self.performance_metrics['total_missions'] += 1
        
        if result['mission_success']:
            self.performance_metrics['successful_missions'] += 1
        
        # 更新平均生存率
        survival_rate = result['blue_survivors'] / initial_blue
        current_avg = self.performance_metrics['average_survival_rate']
        total_missions = self.performance_metrics['total_missions']
        self.performance_metrics['average_survival_rate'] = (
            (current_avg * (total_missions - 1) + survival_rate) / total_missions
        )
        
        # 更新平均任务时间
        current_time_avg = self.performance_metrics['average_mission_time']
        self.performance_metrics['average_mission_time'] = (
            (current_time_avg * (total_missions - 1) + result['duration']) / total_missions
        )
        
        # 记录详细历史
        self.mission_history.append({
            'mission_id': mission_id,
            'result': result,
            'initial_forces': {'blue': initial_blue, 'red': initial_red}
        })
    
    def _print_mission_summary(self, mission_id: int, result: Dict):
        """打印任务总结"""
        print(f"\n--- 任务 {mission_id} 结果总结 ---")
        
        status = "成功" if result['mission_success'] else "失败"
        print(f"任务状态: {status}")
        print(f"持续时间: {result['duration']:.1f}秒")
        print(f"执行步数: {result['steps']}")
        print(f"蓝方存活: {result['blue_survivors']}")
        print(f"红方存活: {result['red_survivors']}")
        
        log = result['log']
        print(f"决策次数: {log['decisions_made']}")
        print(f"机动次数: {log['maneuvers_executed']}")
        print(f"蓝方损失: {log['blue_casualties']}")
        print(f"红方损失: {log['red_casualties']}")
    
    def _print_training_summary(self):
        """打印训练总结"""
        print(f"\n{'='*60}")
        print("训练总结报告")
        print(f"{'='*60}")
        
        metrics = self.performance_metrics
        print(f"总任务数: {metrics['total_missions']}")
        print(f"成功任务数: {metrics['successful_missions']}")
        
        if metrics['total_missions'] > 0:
            success_rate = metrics['successful_missions'] / metrics['total_missions'] * 100
            print(f"任务成功率: {success_rate:.1f}%")
        
        print(f"平均生存率: {metrics['average_survival_rate']:.2f}")
        print(f"平均任务时间: {metrics['average_mission_time']:.1f}秒")
        
        # 分层RL训练统计
        training_stats = self.hierarchical_integrator.hierarchical_decision_system.get_training_statistics()
        print(f"\n--- 分层强化学习训练统计 ---")
        print(f"总训练步数: {training_stats['training_steps']}")
        print(f"高层经验池: {training_stats['buffer_sizes']['high_level']}")
        print(f"低层经验池: {training_stats['buffer_sizes']['low_level']}")
        print(f"最终平均奖励: {training_stats['avg_reward']:.3f}")
        
        if training_stats['high_level_losses']:
            print(f"高层网络平均损失: {np.mean(training_stats['high_level_losses']):.4f}")
        if training_stats['low_level_losses']:
            print(f"低层网络平均损失: {np.mean(training_stats['low_level_losses']):.4f}")
    
    def demonstrate_single_decision_cycle(self):
        """演示单次决策循环"""
        print(f"\n{'='*60}")
        print("单次分层决策循环演示")
        print(f"{'='*60}")
        
        # 重置环境
        self._reset_mission_environment()
        
        # 获取当前态势
        blue_obs = self.combat_system.environment.get_all_observations('blue')
        print(f"当前态势: 蓝方{len(blue_obs)}架, 红方{len(self.combat_system.environment.red_drones)}架")
        
        # 1. 领导者选择
        blue_states = self.combat_system.environment.blue_drones
        leadership = self.hierarchical_integrator.hierarchical_decision_system.rl_agent.select_leaders(blue_states)
        
        print("\n--- 领导者选择结果 ---")
        for drone_id, role in leadership.items():
            capability = self.hierarchical_integrator.hierarchical_decision_system.expert_system.evaluate_leader_capability(
                blue_states[drone_id]
            )
            print(f"{drone_id}: {role.value} (能力评分: {capability:.3f})")
        
        # 2. 分层决策
        decisions = self.hierarchical_integrator.hierarchical_decision_system.get_hierarchical_decisions()
        
        print(f"\n--- 分层决策结果 ---")
        for decision in decisions:
            drone_id, strategy, params = decision
            role = leadership.get(drone_id, "未知")
            param_str = f"目标:{params}" if params else "无参数"
            print(f"{role.value} {drone_id}: {strategy} ({param_str})")
        
        # 3. 专家经验验证
        print(f"\n--- 专家经验验证 ---")
        expert_system = self.hierarchical_integrator.hierarchical_decision_system.expert_system
        
        for drone_id, obs in blue_obs.items():
            if obs.get('enemies'):
                expert_probs = expert_system.get_expert_action_probability(
                    obs['self_state'], TaskType.TARGET_ELIMINATION
                )
                action_names = ['attack', 'evade', 'hold', 'retreat', 'flank', 'climb', 'dive']
                
                print(f"{drone_id} 专家建议概率分布:")
                for i, (action, prob) in enumerate(zip(action_names, expert_probs)):
                    print(f"  {action}: {prob:.3f}")
        
        print(f"\n演示完成")

def run_comprehensive_demo():
    """运行综合演示"""
    print("多弹协同制导分层强化学习系统")
    print("基于动态贝叶斯网络决策与专家经验融合")

    # 创建系统
    guidance_system = MultiMissileCooperativeGuidanceSystem()

    # 1. 单次决策演示
    guidance_system.demonstrate_single_decision_cycle()

    # 2. 训练任务演示
    print(f"\n是否继续执行完整训练? (系统将运行多个任务进行强化学习训练)")

    # 运行简化训练演示 (3个任务，较短时间)
    print("开始执行简化训练演示...")
    guidance_system.run_training_missions(num_missions=3, mission_duration=15.0)


def run_quick_demo():
    """快速演示 - 仅展示核心功能"""
    print("=== 快速功能演示 ===")

    guidance_system = MultiMissileCooperativeGuidanceSystem()

    # 仅演示决策过程
    guidance_system.demonstrate_single_decision_cycle()

    print("\n=== 系统核心特性 ===")
    print("✓ 分层决策架构: 领导者负责任务分配，跟随者执行具体动作")
    print("✓ 强化学习训练: 基于奖励反馈持续优化决策策略")
    print("✓ 专家经验融合: 结合领域专家知识指导学习过程")
    print("✓ 动态领导选择: 根据能力评估自适应选择集群领导者")
    print("✓ 多任务协同: 支持目标消灭、区域扫描、编队防护等任务")
    print("✓ 实时态势感知: 通过图神经网络进行信息融合")


def run_full_training(num_missions: int = 8, mission_duration: float = 25.0):
    """完整训练流程"""
    guidance_system = MultiMissileCooperativeGuidanceSystem()
    guidance_system.run_training_missions(num_missions=num_missions, mission_duration=mission_duration)


def _print_mode_menu():
    print("多弹协同制导分层强化学习系统")
    print("=" * 50)
    print("选择运行模式:")
    print("1. 快速演示 (仅展示核心功能)")
    print("2. 综合演示 (包含训练过程)")
    print("3. 完整训练 (可自定义任务数量与时长)")


def _run_mode(mode: str, missions: int, duration: float):
    if mode == "quick":
        run_quick_demo()
    elif mode == "comprehensive":
        run_comprehensive_demo()
    elif mode == "full":
        run_full_training(num_missions=missions, mission_duration=duration)
    else:
        raise ValueError(f"未知运行模式: {mode}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="多弹协同制导系统演示脚本")
    parser.add_argument("--mode", choices=["quick", "comprehensive", "full", "prompt"], default="prompt", help="选择运行模式，默认为提示用户交互选择")
    parser.add_argument("--missions", type=int, default=8, help="完整训练时的任务数量")
    parser.add_argument("--duration", type=float, default=25.0, help="每个任务的持续时间")
    args = parser.parse_args(argv)

    selected_mode = args.mode
    if selected_mode == "prompt":
        if sys.stdin.isatty():
            _print_mode_menu()
            try:
                choice = input("\n请输入选择 (1/2/3): ").strip()
            except EOFError:
                choice = ""
            mapping = {"1": "quick", "2": "comprehensive", "3": "full"}
            selected_mode = mapping.get(choice, "quick")
            if choice not in mapping:
                print("默认运行快速演示...")
        else:
            print("检测到非交互式环境，自动运行快速演示")
            selected_mode = "quick"

    try:
        _run_mode(selected_mode, args.missions, args.duration)
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as exc:
        print(f"\n程序执行异常: {exc}")
        print("运行快速演示作为备选...")
        run_quick_demo()

    print("\n程序执行完成")


if __name__ == "__main__":
    main()
