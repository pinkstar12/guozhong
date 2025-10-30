from __future__ import annotations

import os
import sys

if __package__ is None or __package__ == "":
    package_root = os.path.dirname(__file__)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from DroneCombatSystem import DroneCombatSystem  # type: ignore
    from AircraftManeuvering import AircraftManeuvering, ManeuverConfig  # type: ignore
else:
    from .DroneCombatSystem import DroneCombatSystem
    from .AircraftManeuvering import AircraftManeuvering, ManeuverConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

# ====================== 主执行入口 (聚焦决策输出) ======================
if __name__ == "__main__":

    # 1. 创建决策系统实例
    combat_system = DroneCombatSystem(red_invasion_mode='concentrated')

    # 2. 获取当前态势下的决策指令集合
    decision_set = combat_system.get_strategy_decisions()

    # 3. 打印最终的输出结果
    print("\n============================================================")
    print("                最终决策输出集合 (机动模块接口)")
    print("============================================================")
    if not decision_set:
        print("没有可用的决策。")
    else:
        # 为了更美观的打印，我们可以对集合进行排序
        sorted_decisions = sorted(list(decision_set))

        # --- 核心修改：处理新的三元组格式 ---
        for decision_tuple in sorted_decisions:
            drone_id, strategy, params = decision_tuple

            # 使用ljust来对齐文本，使输出更整齐
            id_str = f"无人机ID: {drone_id.ljust(8)}"
            strategy_str = f"推荐策略: {strategy.ljust(10)}"

            if params:
                param_str = f"参数 (目标ID): {params}"
            else:
                param_str = "参数: 无"

            print(f"  {id_str} | {strategy_str} | {param_str}")

    print("============================================================")

    # 4. 初始化飞机机动模型
    print("\n--- 初始化飞机机动模型 ---")
    maneuver_config = ManeuverConfig(
        max_speed=80.0,
        max_climb_rate=15.0,
        combat_radius=200.0
    )
    aircraft_maneuvering = AircraftManeuvering(maneuver_config)

    # 5. 执行机动指令
    if decision_set:
        print("\n--- 执行机动指令 ---")
        # 获取当前无人机状态
        blue_states = combat_system.environment.blue_drones
        red_states = combat_system.environment.red_drones
        
        # 执行机动
        updated_states = aircraft_maneuvering.execute_maneuvers(
            decision_set, blue_states, red_states, time_step=0.1
        )
        
        # 更新环境中的无人机状态
        combat_system.environment.blue_drones = updated_states
        
        # 6. 显示机动执行结果
        print("\n--- 机动执行结果 ---")
        status_report = aircraft_maneuvering.get_maneuver_status(updated_states)
        
        for drone_id, status in status_report.items():
            pos = status['position']
            print(f"{drone_id}: 位置({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) "
                  f"速度:{status['speed']:.1f}m/s 策略:{status['strategy']} "
                  f"能量:{status['energy']:.2f}")
    
    print("\n--- 程序执行完成 ---")