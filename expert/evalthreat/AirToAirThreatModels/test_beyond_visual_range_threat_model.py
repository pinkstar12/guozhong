#!/usr/bin/env python3
"""
测试超视距空战威胁评估模型的功能
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from expert.evalthreat.AirToAirThreatModels.BeyondVisualRangeThreatModel import (
    BeyondVisualRangeThreatModel, 
    BaseAttackAreaThreatAnalysis
)

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")
    
    # 创建测试数据
    our_aircraft = {
        'speed': 300,     # m/s
        'height': 8000,   # m
        'heading': 90,    # 度
        'position': np.array([0, 0])  # km
    }
    
    enemy_aircraft = {
        'speed': 280,
        'height': 7000,
        'heading': 270,
        'position': np.array([50, 10])
    }
    
    # 测试新模型
    model = BeyondVisualRangeThreatModel()
    threat_value = model.calculate_single_threat(our_aircraft, enemy_aircraft)
    components = model.calculate_situation_components(our_aircraft, enemy_aircraft)
    
    print(f"威胁评估值: {threat_value:.4f}")
    print(f"距离优势: {components['distance_advantage']:.4f}")
    print(f"角度优势: {components['angle_advantage']:.4f}")
    print(f"高度优势: {components['height_advantage']:.4f}")
    print(f"速度优势: {components['speed_advantage']:.4f}")
    print()

def test_compatibility():
    """测试兼容性包装类"""
    print("=== 测试兼容性包装类 ===")
    
    # 使用原有参数格式
    our_params = {
        'distance_params': {
            'D_Rmax': 120,
            'D_Mmax': 60,
            'D_NZmax': 30,
            'D_NZmin': 5
        },
        'angle_params': {
            'theta_Rmax': 80,
            'theta_Max': 50,
            'theta_Maxmin': 30
        }
    }
    
    # 创建兼容性包装类实例
    compat_model = BaseAttackAreaThreatAnalysis(our_params=our_params)
    
    our_aircraft = {
        'speed': 320,
        'height': 9000,
        'heading': 45,
        'position': np.array([10, 5])
    }
    
    enemy_aircraft = {
        'speed': 300,
        'height': 8000,
        'heading': 225,
        'position': np.array([30, 20])
    }
    
    # 测试兼容性接口
    threat_value = compat_model.calculate_single_threat(our_aircraft, enemy_aircraft)
    components = compat_model.calculate_situation_components(our_aircraft, enemy_aircraft)
    
    print(f"兼容性接口 - 威胁评估值: {threat_value:.4f}")
    print(f"角度态势: {components['angle_situation']:.4f}")
    print(f"距离态势: {components['distance_situation']:.4f}")
    print(f"速度态势: {components['speed_situation']:.4f}")
    print(f"高度态势: {components['height_situation']:.4f}")
    print()

def test_distance_advantage_function():
    """测试距离优势函数的分段特性"""
    print("=== 测试距离优势函数 ===")
    
    model = BeyondVisualRangeThreatModel()
    our_aircraft = {
        'speed': 300,
        'height': 8000,
        'heading': 90,
        'position': np.array([0, 0])
    }
    
    # 测试不同距离下的优势值
    test_distances = [2, 10, 25, 45, 80, 150]  # km
    
    for distance in test_distances:
        enemy_aircraft = {
            'speed': 280,
            'height': 7000,
            'heading': 270,
            'position': np.array([distance, 0])
        }
        
        R_d = model._distance_advantage(our_aircraft, enemy_aircraft)
        print(f"距离 {distance:3d} km: 距离优势 = {R_d:.4f}")
    print()

def test_angle_advantage_function():
    """测试角度优势函数"""
    print("=== 测试角度优势函数 ===")
    
    model = BeyondVisualRangeThreatModel()
    our_aircraft = {
        'speed': 300,
        'height': 8000,
        'heading': 0,  # 正北
        'position': np.array([0, 0])
    }
    
    # 测试不同角度下的优势值
    test_angles = [0, 30, 60, 90, 120, 180]  # 度
    distance = 40  # km，处于导弹攻击范围内
    
    for angle in test_angles:
        # 计算敌机位置（相对于我机的方位角）
        dx = distance * np.cos(np.radians(angle))
        dy = distance * np.sin(np.radians(angle))
        
        enemy_aircraft = {
            'speed': 280,
            'height': 7000,
            'heading': 180,  # 正南（迎头）
            'position': np.array([dx, dy])
        }
        
        R_a = model._angle_advantage(our_aircraft, enemy_aircraft)
        print(f"角度 {angle:3d}°: 角度优势 = {R_a:.4f}")
    print()

def test_height_advantage_function():
    """测试高度优势函数"""
    print("=== 测试高度优势函数 ===")
    
    model = BeyondVisualRangeThreatModel()
    our_aircraft_base = {
        'speed': 300,
        'heading': 90,
        'position': np.array([0, 0])
    }
    
    enemy_aircraft = {
        'speed': 280,
        'height': 7000,  # 固定敌机高度
        'heading': 270,
        'position': np.array([40, 0])
    }
    
    # 测试不同我机高度下的优势值
    test_heights = [5000, 6000, 7000, 8000, 10000]  # m
    
    for height in test_heights:
        our_aircraft = our_aircraft_base.copy()
        our_aircraft['height'] = height
        
        R_h = model._height_advantage(our_aircraft, enemy_aircraft)
        print(f"我机高度 {height:5d} m (敌机7000m): 高度优势 = {R_h:.4f}")
    print()

def test_speed_advantage_function():
    """测试速度优势函数"""
    print("=== 测试速度优势函数 ===")
    
    model = BeyondVisualRangeThreatModel()
    our_aircraft_base = {
        'height': 8000,
        'heading': 90,
        'position': np.array([0, 0])
    }
    
    enemy_aircraft = {
        'speed': 300,  # 固定敌机速度 300 m/s
        'height': 7000,
        'heading': 270,
        'position': np.array([40, 0])
    }
    
    # 测试不同我机速度下的优势值
    test_speeds = [180, 240, 300, 360, 450]  # m/s
    
    for speed in test_speeds:
        our_aircraft = our_aircraft_base.copy()
        our_aircraft['speed'] = speed
        
        R_v = model._speed_advantage(our_aircraft, enemy_aircraft)
        speed_ratio = speed / 300
        print(f"我机速度 {speed:3d} m/s (比值{speed_ratio:.2f}): 速度优势 = {R_v:.4f}")
    print()

def test_integration_with_multiair_system():
    """测试与多机系统的集成"""
    print("=== 测试与多机系统的集成 ===")
    
    # 导入多机系统
    from expert.evalthreat.Mutilair_to_Mutilair import ThreatModel
    
    # 创建测试状态
    state = {
        'our_aircrafts': [
            {
                'speed': 300,
                'height': 8000,
                'heading': 90,
                'position': np.array([0, 0])
            },
            {
                'speed': 320,
                'height': 7500,
                'heading': 85,
                'position': np.array([5, 1])
            }
        ],
        'enemies': [
            {
                'speed': 280,
                'height': 7000,
                'heading': 270,
                'position': np.array([50, 10])
            },
            {
                'speed': 290,
                'height': 8500,
                'heading': 260,
                'position': np.array([60, -5])
            }
        ]
    }
    
    # 创建威胁模型并处理状态
    threat_model = ThreatModel()
    team_advantage, team_risk = threat_model.process_state(state)
    situation = threat_model.evaluate_situation(team_advantage, team_risk)
    
    print(f"团队优势值: {team_advantage:.4f}")
    print(f"团队风险值: {team_risk:.4f}")
    print(f"整体态势: {situation}")
    
    # 测试详细分析功能
    detailed_analysis = threat_model.get_detailed_analysis(state)
    print(f"协同分数: {detailed_analysis['team_coordination']}")
    print()

def main():
    """主测试函数"""
    print("开始测试超视距空战威胁评估模型")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_compatibility()
        test_distance_advantage_function()
        test_angle_advantage_function()
        test_height_advantage_function()
        test_speed_advantage_function()
        test_integration_with_multiair_system()
        
        print("=" * 50)
        print("所有测试完成，新的威胁评估模型运行正常！")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
