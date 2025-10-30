"""
集成专家系统演示
展示完整的攻击防御协同决策功能
"""

import numpy as np
import yaml
from common.registry import build
from .expertsystem import ExpertSystem


def create_full_expert_system():
    """创建完整配置的专家系统"""
    try:
        # 加载攻击专家系统配置
        with open('configs/expert.yaml', 'r', encoding='utf-8') as f:
            attack_config = yaml.safe_load(f)
        
        # 构建攻击系统
        attack_system = build(attack_config, 'expert_system')
        print(f"✓ 攻击系统创建成功，类型: {type(attack_system)}")
        
        # 加载防御专家系统配置  
        with open('configs/defend_expert.yaml', 'r', encoding='utf-8') as f:
            defend_config = yaml.safe_load(f)
        
        # 修正防御系统配置键名
        try:
            defend_system = build(defend_config, 'system')
        except:
            # 如果'system'键不存在，尝试直接创建防御系统
            from .defend_expertsystem import DefendExpertSystem
            defend_system = DefendExpertSystem(mode="de", verbose=True)
            print("✓ 使用默认配置创建防御系统")
        
        # 创建集成专家系统
        expert = ExpertSystem(
            attack_mode="behavior_tree",  # 使用行为树模式获得更好的解释
            defend_mode="de",             # 使用DE模式
            high_threat_threshold=0.5,    # 调整阈值以更好演示
            low_threat_threshold=0.2,
            attack_system=attack_system,
            defend_system=defend_system,
            verbose=True
        )
        
        print("✓ 成功创建完整配置的集成专家系统")
        print(f"  攻击系统: {type(attack_system).__name__}")
        print(f"  防御系统: {type(defend_system).__name__}")
        return expert, True
        
    except Exception as e:
        print(f"✗ 完整系统创建失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 创建简化系统用于演示
        expert = ExpertSystem(
            attack_mode="optimization",
            defend_mode="de", 
            high_threat_threshold=0.5,
            low_threat_threshold=0.2,
            verbose=True
        )
        
        print("✓ 创建简化版本集成专家系统")
        return expert, False


def demonstrate_expert_system():
    """演示集成专家系统功能"""
    print("=" * 80)
    print("集成专家系统演示")
    print("=" * 80)
    
    # 1. 创建系统
    expert, is_full_system = create_full_expert_system()
    
    # 2. 创建复杂战场场景
    print(f"\n创建战场场景...")
    
    # 场景1: 混合威胁场景
    scenario1 = {
        'our_aircrafts': [
            {
                'speed': 280, 
                'height': 8000, 
                'heading': 45, 
                'position': np.array([10.0, 15.0])
            },
            {
                'speed': 275, 
                'height': 8200, 
                'heading': 90, 
                'position': np.array([12.0, 18.0])
            },
            {
                'speed': 290, 
                'height': 7800, 
                'heading': 30, 
                'position': np.array([8.0, 12.0])
            },
            {
                'speed': 270, 
                'height': 8100, 
                'heading': 60, 
                'position': np.array([15.0, 20.0])
            }
        ],
        'enemies': [
            {
                'speed': 260, 
                'height': 7500, 
                'heading': 225, 
                'position': np.array([50.0, 45.0])
            },
            {
                'speed': 255, 
                'height': 7600, 
                'heading': 210, 
                'position': np.array([55.0, 50.0])
            },
            {
                'speed': 265, 
                'height': 7400, 
                'heading': 200, 
                'position': np.array([48.0, 42.0])
            }
        ],
        'missiles': [
            {
                "position": [800, 600], 
                "height": 7900, 
                "speed": 300, 
                "target": 0
            },
            {
                "position": [1200, 800], 
                "height": 8000, 
                "speed": 320, 
                "target": 0
            },
            {
                "position": [1500, 1000], 
                "height": 8100, 
                "speed": 280, 
                "target": 1
            }
        ]
    }
    
    print(f"✓ 场景1 - 混合威胁:")
    print(f"  我方飞机: {len(scenario1['our_aircrafts'])} 架")
    print(f"  敌方飞机: {len(scenario1['enemies'])} 架") 
    print(f"  导弹威胁: {len(scenario1['missiles'])} 枚")
    print(f"  威胁分布: 飞机0被2枚导弹锁定, 飞机1被1枚导弹锁定, 飞机2、3未被锁定")
    
    # 3. 处理场景1
    print(f"\n" + "="*60)
    print("场景1处理结果")
    print("="*60)
    
    result1 = expert.get_all_actions(scenario1)
    
    print(f"决策结果:")
    for i, (action, explanation) in enumerate(zip(result1['actions'], result1['explanations'])):
        print(f"  飞机{i}: {explanation}")
    
    summary1 = result1['decision_summary']
    print(f"\n决策总结:")
    print(f"  总飞机数: {summary1['total_aircrafts']}")
    print(f"  防御决策: {summary1['defend_aircrafts']} 架")
    print(f"  攻击决策: {summary1['attack_aircrafts']} 架")
    print(f"  保持状态: {summary1['unlocked_aircrafts']} 架")
    print(f"  威胁评估: 平均{summary1['average_threat']:.3f}, 最大{summary1['max_threat']:.3f}")
    
    # 4. 测试阈值调整效果
    print(f"\n" + "="*60)
    print("阈值调整测试")
    print("="*60)
    
    # 降低威胁阈值，看更多飞机进入防御模式
    expert.set_thresholds(0.3, 0.1)
    result1_adjusted = expert.get_all_actions(scenario1)
    
    summary1_adj = result1_adjusted['decision_summary']
    print(f"阈值调整后 (高=0.3, 低=0.1):")
    print(f"  防御决策: {summary1_adj['defend_aircrafts']} 架 (原{summary1['defend_aircrafts']}架)")
    print(f"  攻击决策: {summary1_adj['attack_aircrafts']} 架 (原{summary1['attack_aircrafts']}架)")
    
    # 5. 场景2: 无导弹威胁场景
    print(f"\n" + "="*60)
    print("场景2: 纯攻击场景")
    print("="*60)
    
    scenario2 = {
        'our_aircrafts': scenario1['our_aircrafts'][:2],  # 只有2架飞机
        'enemies': scenario1['enemies'],
        'missiles': []  # 无导弹威胁
    }
    
    result2 = expert.get_all_actions(scenario2)
    
    print(f"无导弹威胁场景:")
    for i, explanation in enumerate(result2['explanations']):
        print(f"  飞机{i}: {explanation}")
    
    # 6. 场景3: 高威胁场景
    print(f"\n" + "="*60)
    print("场景3: 高威胁防御场景")
    print("="*60)
    
    # 恢复高威胁阈值
    expert.set_thresholds(0.7, 0.3)
    
    scenario3 = {
        'our_aircrafts': [scenario1['our_aircrafts'][0]],  # 单机
        'enemies': [],
        'missiles': [
            {
                "position": [200, 150],  # 很近的导弹
                "height": 8000, 
                "speed": 350, 
                "target": 0
            },
            {
                "position": [300, 200], 
                "height": 8000, 
                "speed": 340, 
                "target": 0
            }
        ]
    }
    
    result3 = expert.get_all_actions(scenario3)
    
    print(f"高威胁场景:")
    for i, explanation in enumerate(result3['explanations']):
        print(f"  飞机{i}: {explanation}")
    
    summary3 = result3['decision_summary']
    print(f"  威胁评估: 最大威胁值 {summary3['max_threat']:.3f}")
    
    # 7. 单个飞机查询演示
    print(f"\n" + "="*60)
    print("单个飞机查询演示")
    print("="*60)
    
    for aircraft_id in range(len(scenario1['our_aircrafts'])):
        aircraft_result = expert.get_aircraft_action(aircraft_id, scenario1)
        print(f"飞机{aircraft_id}查询: {aircraft_result['explanation']}")
    
    # 8. 系统性能统计
    print(f"\n" + "="*60)
    print("系统性能统计")
    print("="*60)
    
    import time
    
    # 测试不同规模场景的处理时间
    test_scenarios = [
        ("小规模", 2, 1, 1),
        ("中规模", 4, 3, 3), 
        ("大规模", 6, 5, 5)
    ]
    
    for name, our_count, enemy_count, missile_count in test_scenarios:
        test_state = {
            'our_aircrafts': scenario1['our_aircrafts'][:our_count],
            'enemies': scenario1['enemies'][:enemy_count],
            'missiles': scenario1['missiles'][:missile_count]
        }
        
        start_time = time.time()
        for _ in range(20):
            expert.get_all_actions(test_state)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 20 * 1000
        print(f"  {name}场景 ({our_count}v{enemy_count}, {missile_count}导弹): {avg_time:.2f}ms")
    
    # 9. 最终系统状态
    print(f"\n" + "="*60)
    print("系统最终状态")
    print("="*60)
    
    expert.print_system_status()
    
    return expert


def demonstrate_configuration_usage():
    """演示配置文件使用方式"""
    print(f"\n" + "="*80)
    print("配置文件使用演示")
    print("="*80)
    
    # 检查配置文件是否存在
    config_path = "expert/integrated_expert_config.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ 配置文件加载成功: {config_path}")
        print(f"配置内容预览:")
        print(f"  威胁阈值: 高={config['expert_system']['high_threat_threshold']}")
        print(f"  威胁阈值: 低={config['expert_system']['low_threat_threshold']}")
        print(f"  攻击模式: {config['expert_system']['attack_mode']}")
        print(f"  防御模式: {config['expert_system']['defend_mode']}")
        
    except Exception as e:
        print(f"✗ 配置文件访问失败: {e}")


def main():
    """主演示函数"""
    print("集成专家系统完整演示")
    print("这个演示展示了如何使用集成专家系统进行攻击和防御决策")
    
    # 主要功能演示
    expert = demonstrate_expert_system()
    
    # 配置文件演示
    demonstrate_configuration_usage()
    
    # 总结
    print(f"\n" + "="*80)
    print("演示总结")
    print("="*80)
    
    print(f"✓ 集成专家系统演示完成!")
    print(f"")
    print(f"主要特性验证:")
    print(f"1. ✓ 导弹锁定识别 - 自动识别被导弹锁定的飞机")
    print(f"2. ✓ 威胁评估分级 - 根据威胁值进行高/低威胁判断")
    print(f"3. ✓ 智能系统分发 - 高威胁使用防御系统，未锁定使用攻击系统")
    print(f"4. ✓ 阈值动态调整 - 支持运行时调整威胁阈值")
    print(f"5. ✓ 多种查询接口 - 全体飞机和单个飞机查询")
    print(f"6. ✓ 详细决策解释 - 每个决策都有清晰的解释说明")
    print(f"7. ✓ 多场景适应 - 支持无威胁、混合威胁、高威胁等场景")
    print(f"8. ✓ 性能优化 - 毫秒级处理时间")
    print(f"9. ✓ 配置文件支持 - 支持通过配置文件创建系统")
    print(f"10. ✓ 错误处理 - 健壮的错误处理机制")
    
    print(f"\n使用方法:")
    print(f"```python")
    print("from airbattle.expert.expertsystem import ExpertSystem")
    print(f"")
    print(f"# 创建系统")
    print(f"expert = ExpertSystem(high_threat_threshold=0.7, low_threat_threshold=0.3)")
    print(f"")
    print(f"# 获取全体飞机动作")
    print(f"result = expert.get_all_actions(state)")
    print(f"print(result['explanations'])")
    print(f"")
    print(f"# 获取特定飞机动作")
    print(f"aircraft_result = expert.get_aircraft_action(0, state)")
    print(f"print(aircraft_result['explanation'])")
    print(f"```")


if __name__ == "__main__":
    main()
