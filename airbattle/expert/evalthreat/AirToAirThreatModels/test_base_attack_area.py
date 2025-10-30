import numpy as np
from BaseAttackArea import BaseAttackAreaThreatAnalysis

def main():
    # 使用论文中的参数初始化
    our_params = {
        'angle_params': {'theta_Rmax': 80, 'theta_Max': 50, 'theta_Maxmin': 30},
        'distance_params': {'D_Rmax': 120, 'D_Mmax': 60, 'D_Mmin': 5, 'D_NZmax': 30, 'D_NZmin': 5}
    }
    enemy_params = {
        'angle_params': {'phi_Rmax': 70, 'phi_Max': 40, 'phi_Maxmin': 20},
        'distance_params': {'D_Rmax': 100, 'D_Mmax': 40, 'D_Mmin': 5, 'D_NZmax': 20, 'D_NZmin': 2}
    }
    speed_params = {'D_high': 80, 'V_high': 350, 'D_low': 30, 'V_low': 250, 'V_base': 300}
    
    analyzer = BaseAttackAreaThreatAnalysis(
        our_params=our_params,
        enemy_params=enemy_params,
        speed_params=speed_params
    )
    
    print("="*50)
    print("角度态势函数测试（修正坐标系）")
    print("="*50)
    
    # 角度态势测试 - 使用军事坐标系
    test_cases = [
        ("绝对优势(正北敌机,敌机西向)", 
         0,    # 我机航向0°(北)
         np.array([0, 0]), 
         np.array([1, 0]),  # 敌机正北1km
         270   # 敌机航向270°(西)
        ),
        
        ("绝对优势(正东敌机,敌机南向)", 
         90,   # 我机航向90°(东)
         np.array([0, 0]), 
         np.array([0, 1]),  # 敌机正东1km
         180   # 敌机航向180°(南)
        ),
        
        ("优劣相等(正东敌机,敌机东向)", 
         0,    # 我机航向0°(北)
         np.array([0, 0]), 
         np.array([0, 1]),  # 敌机正东1km
         90    # 敌机航向90°(东)
        ),
        
        ("明显劣势(西南敌机,敌机北向)", 
         0,    # 我机航向0°(北)
         np.array([0, 0]), 
         np.array([-1, -1]),  # 敌机西南1.41km
         0     # 敌机航向0°(北)
        ),
    ]
    
    for name, heading, our_pos, enemy_pos, enemy_heading in test_cases:
        sa = analyzer._angle_situation(
            heading=heading,
            our_pos=our_pos,
            enemy={'position': enemy_pos, 'heading': enemy_heading}
        )
        print(f"{name}: SA = {sa:.4f}")
    
    print("\n" + "="*50)
    print("距离态势函数测试")
    print("="*50)
    
    # 距离态势测试
    test_cases = [
        ("远距离优势(D=90km,H=6000m)", np.array([0,0]), 6000, np.array([90,0])),
        ("中距离优势(D=50km,H=6000m)", np.array([0,0]), 6000, np.array([50,0])),
        ("近距离劣势(D=10km,H=6000m)", np.array([0,0]), 6000, np.array([10,0])),
        ("极近距离(D=4km,H=6000m)", np.array([0,0]), 6000, np.array([4,0])),
        ("远距离高度影响(D=90km,H=11000m)", np.array([0,0]), 11000, np.array([90,0])),
    ]
    
    for name, our_pos, height, enemy_pos in test_cases:
        our_aircraft = {'position': our_pos, 'height': height}
        sd = analyzer._distance_situation(our_aircraft, {'position': enemy_pos})
        print(f"{name}: SD = {sd:.4f}")
    
    print("\n" + "="*50)
    print("速度态势函数测试")
    print("="*50)
    
    # 速度态势测试
    test_cases = [
        ("最佳速度(V_w=300m/s,D=50km)", 300, 200, np.array([0,0]), np.array([50,0])),
        ("速度过低(V_w=150m/s,D=50km)", 150, 200, np.array([0,0]), np.array([50,0])),
        ("速度过高(V_w=400m/s,D=50km)", 400, 200, np.array([0,0]), np.array([50,0])),
        ("远距离最佳速度(V_w=350m/s,D=100km)", 350, 200, np.array([0,0]), np.array([100,0])),
        ("近距离最佳速度(V_w=250m/s,D=20km)", 250, 200, np.array([0,0]), np.array([20,0])),
    ]
    
    for name, V_W, V_M, our_pos, enemy_pos in test_cases:
        sv = analyzer._speed_situation(
            V_W, V_M, 
            our_pos, 
            {'position': enemy_pos}
        )
        print(f"{name}: SV = {sv:.4f}")
    
    print("\n" + "="*50)
    print("高度态势函数测试")
    print("="*50)
    
    # 高度态势测试
    test_cases = [
        ("最佳高度(H_w=6000m,H_m=5000m)", 6000, 5000),
        ("高度略低(H_w=5500m,H_m=5000m)", 5500, 5000),
        ("高度略高(H_w=6500m,H_m=5000m)", 6500, 5000),
        ("高度过低(H_w=3000m,H_m=5000m)", 3000, 5000),
        ("高度过高(H_w=8000m,H_m=5000m)", 8000, 5000),
        ("高度相等(H_w=5000m,H_m=5000m)", 5000, 5000),
    ]
    
    for name, H_W, H_M in test_cases:
        sh = analyzer._height_situation(H_W, H_M)
        print(f"{name}: SH = {sh:.4f}")

if __name__ == "__main__":
    main()