import numpy as np

class BaseAttackAreaThreatAnalysis:

    """
    一对一态势分析模块
    负责计算单架我机对单架敌机的威胁态势评估
    """
    def __init__(self, our_params=None, enemy_params=None, speed_params=None, **kwargs):
        """
        初始化单对单态势分析模块
        :param our_params: 我方参数配置
        :param enemy_params: 敌方参数配置
        :param speed_params: 速度参数配置
        """
        # 整理配置
        config = {
            'our_params': our_params or {},
            'enemy_params': enemy_params or {},
            'speed_params': speed_params or {},
        }
        self.initialize(config)

    def initialize(self, config):
        """
        从配置初始化态势评估模块参数
        :param config: 配置字典，包含角度、距离、速度参数
        """
        # our_params
        our_params = config.get('our_params', {})
        self.our_angle_params = our_params.get('angle_params', {
            'theta_Rmax': 80, 'theta_Max': 50, 'theta_Maxmin': 30
        })
        self.our_distance_params = our_params.get('distance_params', {
            'D_Rmax': 120, 'D_Mmax': 60, 'D_Mmin': 5,
            'D_NZmax': 30, 'D_NZmin': 5
        })

        # enemy_params
        enemy_params = config.get('enemy_params', {})
        self.enemy_angle_params = enemy_params.get('angle_params', {
            'phi_Rmax': 70, 'phi_Max': 40, 'phi_Maxmin': 20
        })
        self.enemy_distance_params = enemy_params.get('distance_params', {
            'D_Rmax': 100, 'D_Mmax': 40, 'D_Mmin': 5,
            'D_NZmax': 20, 'D_NZmin': 2
        })

        # speed_params
        self.speed_params = config.get('speed_params', {
            'D_high': 80, 'V_high': 350, 'D_low': 30,
            'V_low': 250, 'V_base': 300
        })

    def calculate_single_threat(self, our_aircraft, enemy_aircraft):
        """
        计算单机对单敌的态势威胁值
        :param our_aircraft: 我方飞机参数字典 {'speed': float, 'height': float, 'heading': float, 'position': np.array}
        :param enemy_aircraft: 敌方飞机参数字典 {'speed': float, 'height': float, 'heading': float, 'position': np.array}
        :return: 单对单威胁评估值 X (0-1)
        """
        return self._calculate_single_X(our_aircraft, enemy_aircraft)

    def calculate_situation_components(self, our_aircraft, enemy_aircraft):
        """
        计算单机对单敌的各个态势分量
        :param our_aircraft: 我方飞机参数字典
        :param enemy_aircraft: 敌方飞机参数字典
        :return: 包含各态势分量的字典
        """
        S_A = self._angle_situation(our_aircraft['heading'], our_aircraft['position'], enemy_aircraft)
        S_D = self._distance_situation(our_aircraft, enemy_aircraft)
        S_V = self._speed_situation(our_aircraft['speed'], enemy_aircraft['speed'], our_aircraft['position'], enemy_aircraft)
        S_H = self._height_situation(our_aircraft['height'], enemy_aircraft['height'])
        
        # 计算综合威胁值
        X = 0.76 * (S_A ** 0.72 * S_D ** 0.28) + 0.14 * S_V + 0.10 * S_H
        
        return {
            'angle_situation': S_A,
            'distance_situation': S_D,
            'speed_situation': S_V,
            'height_situation': S_H,
            'total_threat': X
        }

    def _calculate_single_X(self, our_params, enemy):
        """单敌机X值计算"""
        S_A = self._angle_situation(our_params['heading'], our_params['position'], enemy)
        S_D = self._distance_situation(our_params, enemy)
        S_V = self._speed_situation(our_params['speed'], enemy['speed'], our_params['position'], enemy)
        S_H = self._height_situation(our_params['height'], enemy['height'])

        # 聚合X值
        X = 0.76 * (S_A ** 0.72 * S_D ** 0.28) + 0.14 * S_V + 0.10 * S_H
        return X

    def _angle_situation(self, heading, our_pos, enemy):
        """
        角度态势函数
        :param heading: 我机当前航向角（度）
        :param our_pos: 我机坐标 np.array([x, y])（km） （北，东）
        :param enemy: 敌机信息字典
        :return: 角度态势值SA (0~1)
        """
        # 从类属性获取预设参数
        theta_Rmax = self.our_angle_params['theta_Rmax']
        theta_Max = self.our_angle_params['theta_Max']
        theta_Maxmin = self.our_angle_params['theta_Maxmin']
        phi_Rmax = self.enemy_angle_params['phi_Rmax']
        phi_Max = self.enemy_angle_params['phi_Max']
        phi_Maxmin = self.enemy_angle_params['phi_Maxmin']


        # 计算相对方位角phi（我机头指向与敌机方位夹角）
        enemy_pos = enemy['position']
        dx = enemy_pos[0] - our_pos[0]  # 北方向差值
        dy = enemy_pos[1] - our_pos[1]  # 东方向差值
        # np.arctran2(y, x)则是计算向量[x,y]与向量[1,0]（即x轴正方向）的角度
        target_azimuth = np.arctan2(dy, dx) * 180 / np.pi
        phi = abs(heading - target_azimuth) % 360
        phi = phi if phi <= 180 else 360 - phi

        # 计算进入角theta（敌机头指向与我机方位夹角）
        dx_enemy = our_pos[0] - enemy_pos[0]  # 北方向差值
        dy_enemy = our_pos[1] - enemy_pos[1]  # 东方向差值
        our_azimuth_to_enemy = np.arctan2(dy_enemy, dx_enemy) * 180 / np.pi
        theta = abs(enemy['heading'] - our_azimuth_to_enemy) % 360
        theta = theta if theta <= 180 else 360 - theta

        # 分段计算SA（根据论文式3）
        if 0 <= phi < phi_Maxmin and theta_Rmax <= theta <= 180:
            # 情况1：绝对优势
            denominator = 180 - (theta_Rmax - phi_Maxmin)
            if denominator > 0:
                exponent = ((theta - phi) - (theta_Rmax - phi_Maxmin)) / denominator
                SA = 0.9 - 0.1 * np.exp(-exponent)
            else:
                SA = 0.9

        elif phi_Maxmin <= phi < phi_Max and theta_Rmax <= theta < 180:
            # 情况2：明显占优
            phi1, phi2 = phi_Maxmin, phi_Max
            theta1, theta2 = theta_Rmax, 180
            denominator = (theta2 - phi1) - (theta1 - phi2)
            if denominator > 0:
                exponent = ((theta - phi) - (theta1 - phi2)) / denominator
                SA = 0.8 - 0.15 * np.exp(-exponent)
            else:
                SA = 0.8

        elif 0 <= phi < phi_Maxmin and theta_Max <= theta < theta_Rmax:
            # 情况2：明显占优
            phi1, phi2 = 0, phi_Maxmin
            theta1, theta2 = theta_Max, theta_Rmax
            denominator = (theta2 - phi1) - (theta1 - phi2)
            if denominator > 0:
                exponent = ((theta - phi) - (theta1 - phi2)) / denominator
                SA = 0.8 - 0.15 * np.exp(-exponent)
            else:
                SA = 0.8

        elif phi_Maxmin <= phi < phi_Max and theta_Max <= theta < theta_Rmax:
            # 情况3：略微占优
            phi1, phi2 = phi_Maxmin, phi_Max
            theta1, theta2 = theta_Max, theta_Rmax
            denominator = (theta2 - phi1) - (theta1 - phi2)
            if denominator > 0:
                exponent = ((theta - phi) - (theta1 - phi2)) / denominator
                SA = 0.65 - 0.15 * np.exp(-exponent)
            else:
                SA = 0.65

        elif 0 <= phi < phi_Maxmin and theta_Maxmin <= theta < theta_Max:
            # 情况3：略微占优
            phi1, phi2 = 0, phi_Maxmin
            theta1, theta2 = theta_Maxmin, theta_Max
            denominator = (theta2 - phi1) - (theta1 - phi2)
            if denominator > 0:
                exponent = ((theta - phi) - (theta1 - phi2)) / denominator
                SA = 0.65 - 0.15 * np.exp(-exponent)
            else:
                SA = 0.65

        elif phi_Max <= phi < phi_Rmax and theta_Rmax <= theta < 180:
            # 情况3：略微占优
            phi1, phi2 = phi_Max, phi_Rmax
            theta1, theta2 = theta_Rmax, 180
            denominator = (theta2 - phi1) - (theta1 - phi2)
            if denominator > 0:
                exponent = ((theta - phi) - (theta1 - phi2)) / denominator
                SA = 0.65 - 0.15 * np.exp(-exponent)
            else:
                SA = 0.65

        elif (0 <= phi < phi_Maxmin and 0 <= theta < theta_Maxmin) or \
                (phi_Maxmin <= phi < phi_Max and theta_Maxmin <= theta < theta_Max) or \
                (phi_Max <= phi < phi_Rmax and theta_Max <= theta < theta_Rmax) or \
                (phi_Rmax <= phi and theta_Rmax <= theta):
            # 情况4：优劣相等
            SA = 0.5

        elif phi_Maxmin <= phi < phi_Max and 0 <= theta < theta_Maxmin:
            # 情况5：略微劣势
            phi1, phi2 = phi_Maxmin, phi_Max
            theta1, theta2 = 0, theta_Maxmin
            denominator = (phi2 - theta1) - (phi1 - theta2)
            if denominator > 0:
                exponent = ((phi - theta) - (phi1 - theta2)) / denominator
                SA = 0.35 + 0.15 * np.exp(-exponent)
            else:
                SA = 0.35

        elif phi_Rmax <= phi < 180 and theta_Max <= theta < theta_Rmax:
            # 情况5：略微劣势
            phi1, phi2 = phi_Rmax, 180
            theta1, theta2 = theta_Max, theta_Rmax
            denominator = (phi2 - theta1) - (phi1 - theta2)
            if denominator > 0:
                exponent = ((phi - theta) - (phi1 - theta2)) / denominator
                SA = 0.35 + 0.15 * np.exp(-exponent)
            else:
                SA = 0.35

        elif phi_Max <= phi < phi_Rmax and theta_Maxmin <= theta < theta_Max:
            # 情况5：略微劣势
            phi1, phi2 = phi_Max, phi_Rmax
            theta1, theta2 = theta_Maxmin, theta_Max
            denominator = (phi2 - theta1) - (phi1 - theta2)
            if denominator > 0:
                exponent = ((phi - theta) - (phi1 - theta2)) / denominator
                SA = 0.35 + 0.15 * np.exp(-exponent)
            else:
                SA = 0.35

        elif phi_Rmax <= phi < 180 and theta_Maxmin <= theta < theta_Max:
            # 情况6：明显劣势
            phi1, phi2 = phi_Rmax, 180
            theta1, theta2 = theta_Maxmin, theta_Max
            denominator = (phi2 - theta1) - (phi1 - theta2)
            if denominator > 0:
                exponent = ((phi - theta) - (phi1 - theta2)) / denominator
                SA = 0.2 + 0.15 * np.exp(-exponent)
            else:
                SA = 0.2

        elif phi_Max <= phi < phi_Rmax and 0 <= theta < theta_Maxmin:
            # 情况6：明显劣势
            phi1, phi2 = phi_Max, phi_Rmax
            theta1, theta2 = 0, theta_Maxmin
            denominator = (phi2 - theta1) - (phi1 - theta2)
            if denominator > 0:
                exponent = ((phi - theta) - (phi1 - theta2)) / denominator
                SA = 0.2 + 0.15 * np.exp(-exponent)
            else:
                SA = 0.2

        elif phi_Rmax <= phi and 0 <= theta < theta_Maxmin:
            # 情况7：绝对劣势
            denominator = 180 - (phi_Rmax - theta_Maxmin)
            if denominator > 0:
                exponent = ((phi - theta) - (phi_Rmax - theta_Maxmin)) / denominator
                SA = 0.1 + 0.1 * np.exp(-exponent)
            else:
                SA = 0.1

        else:
            # 未覆盖的边界情况默认取中间值
            SA = 0.5

        return np.clip(SA, 0.0, 1.0)

    def _distance_situation(self, our_params, enemy):
        """
        距离态势函数
        :param our_params: 我机参数（含position, height）
        :param enemy: 敌机参数（含position）
        """
        # 获取预设参数
        D_WRmax = self.our_distance_params['D_Rmax']
        D_WMmax = self.our_distance_params['D_Mmax']
        D_WMmin = self.our_distance_params['D_Mmin']
        D_WKmax = self.our_distance_params['D_NZmax']
        D_WKmin = self.our_distance_params['D_NZmin']
        D_MRmax = self.enemy_distance_params['D_Rmax']
        D_MMmax = self.enemy_distance_params['D_Mmax']
        D_MMmin = self.enemy_distance_params['D_Mmin']
        D_MKmax = self.enemy_distance_params['D_NZmax']
        D_MKmin = self.enemy_distance_params['D_NZmin']
        
        H = our_params['height']
        position = our_params['position']

        # 计算敌我相对距离（km）
        enemy_pos = enemy['position']
        dx = enemy_pos[0] - position[0]
        dy = enemy_pos[1] - position[1]
        D = np.sqrt(dx ** 2 + dy ** 2)

        # 高度影响系数
        m = abs((H - 6000) / 5000)

        # 分段计算SD - 修正远距离优势逻辑
        if D > D_WRmax:
            # 超出我方雷达探测距离 - 绝对劣势
            SD = (1 - m ** 2) * 0.1 * np.exp(-(D - D_WRmax) / D_WRmax)
            
        elif D_MRmax < D <= D_WRmax:
            # 我方能探测敌方，敌方无法探测我方 - 绝对优势
            decay_factor = np.exp(-(D - D_MRmax) / (D_WRmax - D_MRmax))
            SD = (1 - m ** 2) * (0.9 + 0.1 * decay_factor)
            
        elif D_WMmax < D <= D_MRmax:
            # 双方能探测，都无法攻击，但我方雷达距离更远 - 明显优势
            if D <= D_MMmax:
                # 敌方无法攻击，我方也无法攻击
                SD = (1 - m ** 2) * 0.8
            else:
                # 双方都无法攻击
                SD = (1 - m ** 2) * 0.75
                
        elif D_MMmax < D <= D_WMmax:
            # 我方能攻击，敌方无法攻击 - 绝对优势
            denominator = D_WMmax - D_MMmax
            if denominator > 0:
                exponent = (D - D_MMmax) / denominator
                SD = (1 - m ** 2) * (0.95 - 0.15 * exponent)
            else:
                SD = (1 - m ** 2) * 0.95

        elif D_WKmax < D <= D_MMmax:
            # 双方都能攻击，我方禁区距离更远 - 略微优势
            SD = (1 - m ** 2) * 0.65

        elif D_MKmax < D <= D_WKmax:
            # 我方处于最佳攻击距离 - 最佳优势
            SD = (1 - m ** 2)

        elif D_WMmin < D <= D_MKmax:
            # 双方都在攻击距离内 - 势均力敌
            SD = 0.5

        elif D_MMmin < D <= D_WMmin:
            # 进入敌方最小攻击距离，我方仍可攻击 - 略微劣势
            denominator = D_WMmin - D_MMmin
            if denominator > 0:
                SD = (1 - m ** 2) * 0.3 * np.exp(-(D - D_MMmin) / denominator)
            else:
                SD = (1 - m ** 2) * 0.3

        elif D_MKmin <= D <= D_MMmin:
            # 双方都在最小攻击距离内 - 明显劣势
            denominator = D_MMmin - D_MKmin
            if denominator > 0:
                SD = (1 - m ** 2) * 0.2 * np.exp(-(D - D_MKmin) / denominator)
            else:
                SD = (1 - m ** 2) * 0.2

        else:
            # 极近距离或其他边界情况 - 最大劣势
            SD = 0.1 * (1 - m ** 2)

        return np.clip(SD, 0.0, 1.0)

    def _speed_situation(self, V_W, V_M, our_pos, enemy):
        """
        速度态势函数
        :param V_W: 我机当前速度（m/s）
        :param V_M: 敌机当前速度（m/s）
        :param our_pos: 我机位置 np.array
        :param enemy: 敌机信息字典
        :return: 速度态势值SV (0~1)
        """
        # 动态计算最佳空战速度V_max
        enemy_pos = enemy['position']
        dx = enemy_pos[0] - our_pos[0]
        dy = enemy_pos[1] - our_pos[1]
        D = np.sqrt(dx ** 2 + dy ** 2)
        V_max = self._calculate_V_max(D)

        if V_max > 1.5 * V_M:
            # 情况1：V_max > 1.5V_M
            if V_W <= 0.6 * V_M:
                SV = 0.1
            elif 0.6 * V_M < V_W <= 1.5 * V_M:
                SV = -0.5 + (V_W / (1.5 * V_M))
            elif 1.5 * V_M < V_W <= V_max:
                SV = 1.0
            elif V_max < V_W:
                SV = np.exp(-(V_W - V_max) / V_max)
            else:
                SV = 1.0
        else:
            # 情况2：V_max ≤ 1.5V_M
            if V_M <= V_max <= V_W:
                exponent = -(V_W - V_max) / V_max
                SV = np.exp(exponent)
            elif 0.6 * V_M < V_W < V_max:
                term1 = V_W / V_max
                term2 = V_W / V_M
                SV = 0.3 * (term1 + term2)
            else:
                SV = 0.1

        return np.clip(SV, 0.0, 1.0)

    def _calculate_V_max(self, D):
        """
        动态计算最佳空战速度
        :param D: 当前敌我相对距离（km）
        :return: 最佳空战速度V_max（m/s）
        """
        params = self.speed_params

        if D >= params['D_high']:
            return params['V_high']
        elif D <= params['D_low']:
            return params['V_low']
        else:
            ratio = (D - params['D_low']) / (params['D_high'] - params['D_low'])
            return params['V_low'] + ratio * (params['V_high'] - params['V_low'])

    def _height_situation(self, H_W, H_M):
        """
        高度态势函数
        :param H_W: 我机当前高度（m）
        :param H_M: 敌机高度（m）
        :return: 高度态势值SH (0~1)
        """
        # 动态计算H_max
        H_max = H_M + 2000
        H_best = 6000

        # 分段计算SH
        if H_M < H_max <= H_W:
            # Case 1: 超过最佳高度 → 指数衰减
            exponent = -(H_W - H_best) / H_best
            SH = np.exp(exponent)

        elif H_M < H_W < H_max:
            # Case 2: 高于敌机但未超最佳 → 指数增长
            exponent = (H_W - H_best) / H_W
            SH = np.exp(exponent)

        elif 0.6 * H_M <= H_W <= H_M:
            # Case 3: 接近敌机高度 → 线性插值
            if H_M > 0:
                SH = -0.5 + (H_W / H_M)
            else:
                SH = 0.1

        else:
            # Case 4: 远低于敌机 → 最低值
            SH = 0.1

        return np.clip(SH, 0.1, 1.0)
