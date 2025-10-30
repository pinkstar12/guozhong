import numpy as np

class BeyondVisualRangeThreatModel:
    """
    超视距空战威胁评估模型
    实现基于距离、角度、高度、速度优势函数的态势评估方法
    """
    
    def __init__(self, our_params=None, enemy_params=None, speed_params=None, **kwargs):
        """
        初始化态势评估模块
        :param our_params: 我方参数配置
        :param enemy_params: 敌方参数配置
        :param speed_params: 速度参数配置
        """
        config = {
            'our_params': our_params or {},
            'enemy_params': enemy_params or {},
            'speed_params': speed_params or {},
        }
        self.initialize(config)
    
    def initialize(self, config):
        """
        从配置初始化态势评估模块参数
        :param config: 配置字典，包含各种参数
        """
        # 我方参数
        our_params = config.get('our_params', {})
        
        # 雷达参数
        self.D_Rmax = our_params.get('D_Rmax', 120)  # 雷达最大搜索距离(km)
        self.phi_Rmax = our_params.get('phi_Rmax', 80)  # 雷达最大搜索方位角(度)
        
        # 导弹参数
        self.D_Mmax = our_params.get('D_Mmax', 200)   # 导弹最大攻击距离(km)
        self.phi_Mmax = our_params.get('phi_Mmax', 50)  # 空空导弹最大离轴发射角(度)
        
        # 不可逃逸区参数
        self.D_MKmax = our_params.get('D_MKmax', 30)  # 导弹最大不可逃逸距离(km)
        self.D_MKmin = our_params.get('D_MKmin', 5)   # 导弹最小不可逃逸距离(km)
        self.phi_Mkmax = our_params.get('phi_Mkmax', 30)  # 导弹圆锥角(度)
        
        # 高度参数
        self.H_best = our_params.get('H_best', 10000)  # 载机最佳空战高度(m)
        
        # 权重系数
        self.omega_d = our_params.get('omega_d', 0.4)  # 距离优势权重
        self.omega_a = our_params.get('omega_a', 0.3)  # 角度优势权重
        self.omega_h = our_params.get('omega_h', 0.2)  # 高度优势权重
        self.omega_v = our_params.get('omega_v', 0.1)  # 速度优势权重
        
        # 角度优势函数的权重系数
        self.lambda1 = our_params.get('lambda1', 0.6)  # 方位角权重
        self.lambda2 = our_params.get('lambda2', 0.4)  # 进入角权重
        
        # 确保权重和为1
        total_weight = self.omega_d + self.omega_a + self.omega_h + self.omega_v
        if abs(total_weight - 1.0) > 1e-6:
            # 归一化权重
            self.omega_d /= total_weight
            self.omega_a /= total_weight
            self.omega_h /= total_weight
            self.omega_v /= total_weight
    
    def calculate_single_threat(self, our_aircraft, enemy_aircraft):
        """
        计算单机对单敌的态势威胁值
        :param our_aircraft: 我方飞机参数字典 {'speed': float, 'height': float, 'heading': float, 'position': np.array}
        :param enemy_aircraft: 敌方飞机参数字典 {'speed': float, 'height': float, 'heading': float, 'position': np.array}
        :return: 综合优势值 (0-1)
        """
        # 计算各个优势分量
        R_d = self._distance_advantage(our_aircraft, enemy_aircraft)
        R_a = self._angle_advantage(our_aircraft, enemy_aircraft)
        R_h = self._height_advantage(our_aircraft, enemy_aircraft)
        R_v = self._speed_advantage(our_aircraft, enemy_aircraft)
        
        # 综合优势函数 - 公式(3-25)
        R_i = (self.omega_d * R_d + 
               self.omega_a * R_a + 
               self.omega_h * R_h + 
               self.omega_v * R_v)
        
        return np.clip(R_i, 0.0, 1.0)
    
    def calculate_situation_components(self, our_aircraft, enemy_aircraft):
        """
        计算各个态势分量的详细信息
        :param our_aircraft: 我方飞机参数字典
        :param enemy_aircraft: 敌方飞机参数字典
        :return: 包含各态势分量的字典
        """
        R_d = self._distance_advantage(our_aircraft, enemy_aircraft)
        R_a = self._angle_advantage(our_aircraft, enemy_aircraft)
        R_h = self._height_advantage(our_aircraft, enemy_aircraft)
        R_v = self._speed_advantage(our_aircraft, enemy_aircraft)
        
        total_advantage = (self.omega_d * R_d + 
                          self.omega_a * R_a + 
                          self.omega_h * R_h + 
                          self.omega_v * R_v)
        
        return {
            'distance_advantage': R_d,
            'angle_advantage': R_a,
            'height_advantage': R_h,
            'speed_advantage': R_v,
            'total_threat': total_advantage,
            'components': {
                'distance_weighted': self.omega_d * R_d,
                'angle_weighted': self.omega_a * R_a,
                'height_weighted': self.omega_h * R_h,
                'speed_weighted': self.omega_v * R_v
            }
        }
    
    def _distance_advantage(self, our_aircraft, enemy_aircraft):
        """
        距离优势函数 - 公式(3-19)
        :param our_aircraft: 我方飞机参数
        :param enemy_aircraft: 敌方飞机参数
        :return: 距离优势值 R_d (0-1)
        """
        # 计算载机和敌机之间的距离
        our_pos = np.array(our_aircraft['position'])
        enemy_pos = np.array(enemy_aircraft['position'])
        D = np.linalg.norm(our_pos - enemy_pos)
        
        # 按照公式(3-19)分段计算
        if D < self.D_MKmin:
            # 距离过近，优势较低
            exponent = -(D - self.D_MKmax) / (10 - self.D_MKmax)
            R_d = 2 * np.exp(exponent)
            
        elif self.D_MKmin <= D < self.D_MKmax:
            # 不可逃逸区内，优势很高
            R_d = 1.0
            
        elif self.D_MKmax <= D < self.D_Mmax:
            # 导弹攻击区内，优势较高
            exponent = -(D - self.D_MKmax) / (self.D_Mmax - self.D_MKmax)
            R_d = 2.0 * np.exp(exponent)
            
        elif self.D_Mmax <= D < self.D_Rmax:
            # 雷达搜索区内但超出导弹射程，优势中等
            exponent = -(D - self.D_Mmax) / (self.D_Rmax - self.D_Mmax)
            R_d = 0.5 * np.exp(exponent)
            
        else:
            # 超出雷达搜索距离，但考虑协同作战
            exponent = -(D - self.D_Rmax) / self.D_Rmax  # 衰减系数
            R_d = 0.1839 * np.exp(exponent)
        
        return np.clip(R_d, 0.0, 2.0)  # 根据公式，距离优势可以超过1
    
    def _angle_advantage(self, our_aircraft, enemy_aircraft):
        """
        角度优势函数 - 公式(3-20)~(3-22)
        :param our_aircraft: 我方飞机参数
        :param enemy_aircraft: 敌方飞机参数
        :return: 角度优势值 R_a (0-1)
        """
        # 计算目标方位角φ（我机头指向与敌机方位夹角）
        our_pos = np.array(our_aircraft['position'])
        enemy_pos = np.array(enemy_aircraft['position'])
        
        # 目标相对于载机的方位角
        dx = enemy_pos[0] - our_pos[0]
        dy = enemy_pos[1] - our_pos[1]
        target_bearing = np.arctan2(dy, dx) * 180 / np.pi
        
        # 载机航向与目标方位的夹角
        phi = abs(our_aircraft['heading'] - target_bearing)
        phi = min(phi, 360 - phi)  # 取较小角度
        
        # 计算目标进入角q
        # 敌机航向与我机方位的夹角
        dx_reverse = our_pos[0] - enemy_pos[0]
        dy_reverse = our_pos[1] - enemy_pos[1]
        our_bearing_from_enemy = np.arctan2(dy_reverse, dx_reverse) * 180 / np.pi
        q = abs(enemy_aircraft['heading'] - our_bearing_from_enemy)
        q = min(q, 360 - q)  # 取较小角度
        
        # 方位角优势函数 R_φ - 公式(3-20)
        if  phi <= self.phi_Mkmax:
            # 目标在不可逃逸区
            R_phi = 1.0 - phi/(5* self.phi_Mkmax)
        elif self.phi_Mkmax < phi <= self.phi_Mmax:
            # 目标在导弹攻击区但未进入不可逃逸区
            factor = (self.phi_MKmax - phi) / (self.phi_Mmax - self.phi_Mkmax)
            R_phi = 0.8 + 0.5 * factor
        elif self.phi_Mmax < phi <= self.phi_Rmax:
            # 目标在雷达搜索区但未进入导弹攻击区
            factor = (phi-self.phi_Mmax ) / (self.phi_Rmax - self.phi_Mmax)
            R_phi = 0.3 * (1 - factor)
        else:
            # 目标超出雷达搜索区
            R_phi = 0.0
        
        # 进入角优势函数 R_q - 公式(3-21)
        if q <= 90:
            # 迎头交战，优势较大
            R_q = np.exp(-q * np.pi / 180)
        else:
            # 尾追情况，优势较小
            R_q = np.exp(-(180 - q) * np.pi / 180)
        
        # 角度优势函数 - 公式(3-22)
        R_a = self.lambda1 * R_phi + self.lambda2 * R_q
        
        return np.clip(R_a, 0.0, 1.0)
    
    def _height_advantage(self, our_aircraft, enemy_aircraft):
        """
        高度优势函数 - 公式(3-23)
        :param our_aircraft: 我方飞机参数
        :param enemy_aircraft: 敌方飞机参数
        :return: 高度优势值 R_h (0-1)
        """
        H = our_aircraft['height']  # 载机高度
        H_T = enemy_aircraft['height']  # 蓝方战机高度
        
        if H < H_T:
            # 载机高度低于敌机，劣势
            exponent = -(H_T - H) / self.H_best
            R_h = np.exp(exponent)
        elif H >= H_T:
            # 载机高度高于敌机
            if H <= self.H_best:
                # 未超过最佳高度，优势
                R_h = 1.0
            else:
                # 超过最佳高度，优势递减
                exponent = -(H - self.H_best) / (H_T + self.H_best)
                R_h = np.exp(exponent) + 0.5
        
        return np.clip(R_h, 0.0, 1.0)
    
    def _speed_advantage(self, our_aircraft, enemy_aircraft):
        """
        速度优势函数 - 公式(3-24)
        :param our_aircraft: 我方飞机参数
        :param enemy_aircraft: 敌方飞机参数
        :return: 速度优势值 R_v (0-1)
        """
        V = our_aircraft['speed']  # 载机速度
        V_T = enemy_aircraft['speed']  # 蓝方战机速度
        
        if V_T == 0:
            return 1.0  # 避免除零
        
        speed_ratio = V / V_T
        
        if speed_ratio >= 1.5:
            # 速度优势明显
            R_v = 1.0
        elif 0.6 < speed_ratio < 1.5:
            # 速度优势适中
            R_v = 0.5 * speed_ratio - 0.5 + 0.6
        elif speed_ratio <= 0.6:
            # 速度劣势
            R_v = 0.1
        else:
            R_v = 0.5
        
        return np.clip(R_v, 0.0, 1.0)


class BaseAttackAreaThreatAnalysis:
    """
    兼容性包装类，保持与原有接口的兼容性
    """
    
    def __init__(self, our_params=None, enemy_params=None, speed_params=None, **kwargs):
        """
        初始化，创建新的超视距威胁模型实例
        """
        # 将原有参数映射到新模型
        mapped_params = self._map_legacy_params(our_params, enemy_params, speed_params)
        self.model = BeyondVisualRangeThreatModel(**mapped_params)
        
        # 保持原有的参数结构用于兼容性
        self.our_angle_params = our_params.get('angle_params', {}) if our_params else {}
        self.our_distance_params = our_params.get('distance_params', {}) if our_params else {}
        self.enemy_angle_params = enemy_params.get('angle_params', {}) if enemy_params else {}
        self.enemy_distance_params = enemy_params.get('distance_params', {}) if enemy_params else {}
        self.speed_params = speed_params or {}
    
    def _map_legacy_params(self, our_params, enemy_params, speed_params):
        """
        将原有参数格式映射到新模型参数格式
        """
        mapped = {}
        
        if our_params:
            distance_params = our_params.get('distance_params', {})
            angle_params = our_params.get('angle_params', {})
            
            mapped['our_params'] = {
                'D_Rmax': distance_params.get('D_Rmax', 120),
                'D_Mmax': distance_params.get('D_Mmax', 60),
                'D_MKmax': distance_params.get('D_NZmax', 30),
                'D_MKmin': distance_params.get('D_NZmin', 5),
                'phi_Rmax': angle_params.get('theta_Rmax', 80),
                'phi_Mmax': angle_params.get('theta_Max', 50),
                'phi_Mkmax': angle_params.get('theta_Maxmin', 30),
            }
        
        return mapped
    
    def initialize(self, config):
        """兼容性方法"""
        mapped_params = self._map_legacy_params(
            config.get('our_params'), 
            config.get('enemy_params'), 
            config.get('speed_params')
        )
        self.model = BeyondVisualRangeThreatModel(**mapped_params)
    
    def calculate_single_threat(self, our_aircraft, enemy_aircraft):
        """兼容性接口：计算单机对单敌的态势威胁值"""
        return self.model.calculate_single_threat(our_aircraft, enemy_aircraft)
    
    def calculate_situation_components(self, our_aircraft, enemy_aircraft):
        """兼容性接口：计算态势分量"""
        components = self.model.calculate_situation_components(our_aircraft, enemy_aircraft)
        
        # 映射到原有格式
        return {
            'angle_situation': components['angle_advantage'],
            'distance_situation': components['distance_advantage'],
            'speed_situation': components['speed_advantage'],
            'height_situation': components['height_advantage'],
            'total_threat': components['total_threat']
        }
    
    # 保留原有方法的存根，避免调用错误
    def _calculate_single_X(self, our_params, enemy):
        """兼容性方法"""
        return self.calculate_single_threat(our_params, enemy)
    
    def _angle_situation(self, heading, our_pos, enemy):
        """兼容性方法"""
        our_aircraft = {'heading': heading, 'position': our_pos, 'height': 6000, 'speed': 300}
        return self.model._angle_advantage(our_aircraft, enemy)
    
    def _distance_situation(self, our_params, enemy):
        """兼容性方法"""
        return self.model._distance_advantage(our_params, enemy)
    
    def _speed_situation(self, V_W, V_M, our_pos, enemy):
        """兼容性方法"""
        our_aircraft = {'speed': V_W, 'position': our_pos, 'height': 6000, 'heading': 0}
        enemy_aircraft = {'speed': V_M, 'position': enemy['position'], 'height': enemy.get('height', 6000), 'heading': enemy.get('heading', 0)}
        return self.model._speed_advantage(our_aircraft, enemy_aircraft)
    
    def _height_situation(self, H_W, H_M):
        """兼容性方法"""
        our_aircraft = {'height': H_W, 'speed': 300, 'position': np.array([0, 0]), 'heading': 0}
        enemy_aircraft = {'height': H_M, 'speed': 300, 'position': np.array([1000, 1000]), 'heading': 0}
        return self.model._height_advantage(our_aircraft, enemy_aircraft)
