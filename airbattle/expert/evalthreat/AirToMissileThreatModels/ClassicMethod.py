import numpy as np
import warnings

"""
层次分析法
"""
class AHP:
    def __init__(self, criteria):
        self.RI = (0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)
        self.criteria = criteria
        self.num_criteria = criteria.shape[0]

    def calculate_weights(self, input_matrix):
        input_matrix = np.array(input_matrix)
        n, n1 = input_matrix.shape
        assert n == n1, "the matrix is not orthogonal"
        for i in range(n):
            for j in range(n):
                if np.abs(input_matrix[i, j] * input_matrix[j, i] - 1) > 1e-7:
                    raise ValueError("the matrix is not symmetric")
        eigen_values, eigen_vectors = np.linalg.eig(input_matrix)
        max_eigen = np.max(eigen_values)
        max_index = np.argmax(eigen_values)
        eigen = eigen_vectors[:, max_index]
        eigen = eigen / eigen.sum()
        if n > 9:
            CR = None
            warnings.warn("can not judge the uniformity")
        else:
            CI = (max_eigen - n) / (n - 1)
            CR = CI / self.RI[n - 1]
        return max_eigen, CR, eigen

    def calculate_mean_weights(self, input_matrix):
        input_matrix = np.array(input_matrix)
        n, n1 = input_matrix.shape
        assert n == n1, "the matrix is not orthogonal"
        A_mean = []
        for i in range(n):
            mean_value = input_matrix[:, i] / np.sum(input_matrix[:, i])
            A_mean.append(mean_value)
        eigen = []
        A_mean = np.array(A_mean)
        for i in range(n):
            eigen.append(np.sum(A_mean[:, i]) / n)
        eigen = np.array(eigen)
        matrix_sum = np.dot(input_matrix, eigen)
        max_eigen = np.mean(matrix_sum / eigen)
        if n > 9:
            CR = None
            warnings.warn("can not judge the uniformity")
        else:
            CI = (max_eigen - n) / (n - 1)
            CR = CI / self.RI[n - 1]
        return max_eigen, CR, eigen

    def run(self, method="calculate_weights"):
        weight_func = eval(f"self.{method}")
        max_eigen, CR, criteria_eigen = weight_func(self.criteria)
        print('准则层：最大特征值{:<5f},CR={:<5f},检验{}通过'.format(max_eigen, CR, '' if CR < 0.1 else '不'))
        print('准则层权重={}\n'.format(criteria_eigen))
        return criteria_eigen



import math as m




"""
    1、角度威胁模型
        初始化参数：

    ----------

    Heading_max: 0 < double类型 < pi/2
           来袭导弹的最大水平攻击角度

    Pitch_max:0 < double类型 < pi/2
              来袭导弹的最大俯仰攻击角度

    omega:0 < double类型 < 1
          水平角度的加权值，越大说明水平角度对角度威胁的影响越大


    Returns

    -----------

    AngleThreat对象
"""


class AngleThreat:
    def __init__(self, Heading_max, Pitch_max, omega):
        self.Heading_max = Heading_max
        self.Pitch_max = Pitch_max
        self.omega = omega

    """
        角度威胁值计算函数

            参数：

            ----------

            Heading:double类型 
                    我方战斗机相对于敌方导弹单位的水平角度

            Pitch:double类型
                  我方导弹相对于敌方反导单位的俯仰角度


            Returns

            -----------

            角度威胁值：0 <= Ta <= 1                       
    """

    def CalTa(self, Heading, Pitch):
        Ta = m.exp(
            max(1 - self.omega * (abs(Heading) / self.Heading_max) - (1 - self.omega) * (abs(Pitch) / self.Pitch_max),
                0))
        return Ta/m.e


"""
    2、距离威胁模型
        初始化参数：

    ----------

    dist_max: 0 < double类型 
           敌方导弹的最大攻击距离

    Returns

    -----------

    DistanceThreat对象
"""


class DistanceThreat:
    def __init__(self, kd, sigma, dist_max):
        self.dist_max = dist_max
        self.kd = kd
        self.sigma = sigma

    """
        距离威胁值计算函数

            参数：

            ----------

            dist:double类型 
                    我方战斗机与敌方来袭导弹之间的距离


            Returns

            -----------

            距离威胁值：0 <= Td <= 1                       
    """

    def CalTd(self, dist):
        # Td = m.exp(max(1 - (abs(dist) / self.dist_max), 0))
        # return Td/m.e
        Td = self.kd * (1 / (abs(dist) + self.sigma) - 1 / self.dist_max) ** 2 * (abs(dist) + self.sigma) ** 2
        return Td


"""
    3、速度威胁模型
        初始化参数：

    ----------

    ve: 0 < double类型 
        敌方导弹的预估速度
    
    v: 0 < double类型
      我方导弹的预估速度
    
    Returns

    -----------

    SpeedThreat对象
"""


# 3、速度威胁模型
class SpeedThreat:
    def __init__(self, ve, v):
        self.ve = ve
        self.v = v

    """
        速度威胁值计算函数

            参数：

            ----------
            无

            Returns

            -----------

            速度威胁值：0 <= Ts <= 1                       
    """

    def CalTs(self):
        Adv_speed = self.ve / self.v
        if Adv_speed <= 0.6:
            return 0.1
        if Adv_speed > 1.5:
            return 1
        else:
            return Adv_speed - 0.5

    def setV(self, ve, v):
        self.ve = ve
        self.v = v


# 4、总体受威胁态势建模
class TreatEva:
    def __init__(self, at, dt, st, rola, rold, rols):
        self.AT = at
        self.DT = dt
        self.ST = st
        self.rola = rola
        self.rold = rold
        self.rols = rols

    def CalT(self):
        T = self.AT.CalTa * self.rola + self.DT.CalTd * self.rold + self.ST.CalTs * self.rols
        return T


"""
    威胁评估的超参数
"""

fi_max, theta_max = m.pi * 0.5, m.pi * 0.5
omega = 0.2
dist_max = 150000
angle_threat = AngleThreat(fi_max, theta_max, omega)
distance_threat = DistanceThreat(kd=1, sigma=1e-8, dist_max=dist_max)
speed_threat = SpeedThreat(100, 200)
# 输入Angle、distance、speed威胁评估矩阵
#           角威胁 速度威胁 距离威胁
# 角威胁
# 速度威胁
# 距离威胁
criteria = np.array([[1, 1 / 2, 1 / 8],
                     [2, 1, 1 / 6],
                     [8, 6, 1]])
a = AHP(criteria).run("calculate_mean_weights")


# 根据飞机和导弹的位置计算威胁度（传入对象）- 单个导弹版本

def CalTreat_single_missile(plane: dict, missile: dict):
    """
    计算单个飞机对单个导弹的威胁评估
    
    参数:
        plane: dict - 飞机信息字典，包含position、height、speed等字段
        missile: dict - 导弹信息字典，包含position、height、speed等字段
    
    返回:
        float - 威胁值 (0-1之间)
    """
    # 计算角度威胁
    plane_position = [plane["position"][0], plane["height"], plane["position"][1]] #北天东坐标系 北→positon[0] 天→height 东→position[1]
    missile_position = [missile["position"][0], missile["height"],missile["position"][1]]
    V_m = missile["speed"]
    V_p = plane["speed"]
    angle_Heading = ComputeHeading(plane_position, missile_position)
    angle_Pitch = ComputePitch(plane_position, missile_position)
    Treat_a = angle_threat.CalTa(angle_Heading, angle_Pitch)

    # 计算距离威胁
    dist = CalDistance(plane_position, missile_position)
    Treat_d = distance_threat.CalTd(dist)

    # 计算速度威胁
    speed_threat.setV(V_m, V_p)
    Treat_v = speed_threat.CalTs()

    # 计算加权后的威胁度
    Treat_tot = np.array([Treat_a, Treat_v, Treat_d])
    Treat = np.dot(Treat_tot, a)

    return Treat


def CalTreat_obj(plane: dict, missile: list[dict]):
    """
    计算单个飞机对多个导弹的威胁评估
    
    参数:
        plane: dict - 飞机信息字典，包含position、height、speed等字段
        missile: list[dict] - 导弹信息列表，每个元素包含position、height、speed等字段
    
    返回:
        list[float] - 威胁值列表，每个值对应一个导弹的威胁评估
    """
    if not missile:  # 处理空导弹列表
        return []
    
    threats = []
    for single_missile in missile:
        threat = CalTreat_single_missile(plane, single_missile)
        threats.append(threat)
    
    return threats


# 根据飞机和导弹的位置计算威胁度

def CalTreat(plane_position, missile_position, V_p, V_m):
    # 计算角度威胁
    angle_Heading = ComputeHeading(plane_position, missile_position)
    angle_Pitch = ComputePitch(plane_position, missile_position)
    Treat_a = angle_threat.CalTa(angle_Heading, angle_Pitch)

    # 计算距离威胁
    dist = CalDistance(plane_position, missile_position)
    Treat_d = distance_threat.CalTd(dist)

    # 计算速度威胁
    speed_threat.setV(V_m, V_p)
    Treat_v = speed_threat.CalTs()

    # 计算加权后的威胁度
    Treat_tot = np.array([Treat_a, Treat_v, Treat_d])
    Treat = np.dot(Treat_tot, a)

    return Treat

# 置信区间输出威胁等级

def intervalEvaluation(Treat):
    low = 0.1
    mid = 0.5
    EvaResult = ''
    if Treat <= low:
        EvaResult = '低危'
    elif low < Treat <= mid:
        EvaResult = '中危'
    else:
        EvaResult = '高危'
    return EvaResult


'''
根据当前位置计算自身指向目标时的偏航角
（相较于坐标轴的角度）
'''


def ComputeHeading(TargetPos, SelfPos):
    m_x = SelfPos[0]
    a_x = TargetPos[0]
    m_z = SelfPos[2]
    a_z = TargetPos[2]

    x = a_x - m_x
    z = a_z - m_z
    Heading = m.atan(abs(z) / abs(x + 10e-8))

    if (x >= 0 and z > 0):
        Heading = -Heading

    elif (x < 0 and z <= 0):
        Heading = m.pi - Heading

    elif (x < 0 and z > 0):
        Heading = Heading - m.pi

    return Heading  # 模型的xz坐标系，当x轴正方向向上↑👆的时候z轴正方向向左←👈


def ComputePitch(TargetPos, SelfPos):
    m_y = SelfPos[1]
    a_y = TargetPos[1]

    m_x = SelfPos[0]
    a_x = TargetPos[0]

    m_z = SelfPos[2]
    a_z = TargetPos[2]

    Pitch = m.atan((a_y - m_y) / m.sqrt(((a_x - m_x) ** 2) + ((a_z - m_z) ** 2) + 10e-8))

    return Pitch

"""
    函数功能：根据两个点的坐标计算【距离】

    输入参数：
            SelfPosition：[x, y, z]坐标
            TargetPosition: [x, y, z]坐标


    ----------
    Returns

    ----------
    距离：distance： double型
"""


def CalDistance(SelfPosition, TargetPosition):
    SelfPosition = np.array(SelfPosition)
    TargetPosition = np.array(TargetPosition)

    distance = abs(np.linalg.norm(SelfPosition - TargetPosition))

    return distance
