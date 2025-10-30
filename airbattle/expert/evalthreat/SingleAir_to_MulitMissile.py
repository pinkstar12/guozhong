"""
单个飞机对多个导弹的威胁评估模块

该模块通过调用ClassicMethod中的威胁评估函数，
实现单个飞机对多个导弹的威胁评估功能。
"""

import numpy as np
from .AirToMissileThreatModels.ClassicMethod import CalTreat_obj, CalTreat_single_missile, intervalEvaluation


class SingleAirToMultiMissile:
    """
    单个飞机对多个导弹的威胁评估类
    """
    
    def __init__(self):
        """
        初始化威胁评估器
        """
        pass
    
    def evaluate_threats(self, plane: dict, missiles: list[dict]) -> list[float]:
        """
        评估单个飞机对多个导弹的威胁度
        
        参数:
            plane: dict - 飞机信息字典，包含以下字段：
                - position: [x, z] - 飞机在x-z平面的位置坐标
                - height: float - 飞机高度(y坐标)
                - speed: float - 飞机速度
            missiles: list[dict] - 导弹信息列表，每个导弹包含：
                - position: [x, z] - 导弹在x-z平面的位置坐标
                - height: float - 导弹高度(y坐标)
                - speed: float - 导弹速度
        
        返回:
            list[float] - 威胁值列表，每个值对应一个导弹的威胁评估(0-1之间)
        """
        return CalTreat_obj(plane, missiles)
    
    def evaluate_single_threat(self, plane: dict, missile: dict) -> float:
        """
        评估单个飞机对单个导弹的威胁度
        
        参数:
            plane: dict - 飞机信息字典
            missile: dict - 导弹信息字典
        
        返回:
            float - 威胁值 (0-1之间)
        """
        return CalTreat_single_missile(plane, missile)
    
    def get_max_threat(self, plane: dict, missiles: list[dict]) -> tuple[float, int]:
        """
        获取最大威胁值及其对应的导弹索引
        
        参数:
            plane: dict - 飞机信息字典
            missiles: list[dict] - 导弹信息列表
        
        返回:
            tuple[float, int] - (最大威胁值, 导弹索引)
        """
        threats = self.evaluate_threats(plane, missiles)
        if not threats:
            return 0.0, -1
        
        max_threat = max(threats)
        max_index = threats.index(max_threat)
        return max_threat, max_index
    
    def get_threat_rankings(self, plane: dict, missiles: list[dict]) -> list[tuple[int, float]]:
        """
        获取威胁度排名（从高到低）
        
        参数:
            plane: dict - 飞机信息字典
            missiles: list[dict] - 导弹信息列表
        
        返回:
            list[tuple[int, float]] - [(导弹索引, 威胁值), ...] 按威胁值降序排列
        """
        threats = self.evaluate_threats(plane, missiles)
        if not threats:
            return []
        
        # 创建(索引, 威胁值)的元组列表
        indexed_threats = [(i, threat) for i, threat in enumerate(threats)]
        # 按威胁值降序排序
        indexed_threats.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_threats
    
    def get_threat_levels(self, plane: dict, missiles: list[dict]) -> list[tuple[int, float, str]]:
        """
        获取威胁等级评估
        
        参数:
            plane: dict - 飞机信息字典
            missiles: list[dict] - 导弹信息列表
        
        返回:
            list[tuple[int, float, str]] - [(导弹索引, 威胁值, 威胁等级), ...]
        """
        threats = self.evaluate_threats(plane, missiles)
        if not threats:
            return []
        
        results = []
        for i, threat in enumerate(threats):
            level = intervalEvaluation(threat)
            results.append((i, threat, level))
        
        return results
    
    def filter_high_threats(self, plane: dict, missiles: list[dict], threshold: float = 0.5) -> list[tuple[int, float]]:
        """
        筛选高威胁导弹
        
        参数:
            plane: dict - 飞机信息字典
            missiles: list[dict] - 导弹信息列表
            threshold: float - 威胁值阈值，默认0.5
        
        返回:
            list[tuple[int, float]] - [(导弹索引, 威胁值), ...] 威胁值大于阈值的导弹
        """
        threats = self.evaluate_threats(plane, missiles)
        if not threats:
            return []
        
        high_threats = []
        for i, threat in enumerate(threats):
            if threat > threshold:
                high_threats.append((i, threat))
        
        # 按威胁值降序排序
        high_threats.sort(key=lambda x: x[1], reverse=True)
        
        return high_threats


# 便捷函数接口
def evaluate_air_to_missiles(plane: dict, missiles: list[dict]) -> list[float]:
    """
    便捷函数：评估单个飞机对多个导弹的威胁度
    
    参数:
        plane: dict - 飞机信息字典
        missiles: list[dict] - 导弹信息列表
    
    返回:
        list[float] - 威胁值列表
    """
    evaluator = SingleAirToMultiMissile()
    return evaluator.evaluate_threats(plane, missiles)


def get_max_threat_missile(plane: dict, missiles: list[dict]) -> tuple[float, int]:
    """
    便捷函数：获取最大威胁值及其对应的导弹索引
    
    参数:
        plane: dict - 飞机信息字典
        missiles: list[dict] - 导弹信息列表
    
    返回:
        tuple[float, int] - (最大威胁值, 导弹索引)
    """
    evaluator = SingleAirToMultiMissile()
    return evaluator.get_max_threat(plane, missiles)


# 示例使用
if __name__ == "__main__":
    # 示例数据
    plane = {
        "position": [0, 0],     # x, z坐标
        "height": 1000,         # 高度
        "speed": 300           # 速度
    }
    
    missiles = [
        {
            "position": [1000, 500],   # x, z坐标
            "height": 800,             # 高度
            "speed": 250              # 速度
        },
        {
            "position": [2000, 1000],  # x, z坐标
            "height": 900,             # 高度
            "speed": 200              # 速度
        },
        {
            "position": [500, 300],    # x, z坐标
            "height": 700,             # 高度
            "speed": 280              # 速度
        }
    ]
    
    # 创建评估器
    evaluator = SingleAirToMultiMissile()
    
    # 评估威胁
    threats = evaluator.evaluate_threats(plane, missiles)
    print(f"威胁值列表: {threats}")
    
    # 获取最大威胁
    max_threat, max_index = evaluator.get_max_threat(plane, missiles)
    print(f"最大威胁值: {max_threat:.4f}, 导弹索引: {max_index}")
    
    # 获取威胁排名
    rankings = evaluator.get_threat_rankings(plane, missiles)
    print(f"威胁排名: {rankings}")
    
    # 获取威胁等级
    levels = evaluator.get_threat_levels(plane, missiles)
    print(f"威胁等级: {levels}")
    
    # 筛选高威胁导弹
    high_threats = evaluator.filter_high_threats(plane, missiles, threshold=0.3)
    print(f"高威胁导弹: {high_threats}")
