import yaml
import numpy as np
import pickle
import time
import logging
import sys
import shutil
import re
import random
from typing import List, Dict,Tuple, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from .expertsystem import create_expert_system_from_config
from env.pyarkservice.arksim_env import arksim_env as AirCombatEnv
from tqdm import trange
import os
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import transform_state, inverse_transform_actions

# 导入威胁评估便捷函数
from .evalthreat.Mutilair_to_Mutilair import (
    get_optimal_targets_from_state, 
    get_max_threats_from_state
)
from .evalthreat.SingleAir_to_MulitMissile import get_max_threat_missile
# 配置日志
# 定义绿色ANSI转义码
GREEN = "\033[32m"
RESET = "\033[0m"

class GreenInfoFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        if record.levelno == logging.INFO:
            return f"{GREEN}{msg}{RESET}"
        return msg

# 用你原来的格式
fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

# 新建StreamHandler并设置自定义Formatter
handler = logging.StreamHandler()
handler.setFormatter(GreenInfoFormatter(fmt))

# 重新设置root logger
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

def load_config(path: str = "configs/generators.yaml") -> dict:
    if not os.path.isabs(path):
        # 用项目根目录和 path 拼接
        # 先找到当前文件（generators.py）路径的上一级（即项目根目录）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        #path = os.path.join(project_root, path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)




class ExperienceGenerator:
    def __init__(self, cfg_path: str = "configs/generators.yaml"):
        cfg = load_config(cfg_path)
        try:
            # 使用新的整合专家系统创建函数
            self.expert = create_expert_system_from_config(cfg_path)
        except Exception as e:
            logger.error(f"Failed to create integrated expert system: {e}")
            raise

        gen_cfg = cfg.get('generator', {})
        self.num_episodes = int(gen_cfg.get('num_episodes', 300))
        self.env_seed = int(gen_cfg.get('env_seed', 0))
        self.dataset_path = gen_cfg.get('dataset_path', "data/expert_v1.pkl")
        self.steps_per_env = int(gen_cfg.get('steps_per_env', 128))
        self.save_interval = int(gen_cfg.get('save_interval', 10))  # 每10个episode保存一次
        self.env = None
        self.service_address = gen_cfg.get('service_address', "tcp://10.100.70.140:60004")
        self.scenario_paths = gen_cfg.get('scenario_paths', ["D:/Code/air_to_air/2v2.txt"])
        
        # 获取场景修改器配置并传递给 RandomScenarioModifier
        scenario_modifier_config = gen_cfg.get('scenario_modifier', {})
        scenario_path = gen_cfg.get('scenario', ["scenarios/2v2_six_dof.txt"])[0]
        self.scenario_modifier = RandomScenarioModifier(scenario_path, scenario_modifier_config)
        
        # 超时监控配置
        self.env_step_timeout = int(gen_cfg.get('env_step_timeout', 60))  # 环境step超时时间
        self.env_close_timeout = int(gen_cfg.get('env_close_timeout', 10))  # 环境关闭超时时间
        self.timeout_count = 0  # 超时计数器

        # 整合专家系统没有 our_init 属性，这里设置默认值
        self.N = len(gen_cfg.get('scenario_paths', []))
        if self.N == 0:
            logger.warning("No scenario paths configured.")

    def explain_actions(self, actions: List[dict], our_planes: List[dict], enemies: List[dict]) -> List[dict]:
        """
        生成动作解释（参考comprehensive_expert_test.py的实现）
        """
        explanations = []
        for i, action in enumerate(actions):
            our = our_planes[i]
            
            # 计算位置变化
            orig_pos = np.array(our['position'])
            action_pos = np.array(action['position'])
            pos_change = action_pos - orig_pos
            
            # 计算各项变化
            speed_change = action['speed'] - our['speed']
            height_change = action['height'] - our['height']
            heading_change = action['heading'] - our['heading']
            
            # 计算移动距离和方向
            horizontal_distance = np.linalg.norm(pos_change)
            if horizontal_distance > 0.1:
                direction = np.arctan2(pos_change[1], pos_change[0]) * 180 / np.pi
                direction = (direction + 360) % 360
                direction_desc = self._get_direction_description(direction)
            else:
                direction_desc = "无明显移动"
            
            # 分析策略类型
            strategy_type = self._analyze_strategy_type(pos_change, height_change, speed_change)
            
            # 目标分配分析
            target_assignment = self._analyze_target_assignment(i, our, enemies)
            
            explanation = {
                'aircraft_id': i,
                'original_state': {
                    'position': orig_pos.tolist(),
                    'speed': our['speed'],
                    'height': our['height'],
                    'heading': our['heading']
                },
                'action_state': {
                    'position': action_pos.tolist(),
                    'speed': action['speed'],
                    'height': action['height'],
                    'heading': action['heading']
                },
                'changes': {
                    'position_change': pos_change.tolist(),
                    'horizontal_distance': horizontal_distance,
                    'direction': direction_desc,
                    'speed_change': speed_change,
                    'height_change': height_change,
                    'heading_change': heading_change
                },
                'analysis': {
                    'strategy_type': strategy_type,
                    'target_assignment': target_assignment
                }
            }
            explanations.append(explanation)
        
        return explanations
    
    def _get_direction_description(self, angle_degrees: float) -> str:
        """将角度转换为方向描述"""
        angle = angle_degrees % 360
        
        if 337.5 <= angle or angle < 22.5:
            return "正东"
        elif 22.5 <= angle < 67.5:
            return "东北"
        elif 67.5 <= angle < 112.5:
            return "正北"
        elif 112.5 <= angle < 157.5:
            return "西北"
        elif 157.5 <= angle < 202.5:
            return "正西"
        elif 202.5 <= angle < 247.5:
            return "西南"
        elif 247.5 <= angle < 292.5:
            return "正南"
        elif 292.5 <= angle < 337.5:
            return "东南"
        else:
            return f"{angle:.0f}°方向"
    
    def _analyze_strategy_type(self, pos_change: np.ndarray, height_change: float, speed_change: float) -> str:
        """分析策略类型"""
        horizontal_movement = np.linalg.norm(pos_change)
        
        strategies = []
        
        if horizontal_movement > 5.0:
            strategies.append("水平机动")
        
        if abs(height_change) > 300:
            if height_change > 0:
                strategies.append("爬升机动")
            else:
                strategies.append("俯冲机动")
        
        if abs(speed_change) > 15:
            if speed_change > 0:
                strategies.append("加速前进")
            else:
                strategies.append("减速调整")
        
        if not strategies:
            return "保持态势"
        
        return " + ".join(strategies)
    
    def _analyze_target_assignment(self, aircraft_idx: int, aircraft: dict, enemies: List[dict]) -> str:
        """分析目标分配"""
        if not enemies:
            return "无敌机目标"
        
        aircraft_pos = np.array(aircraft['position'])
        
        # 计算到各敌机的距离
        distances = []
        for i, enemy in enumerate(enemies):
            enemy_pos = np.array(enemy['position'])
            distance = np.linalg.norm(aircraft_pos - enemy_pos)
            distances.append((i, distance))
        
        # 按距离排序
        distances.sort(key=lambda x: x[1])
        
        closest_enemy_idx, closest_distance = distances[0]
        
        # 根据距离给出优先级
        if closest_distance < 30:
            priority = "高优先级"
        elif closest_distance < 60:
            priority = "中优先级"
        else:
            priority = "低优先级"
        
        return f"敌机{closest_enemy_idx} (距离{closest_distance:.1f}单位, {priority})"

    def calculate_situation_metrics(self, obs):
        """计算态势评估指标"""
        metrics = {}
        
        try:
            # 处理状态以获取态势矩阵
            self.expert.attack_system.threat_model.process_state(obs)
            situation_matrix = self.expert.attack_system.threat_model.situation_matrix
            
            # 获取最优攻击目标和最大威胁敌方
            optimal_targets = get_optimal_targets_from_state(obs, self.expert.attack_system.threat_model)
            max_threat_enemies = get_max_threats_from_state(obs, self.expert.attack_system.threat_model)
            
            num_aircraft = len(obs['our_aircrafts'])
            
            # 1. 最优攻击目标态势值
            optimal_situation_values = []
            for i, target_idx in enumerate(optimal_targets):
                if (situation_matrix is not None and target_idx >= 0 
                    and target_idx < situation_matrix.shape[1]):
                    value = float(situation_matrix[i, target_idx])
                else:
                    value = 0.0
                optimal_situation_values.append(value)
            
            # 2. 最大威胁敌方威胁值
            threat_values = []
            for i, threat_idx in enumerate(max_threat_enemies):
                if (situation_matrix is not None and threat_idx >= 0 
                    and threat_idx < situation_matrix.shape[1]):
                    value = float(1.0 - situation_matrix[i, threat_idx])
                else:
                    value = 0.0
                threat_values.append(value)
            
            # 3. 导弹威胁值和锁定状态
            missile_threat_values = []
            missile_locked_status = []
            missiles = obs.get('missiles', [])
            
            for i, aircraft in enumerate(obs['our_aircrafts']):
                if missiles:
                    locked_missiles = []
                    for missile in missiles:
                        if missile.get('target') == i:
                            locked_missiles.append(missile)
                    
                    if locked_missiles:
                        # 进行单位转换以保持与expertsystem.py的一致性
                        # 将位置从千米转换为米，供威胁评估使用
                        converted_aircraft = self._convert_aircraft_units_to_meters(aircraft)
                        converted_missiles = self._convert_missiles_units_to_meters(locked_missiles)
                        threat_value, _ = get_max_threat_missile(converted_aircraft, converted_missiles)
                        missile_threat_values.append(float(threat_value))
                        missile_locked_status.append(True)
                    else:
                        missile_threat_values.append(0.0)
                        missile_locked_status.append(False)
                else:
                    missile_threat_values.append(0.0)
                    missile_locked_status.append(False)
            
            metrics = {
                'optimal_situation_values': optimal_situation_values,
                'threat_values': threat_values,
                'missile_threat_values': missile_threat_values,
                'missile_locked_status': missile_locked_status
            }
            
        except Exception as e:
            logger.warning(f"计算态势指标失败: {e}")
            num_aircraft = len(obs['our_aircrafts'])
            metrics = {
                'optimal_situation_values': [0.0] * num_aircraft,
                'threat_values': [0.0] * num_aircraft,
                'missile_threat_values': [0.0] * num_aircraft,
                'missile_locked_status': [False] * num_aircraft
            }
        
        return metrics

    def calculate_rewards(self, obs, next_obs, enhanced_actions):
        """计算基于态势评估对比的奖励"""
        try:
            # 计算当前状态和下一状态的态势指标
            current_metrics = self.calculate_situation_metrics(obs)
            next_metrics = self.calculate_situation_metrics(next_obs)
            
            num_aircraft = len(obs['our_aircrafts'])
            rewards = []
            
            # 奖励权重配置
            OPTIMAL_SITUATION_WEIGHT = 0.6  # 最优攻击目标态势奖励权重
            THREAT_RESPONSE_WEIGHT = 0.4    # 威胁应对奖励权重
            MISSILE_EVASION_WEIGHT = 0.8    # 导弹规避奖励权重（被锁定时）
            SURVIVAL_BONUS_WEIGHT = 0.2     # 生存奖励权重（被锁定时）
            
            # 态势值阈值
            HIGH_SITUATION_THRESHOLD = 0.5
            
            for i in range(num_aircraft):
                # 获取当前和下一状态的指标
                curr_optimal_situation = current_metrics['optimal_situation_values'][i]
                next_optimal_situation = next_metrics['optimal_situation_values'][i]
                
                curr_threat_value = current_metrics['threat_values'][i]
                next_threat_value = next_metrics['threat_values'][i]
                
                curr_missile_threat = current_metrics['missile_threat_values'][i]
                next_missile_threat = next_metrics['missile_threat_values'][i]
                
                is_missile_locked = current_metrics['missile_locked_status'][i]
                
                # 计算各项奖励分量
                
                # 1. 最优攻击目标态势奖励
                situation_improvement = (next_optimal_situation - curr_optimal_situation) * 2.0
                high_situation_bonus = 0.2 if next_optimal_situation > HIGH_SITUATION_THRESHOLD else 0.0
                optimal_situation_reward = situation_improvement + high_situation_bonus
                
                # 2. 威胁应对奖励（威胁值降低）
                threat_reduction = (curr_threat_value - next_threat_value) * 1.5
                threat_response_reward = max(0.0, threat_reduction)  # 只奖励威胁降低
                
                # 3. 导弹威胁规避奖励
                missile_evasion_reward = (curr_missile_threat - next_missile_threat) * 2.0
                
                # 根据导弹锁定状态选择不同的奖励策略
                if is_missile_locked:
                    # 被导弹锁定时：优先考虑导弹规避，忽略态势恶化的负面影响
                    reward = (MISSILE_EVASION_WEIGHT * missile_evasion_reward + 
                             SURVIVAL_BONUS_WEIGHT * max(0.0, optimal_situation_reward))
                    
                    logger.debug(f"友机{i}被导弹锁定: 导弹威胁{curr_missile_threat:.3f}->{next_missile_threat:.3f}, "
                               f"规避奖励={missile_evasion_reward:.3f}, 总奖励={reward:.3f}")
                else:
                    # 正常状态：平衡攻击态势和防御应对
                    reward = (OPTIMAL_SITUATION_WEIGHT * optimal_situation_reward + 
                             THREAT_RESPONSE_WEIGHT * threat_response_reward)
                    
                    logger.debug(f"友机{i}正常状态: 态势{curr_optimal_situation:.3f}->{next_optimal_situation:.3f}, "
                               f"威胁{curr_threat_value:.3f}->{next_threat_value:.3f}, "
                               f"态势奖励={optimal_situation_reward:.3f}, 威胁奖励={threat_response_reward:.3f}, "
                               f"总奖励={reward:.3f}")
                
                rewards.append(float(reward))
            
            return rewards
            
        except Exception as e:
            logger.warning(f"计算奖励失败: {e}")
            # 返回默认奖励（0奖励）
            return [0.0] * len(obs['our_aircrafts'])

    def calculate_enhanced_info(self, obs):
        """计算增强信息"""
        enhanced_info = {}
        
        try:
            # 获取最优攻击目标
            enhanced_info['optimal_targets'] = get_optimal_targets_from_state(obs,self.expert.attack_system.threat_model)
            logger.info(f"最优攻击目标: {enhanced_info['optimal_targets']}")
        except Exception as e:
            logger.warning(f"计算最优攻击目标失败: {e}")
            enhanced_info['optimal_targets'] = [-1] * len(obs['our_aircrafts'])
        
        try:
            # 获取最高威胁敌方
            enhanced_info['max_threat_enemies'] = get_max_threats_from_state(obs,self.expert.attack_system.threat_model)
            logger.info(f"最高威胁敌方: {enhanced_info['max_threat_enemies']}")
        except Exception as e:
            logger.warning(f"计算最高威胁敌方失败: {e}")
            enhanced_info['max_threat_enemies'] = [-1] * len(obs['our_aircrafts'])
        
        # 计算态势评估值相关信息
        try:
            # 处理状态以获取态势矩阵
            self.expert.attack_system.threat_model.process_state(obs)
            situation_matrix = self.expert.attack_system.threat_model.situation_matrix
            
            if situation_matrix is not None:
                # 1. 获取最优攻击目标的态势评估值
                optimal_targets_evalThreatValue = []
                for i, target_idx in enumerate(enhanced_info['optimal_targets']):
                    if target_idx >= 0 and target_idx < situation_matrix.shape[1]:
                        eval_value = float(situation_matrix[i, target_idx])
                    else:
                        eval_value = 1.0
                    optimal_targets_evalThreatValue.append(eval_value)
                enhanced_info['optimal_targets_evalThreatValue'] = optimal_targets_evalThreatValue
                
                # 2. 获取最大威胁敌方的威胁值
                max_threat_enemies_evalThreatValue = []
                for i, threat_idx in enumerate(enhanced_info['max_threat_enemies']):
                    if threat_idx >= 0 and threat_idx < situation_matrix.shape[1]:
                        eval_value = float(1-situation_matrix[i, threat_idx])
                    else:
                        eval_value = 0.0
                    max_threat_enemies_evalThreatValue.append(eval_value)
                enhanced_info['max_threat_enemies_evalThreatValue'] = max_threat_enemies_evalThreatValue
                
                # 3. 获取友方飞机的综合态势评估值
                friendly_situation_values = []
                for i in range(situation_matrix.shape[0]):
                    avg_situation = float(np.mean(situation_matrix[i, :]))
                    friendly_situation_values.append(avg_situation)
                enhanced_info['friendly_situation_values'] = friendly_situation_values
                
                # 4. 获取总体态势评估值
                overall_advantage, overall_risk = self.expert.attack_system.threat_model.get_overall_situation()
                enhanced_info['overall_situation_value'] = float(overall_advantage)
                
                logger.debug(f"最优攻击目标态势评估值: {optimal_targets_evalThreatValue}")
                logger.debug(f"最大威胁敌方态势评估值: {max_threat_enemies_evalThreatValue}")
                logger.debug(f"友方综合态势评估值: {friendly_situation_values}")
                logger.debug(f"总体态势评估值: {overall_advantage}")
                
            else:
                # 如果态势矩阵为空，设置默认值
                num_aircraft = len(obs['our_aircrafts'])
                enhanced_info['optimal_targets_evalThreatValue'] = [0.0] * num_aircraft
                enhanced_info['max_threat_enemies_evalThreatValue'] = [0.0] * num_aircraft
                enhanced_info['friendly_situation_values'] = [0.0] * num_aircraft
                enhanced_info['overall_situation_value'] = 0.0
                logger.warning("态势矩阵为空，使用默认值")
                
        except Exception as e:
            logger.warning(f"计算态势评估值失败: {e}")
            # 设置默认值
            num_aircraft = len(obs['our_aircrafts'])
            enhanced_info['optimal_targets_evalThreatValue'] = [0.0] * num_aircraft
            enhanced_info['max_threat_enemies_evalThreatValue'] = [0.0] * num_aircraft
            enhanced_info['friendly_situation_values'] = [0.0] * num_aircraft
            enhanced_info['overall_situation_value'] = 0.0
        
        # 计算每架友机的最高威胁导弹
        max_threat_missiles = []
        missiles = obs.get('missiles', [])
        
        for i, aircraft in enumerate(obs['our_aircrafts']):
            try:
                if missiles:
                    locked_missiles = []
                    for missile in missiles:
                        target_id = missile['target']
                        if target_id == i:
                            locked_missiles.append(missile)
                    if locked_missiles is None:
                        threat_value = 0
                    else:
                        # 进行单位转换以保持与expertsystem.py的一致性
                        # 将位置从千米转换为米，供威胁评估使用
                        converted_aircraft = self._convert_aircraft_units_to_meters(aircraft)
                        converted_missiles = self._convert_missiles_units_to_meters(locked_missiles)
                        threat_value, missile_id = get_max_threat_missile(converted_aircraft, converted_missiles)
                    # 如果没有威胁，置0和-1
                    if threat_value == 0 or threat_value is None:
                        missile_id = -1
                        threat_value = 0.0
                else:
                    # 没有导弹时，威胁值为0，id为-1
                    threat_value = 0.0
                    missile_id = -1
                
                max_threat_missiles.append((float(threat_value), int(missile_id)))
                logger.info(f"友机{i}最高威胁导弹: 威胁值={threat_value}, 导弹ID={missile_id}")
                
            except Exception as e:
                logger.warning(f"计算友机{i}最高威胁导弹失败: {e}")
                max_threat_missiles.append((0.0, -1))
        
        enhanced_info['max_threat_missiles'] = max_threat_missiles
        
        return enhanced_info


    # def _execute_env_step_with_timeout(self, actions):
    #     """
    #     使用超时监控执行环境step操作
    #     """
    #     def _env_step():
    #         return self.env.step(actions)
        
    #     with ThreadPoolExecutor(max_workers=1) as executor:
    #         future = executor.submit(_env_step)
    #         try:
    #             result = future.result(timeout=self.env_step_timeout)
    #             return result
    #         except FutureTimeoutError:
    #             logger.error(f"Environment step timeout after {self.env_step_timeout} seconds")
    #             self.timeout_count += 1
    #             raise FutureTimeoutError("Environment step operation timed out")

    def run(self):
        # 如果已有文件，支持追加
        start_ep = 0
        try:
            with open(self.dataset_path, "rb") as f:
                logger.info(f"Existing dataset detected, will append new data.")
                # 读取一次确保文件有效性，不实际载入
        except FileNotFoundError:
            logger.info(f"No existing dataset, start new collection.")

        for ep in trange(start_ep, self.num_episodes, desc="Collecting Episodes"):
            # 1. 随机化战场想定
            logger.info("步骤 1: 随机化战场想定参数...")
            aircraft_params = self.scenario_modifier.randomize_scenario()
            if not aircraft_params:
                raise Exception("战场想定随机化失败")
            # 记录随机化参数
            logger.info("随机化参数详情:")
            for aircraft_name, params in aircraft_params.items():
                logger.info(f"  {aircraft_name}:")
                logger.info(f"    速度: 北{params['velocity'][0]:.1f} 东{params['velocity'][1]:.1f} fps")
                logger.info(f"    位置: 纬度{params['position'][0]:.2f} 经度{params['position'][1]:.2f}")
                logger.info(f"    高度: {params['altitude']:.0f} m")
                logger.info(f"    航向: {params['heading']:.1f}°")

            if(self.env is None):
                try:
                    self.env = AirCombatEnv(service_address=self.service_address, scenario_paths=self.scenario_paths)
                except Exception as e:
                    logger.error(f"Failed to initialize AirCombatEnv: {e}")
                    raise
            done = False
            obs = self.env.reset()

            while (not obs):
                time.sleep(1)
                obs = self.env.get_obs()
            # 转换obs格式：从[纬度,经度]转换为专家系统所需的NED坐标格式
            transformed_obs = transform_state(obs)
            obs = transformed_obs
            our_planes = obs['our_aircrafts']
            episode_buffer = []  # 全局状态数据集：存储每个step的完整state和actions
            step_count = 0
            logger.info(f"==== Episode {ep + 1}/{self.num_episodes} started ====")
            while step_count < self.steps_per_env and not done:
                
                # 调试信息：输出专家系统处理前的状态
                # logger.info("==== 专家系统输入状态 ====")
                # logger.info(f"当前obs: {obs}")
                
                result = self.expert.process(obs)
                actions = result['actions']
                
                # 调试信息：输出专家系统处理结果
                # logger.info("==== 专家系统处理结果 ====")
                # logger.info(f"专家系统结果: {result}")
               
                # 保存原始actions用于特征提取（专家系统格式）
                original_actions = result['actions']
                
                # 确保动作包含标志位
                enhanced_actions = []
                for i, action in enumerate(original_actions):
                    enhanced_action = action.copy()
                    # 添加动作标志位（如果专家系统没有提供）
                    enhanced_action['do_maneuver'] = result.get('do_maneuver', [False] * len(original_actions))[i]
                    enhanced_action['launch_missile'] = result.get('launch_missile', [False] * len(original_actions))[i]
                    enhanced_actions.append(enhanced_action)
                
                # 生成动作解释
                enemies = obs['enemies']
                action_explanations = self.explain_actions(enhanced_actions, our_planes, enemies)
                
                # # 调试信息：输出动作解释
                # logger.info("==== 动作解释详情 ====")
                # for explanation in action_explanations:
                #     aircraft_id = explanation['aircraft_id']
                #     changes = explanation['changes']
                #     analysis = explanation['analysis']
                #     logger.info(f"飞机{aircraft_id}:")
                #     logger.info(f"  原始状态: {explanation['original_state']}")
                #     logger.info(f"  优化状态: {explanation['action_state']}")
                #     logger.info(f"  位置变化: {changes['direction']} 距离{changes['horizontal_distance']:.1f}单位")
                #     logger.info(f"  速度变化: {changes['speed_change']:+.0f}节")
                #     logger.info(f"  高度变化: {changes['height_change']:+.0f}米")
                #     logger.info(f"  航向变化: {changes['heading_change']:+.0f}°")
                #     logger.info(f"  策略类型: {analysis['strategy_type']}")
                #     logger.info(f"  目标分配: {analysis['target_assignment']}")
                
                # 转换actions格式：从专家系统的NED坐标转换为环境所需的[纬度,经度]格式
                converted_actions = inverse_transform_actions(original_actions)
                
                
                actions = converted_actions
                logger.info(
                    f"Ep {ep + 1} Step {step_count + 1}: Collected {len(actions)} actions for {len(our_planes)} aircrafts.")
                try:
                    # 使用超时监控执行环境step操作
                    next_obs, _, done, _ = self.env.step(actions)
                    # 调试信息：输出环境执行结果
                    logger.debug("==== 环境执行结果 ====")
                    logger.debug(f"环境返回的next_obs: {next_obs}")
                    logger.debug(f"环境状态 done: {done}")
        
                # except FutureTimeoutError:
                #     logger.error(f"Ep {ep + 1} Step {step_count + 1}: Environment step timeout, ending current episode")
                #     # 环境重置
                #     try:
                #         self.env.close()
                #     except Exception as close_e:
                #         logger.warning(f"Failed to close environment: {close_e}")
                #     self.env = None
                #     break  # 直接结束当前episode
                except Exception as e:
                    logger.error(f"Ep {ep + 1} Step {step_count + 1}: Env step failed: {e}")
                    break

                # 计算增强信息
                enhanced_info = self.calculate_enhanced_info(obs)
                
                # 计算奖励（需要next_obs）
                rewards = []
                if next_obs:
                    # 转换next_obs格式以保持一致性
                    transformed_next_obs = transform_state(next_obs)
                    rewards = self.calculate_rewards(obs, transformed_next_obs, enhanced_actions)
                    logger.debug(f"计算奖励: {rewards}")
                else:
                    # 如果没有next_obs，设置默认奖励
                    rewards = [0.0] * len(obs['our_aircrafts'])
                    logger.warning("没有next_obs，使用默认奖励")
                
                # 保存全局状态数据：包含完整的state、actions、enhanced_info和rewards
                step_data = {
                    'state': {
                        'raw_state': obs,  # 原始战场状态
                        'enhanced_info': enhanced_info  # 增强信息
                    },
                    'actions': enhanced_actions,  # 增强后的动作列表（包含标志位）
                    'rewards': rewards  # 奖励列表
                }
                episode_buffer.append(step_data)
                
                # 输出增强信息的调试日志
                # logger.info("==== 增强信息详情 ====")
                # logger.info(f"最优攻击目标: {enhanced_info.get('optimal_targets', [])}")
                # logger.info(f"最高威胁敌方: {enhanced_info.get('max_threat_enemies', [])}")
                # logger.info(f"最高威胁导弹: {enhanced_info.get('max_threat_missiles', [])}")
                
                logger.info(f"Ep {ep + 1} Step {step_count + 1}: Saved enhanced global state with {len(original_actions)} actions")

                if hasattr(self.expert, 'update'):
                    self.expert.update(obs)
                # 转换next_obs格式以保持一致性
                if next_obs:
                    # 调试信息：输出next_obs转换前后
                    logger.debug("==== next_obs转换 ====")
                    logger.debug(f"转换前next_obs: {next_obs}")
                    transformed_next_obs = transform_state(next_obs)
                    logger.debug(f"转换后next_obs: {transformed_next_obs}")
                    obs = transformed_next_obs
                else:
                    obs = next_obs
                step_count += 1
            logger.info(
                f"==== Episode {ep + 1} finished: collected {step_count} steps ====")
            # 每个 episode 结束后直接写入文件（append）
            self.save_episode(episode_buffer)
            
            if (ep + 1) % self.save_interval == 0:
                logger.info(f"Episodes {ep + 1} saved.")
            #关闭环境（带超时保护）
            self.env.close()
            self.env = None
        
        # # 输出超时统计信息
        # if self.timeout_count > 0:
        #     logger.info(f"==== 超时统计 ====")
        #     logger.info(f"总超时次数: {self.timeout_count}")
        #     logger.info(f"超时率: {self.timeout_count / self.num_episodes:.2%}")
        # else:
        #     logger.info("==== 无超时发生 ====")

    def _convert_aircraft_units_to_meters(self, aircraft):
        """
        将飞机状态的位置单位从千米转换为米
        
        参数:
            aircraft: 飞机状态字典，位置单位为千米
        
        返回:
            转换后的飞机状态字典，位置单位为米
        """
        converted_aircraft = aircraft.copy()
        if 'position' in converted_aircraft:
            # 将位置从千米转换为米
            converted_aircraft['position'] = [coord * 1000 for coord in converted_aircraft['position']]
        return converted_aircraft
    
    def _convert_missiles_units_to_meters(self, missiles):
        """
        将导弹列表的位置单位从千米转换为米
        
        参数:
            missiles: 导弹列表，位置单位为千米
        
        返回:
            转换后的导弹列表，位置单位为米
        """
        converted_missiles = []
        for missile in missiles:
            converted_missile = missile.copy()
            if 'position' in converted_missile:
                # 将位置从千米转换为米
                converted_missile['position'] = [coord * 1000 for coord in converted_missile['position']]
            converted_missiles.append(converted_missile)
        return converted_missiles

    def save_episode(self, episode_buffer):
        """将全局状态数据集的episode追加到pkl文件"""
        # 用追加二进制写入
        data_dir = os.path.dirname(self.dataset_path)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir)
        with open(self.dataset_path, "ab") as f:
            pickle.dump(episode_buffer, f)

    def load_all(self):
        """加载全部采集数据，供后续分析/训练"""
        episodes = []
        with open(self.dataset_path, "rb") as f:
            while True:
                try:
                    ep = pickle.load(f)
                    episodes.append(ep)
                except EOFError:
                    break
        return episodes
    

class RandomScenarioModifier:
    """战场想定随机化修改器"""
    
    def __init__(self, scenario_path: str, config: dict = None):
        self.scenario_path = scenario_path
        self.backup_path = scenario_path + '.backup'
        
        # 从配置中读取参数，如果没有配置则使用默认值
        if config is None:
            config = {}
        
        # 飞机数量配置
        blue_nums = config.get('blue_aircraft_nums', 2)
        red_nums = config.get('red_aircraft_nums', 2)
        
        # 动态生成飞机平台名称
        self.AIRCRAFT_PLATFORMS = []
        for i in range(1, blue_nums + 1):
            self.AIRCRAFT_PLATFORMS.append(f'blue_{i}')
        for i in range(1, red_nums + 1):
            self.AIRCRAFT_PLATFORMS.append(f'red_{i}')
        
        # 速度参数配置
        velocity_config = config.get('velocity', {})
        
        # 读取双范围速度配置
        north_config = velocity_config.get('north_range', {})
        east_config = velocity_config.get('east_range', {})
        
        # 支持新格式（双范围）和旧格式（单范围）的兼容
        if isinstance(north_config, dict) and 'positive' in north_config:
            # 新的双范围格式
            self.VELOCITY_NORTH_POSITIVE = tuple(north_config.get('positive', [600, 1200]))
            self.VELOCITY_NORTH_NEGATIVE = tuple(north_config.get('negative', [-1200, -600]))
        else:
            # 旧的单范围格式兼容
            self.VELOCITY_NORTH_POSITIVE = tuple(north_config if isinstance(north_config, list) else [600, 1200])
            self.VELOCITY_NORTH_NEGATIVE = tuple([-self.VELOCITY_NORTH_POSITIVE[1], -self.VELOCITY_NORTH_POSITIVE[0]])
            
        if isinstance(east_config, dict) and 'positive' in east_config:
            # 新的双范围格式
            self.VELOCITY_EAST_POSITIVE = tuple(east_config.get('positive', [600, 1200]))
            self.VELOCITY_EAST_NEGATIVE = tuple(east_config.get('negative', [-1200, -600]))
        else:
            # 旧的单范围格式兼容
            self.VELOCITY_EAST_POSITIVE = tuple(east_config if isinstance(east_config, list) else [600, 1200])
            self.VELOCITY_EAST_NEGATIVE = tuple([-self.VELOCITY_EAST_POSITIVE[1], -self.VELOCITY_EAST_POSITIVE[0]])
            
        self.VELOCITY_DOWN = velocity_config.get('down', 0)
        
        # 位置参数配置
        position_config = config.get('position', {})
        self.LATITUDE_RANGE = tuple(position_config.get('latitude_range', [-2.0, 2.0]))
        self.LONGITUDE_BLUE = tuple(position_config.get('longitude_blue', [-1.5, -0.5]))
        self.LONGITUDE_RED = tuple(position_config.get('longitude_red', [0.5, 1.5]))
        
        # 高度参数配置
        self.ALTITUDE_RANGE = tuple(config.get('altitude_range', [6000, 13000]))
        
        logger.info(f"RandomScenarioModifier 初始化完成:")
        logger.info(f"  飞机平台: {self.AIRCRAFT_PLATFORMS}")
        logger.info(f"  速度范围: 北(正){self.VELOCITY_NORTH_POSITIVE}(负){self.VELOCITY_NORTH_NEGATIVE} 东(正){self.VELOCITY_EAST_POSITIVE}(负){self.VELOCITY_EAST_NEGATIVE} fps")
        logger.info(f"  位置范围: 纬度{self.LATITUDE_RANGE} 蓝机经度{self.LONGITUDE_BLUE} 红机经度{self.LONGITUDE_RED}")
        logger.info(f"  高度范围: {self.ALTITUDE_RANGE} m")
    
    def backup_original_file(self):
        """备份原始场景文件"""
        try:
            if not os.path.exists(self.backup_path):
                shutil.copy2(self.scenario_path, self.backup_path)
                logger.info(f"已备份原始场景文件到: {self.backup_path}")
            return True
        except Exception as e:
            logger.error(f"备份文件失败: {e}")
            return False
    
    def calculate_heading_from_velocity(self, north_vel: float, east_vel: float) -> float:
        """
        根据NED坐标系的北向和东向速度计算航向角
        
        Args:
            north_vel: 北向速度
            east_vel: 东向速度
            
        Returns:
            航向角(度数，0-360范围)
        """
        # 计算航向角（弧度）
        heading_rad = math.atan2(east_vel, north_vel)
        
        # 转换为度数
        heading_deg = math.degrees(heading_rad)
        
        # 确保在0-360范围内
        if heading_deg < 0:
            heading_deg += 360
            
        return heading_deg
    
    def generate_aircraft_params(self, aircraft_name: str) -> Dict:
        """
        生成单架飞机的随机参数
        
        Args:
            aircraft_name: 飞机名称 (blue_1, blue_2, red_1, red_2)
            
        Returns:
            包含飞机参数的字典
        """
        is_blue_side = aircraft_name.startswith('blue')
        
        # 随机生成速度分量 - 支持正负双向
        # 北向速度：随机选择正向或负向
        if random.choice([True, False]):
            north_vel = random.uniform(*self.VELOCITY_NORTH_POSITIVE)
        else:
            north_vel = random.uniform(*self.VELOCITY_NORTH_NEGATIVE)
        
        # 东向速度：随机选择正向或负向
        if random.choice([True, False]):
            east_vel = random.uniform(*self.VELOCITY_EAST_POSITIVE)
        else:
            east_vel = random.uniform(*self.VELOCITY_EAST_NEGATIVE)
        
        # 根据速度计算匹配的航向角
        heading = self.calculate_heading_from_velocity(north_vel, east_vel)
        
        # 生成位置参数
        latitude = random.uniform(*self.LATITUDE_RANGE)
        if is_blue_side:
            longitude = random.uniform(*self.LONGITUDE_BLUE)
        else:
            longitude = random.uniform(*self.LONGITUDE_RED)
        
        # 生成高度
        altitude = random.uniform(*self.ALTITUDE_RANGE)
        
        params = {
            'velocity': (north_vel, east_vel, self.VELOCITY_DOWN),
            'position': (latitude, longitude),
            'altitude': altitude,
            'heading': heading,
            'aircraft_name': aircraft_name
        }
        
        logger.debug(f"{aircraft_name} 参数: 速度({north_vel:.1f}, {east_vel:.1f}, 0), "
                    f"位置({latitude:.2f}, {longitude:.2f}), "
                    f"高度{altitude:.0f}m, 航向{heading:.1f}°")
        
        return params
    
    def update_aircraft_in_scenario(self, content: str, params: Dict) -> str:
        """
        更新场景文件中指定飞机的参数
        
        Args:
            content: 场景文件内容
            params: 飞机参数字典
            
        Returns:
            更新后的文件内容
        """
        aircraft_name = params['aircraft_name']
        
        # 构建平台块的正则表达式模式
        platform_pattern = rf'(platform\s+{aircraft_name}\s+.*?)(end_platform)'
        
        def update_platform_block(match):
            platform_content = match.group(1)
            end_tag = match.group(2)
            
            # 更新速度参数
            velocity_pattern = r'(\s+six_dof_set_velocity_ned_fps\s+)[0-9.-]+\s+[0-9.-]+\s+[0-9.-]+'
            new_velocity = f"\\g<1>{params['velocity'][0]:.1f} {params['velocity'][1]:.1f} {params['velocity'][2]:.1f}"
            platform_content = re.sub(velocity_pattern, new_velocity, platform_content)
            
            # 更新位置参数
            position_pattern = r'(\s+six_dof_position\s+)[0-9.-]+\s+[0-9.-]+'
            new_position = f"\\g<1>{params['position'][0]:.2f} {params['position'][1]:.2f}"
            platform_content = re.sub(position_pattern, new_position, platform_content)
            
            # 更新高度参数
            altitude_pattern = r'(\s+six_dof_alt\s+)[0-9.-]+(\s+m(?:eter)?)'
            new_altitude = f"\\g<1>{params['altitude']:.0f}\\g<2>"
            platform_content = re.sub(altitude_pattern, new_altitude, platform_content)
            
            # 更新航向角参数
            heading_pattern = r'(\s+six_dof_ned_heading\s+)[0-9.-]+(\s+deg)'
            new_heading = f"\\g<1>{params['heading']:.1f}\\g<2>"
            platform_content = re.sub(heading_pattern, new_heading, platform_content)
            
            return platform_content + end_tag
        
        # 执行替换
        updated_content = re.sub(platform_pattern, update_platform_block, content, flags=re.DOTALL)
        
        return updated_content
    
    def randomize_scenario(self) -> Dict[str, Dict]:
        """
        随机化整个战场想定
        
        Returns:
            所有飞机的参数字典
        """
        # 备份原始文件
        if not self.backup_original_file():
            return {}
        
        try:
            # 读取场景文件
            with open(self.scenario_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 为每架飞机生成随机参数
            all_params = {}
            for aircraft_name in self.AIRCRAFT_PLATFORMS:
                params = self.generate_aircraft_params(aircraft_name)
                all_params[aircraft_name] = params
                
                # 更新文件内容
                content = self.update_aircraft_in_scenario(content, params)
            
            # 写入更新后的文件
            with open(self.scenario_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"已成功随机化战场想定文件: {self.scenario_path}")
            return all_params
            
        except Exception as e:
            logger.error(f"随机化战场想定失败: {e}")
            return {}
    
    def restore_backup(self):
        """恢复备份文件"""
        try:
            if os.path.exists(self.backup_path):
                shutil.copy2(self.backup_path, self.scenario_path)
                logger.info(f"已恢复原始场景文件")
                return True
        except Exception as e:
            logger.error(f"恢复备份文件失败: {e}")
        return False

if __name__ == "__main__":
    gen = ExperienceGenerator(cfg_path="configs/generators.yaml")
    gen.run()
    logger.info("All episodes finished!")
    # 如需全部加载：data = gen.load_all()
