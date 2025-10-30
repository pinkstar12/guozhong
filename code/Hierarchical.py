import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math

# ====================== 分层强化学习配置 ======================
@dataclass
class HierarchicalRLConfig:
    """分层强化学习配置参数"""
    # 高层决策参数
    high_level_lr: float = 0.001
    high_level_epsilon: float = 0.1
    high_level_gamma: float = 0.95
    high_level_update_frequency: int = 10
    
    # 低层决策参数  
    low_level_lr: float = 0.0005
    low_level_epsilon: float = 0.2
    low_level_gamma: float = 0.9
    
    # 网络结构参数
    high_level_hidden_dim: int = 128
    low_level_hidden_dim: int = 64
    state_dim: int = 32
    action_dim: int = 16
    
    # 经验回放参数
    buffer_size: int = 10000
    batch_size: int = 32
    target_update_frequency: int = 100
    
    # 集群参数
    max_cluster_size: int = 8
    leader_selection_threshold: float = 0.7

class TaskType(Enum):
    """任务类型枚举"""
    AREA_SWEEP = "area_sweep"
    TARGET_ELIMINATION = "target_elimination"  
    FORMATION_DEFENSE = "formation_defense"
    RECONNAISSANCE = "reconnaissance"
    ESCORT = "escort"

class LeadershipRole(Enum):
    """领导角色枚举"""
    LEADER = "leader"
    FOLLOWER = "follower"
    AUTONOMOUS = "autonomous"

# ====================== 经验回放缓冲区 ======================
class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """从缓冲区采样批量经验"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# ====================== 高层决策网络 (领导者) ======================
class HighLevelPolicyNetwork(nn.Module):
    """高层策略网络 - 领导者决策网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(HighLevelPolicyNetwork, self).__init__()
        
        # 态势感知层
        self.situation_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # 任务分配层
        self.task_allocation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # 资源调度层
        self.resource_scheduling_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # 价值估计层
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        print("高层决策网络(领导者)初始化完成")
    
    def forward(self, state):
        """前向传播"""
        # 态势编码
        encoded_state = self.situation_encoder(state)
        
        # 多头输出
        task_logits = self.task_allocation_head(encoded_state)
        resource_logits = self.resource_scheduling_head(encoded_state)
        value = self.value_head(encoded_state)
        
        return {
            'task_allocation': F.softmax(task_logits, dim=-1),
            'resource_scheduling': F.softmax(resource_logits, dim=-1),
            'value': value
        }

# ====================== 低层执行网络 (跟随者) ======================
class LowLevelPolicyNetwork(nn.Module):
    """低层策略网络 - 跟随者执行网络"""
    
    def __init__(self, state_dim: int, goal_dim: int, action_dim: int, hidden_dim: int = 64):
        super(LowLevelPolicyNetwork, self).__init__()
        
        input_dim = state_dim + goal_dim
        
        # 目标理解层
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # 状态编码层
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # 融合执行层
        self.execution_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 价值网络
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        print("低层执行网络(跟随者)初始化完成")
    
    def forward(self, state, goal):
        """前向传播"""
        # 编码目标和状态
        goal_encoded = self.goal_encoder(goal)
        state_encoded = self.state_encoder(state)
        
        # 拼接特征
        combined = torch.cat([goal_encoded, state_encoded], dim=-1)
        
        # 执行策略和价值
        action_logits = self.execution_network(combined)
        value = self.value_network(combined)
        
        return F.softmax(action_logits, dim=-1), value

# ====================== 专家经验系统 ======================
class ExpertKnowledgeSystem:
    """专家经验知识库"""
    
    def __init__(self):
        # 专家规则库
        self.expert_rules = {
            # 态势感知规则
            'threat_assessment': {
                'high_threat': {'min_distance': 0, 'max_distance': 200, 'action_weight': 0.8},
                'medium_threat': {'min_distance': 200, 'max_distance': 500, 'action_weight': 0.5},
                'low_threat': {'min_distance': 500, 'max_distance': float('inf'), 'action_weight': 0.2}
            },
            
            # 任务分配规则
            'task_priority': {
                TaskType.TARGET_ELIMINATION: 0.9,
                TaskType.FORMATION_DEFENSE: 0.7,
                TaskType.AREA_SWEEP: 0.5,
                TaskType.RECONNAISSANCE: 0.6,
                TaskType.ESCORT: 0.8
            },
            
            # 协同规则
            'cooperation_rules': {
                'min_group_size': 2,
                'max_group_size': 4,
                'formation_distance': 150.0,
                'communication_range': 300.0
            }
        }
        
        print("专家经验系统初始化完成")
    
    def get_expert_action_probability(self, state_dict: Dict, task_type: TaskType) -> np.ndarray:
        """基于专家经验给出动作概率分布"""
        action_probs = np.ones(7) * 0.1  # 7种基础动作，默认概率
        
        # 根据威胁等级调整
        threat_level = state_dict.get('threat_level', 0.5)
        if threat_level > 0.7:
            # 高威胁：优先规避和撤退
            action_probs[1] = 0.4  # evade
            action_probs[3] = 0.3  # retreat
        elif threat_level > 0.4:
            # 中威胁：攻击和侧翼
            action_probs[0] = 0.4  # attack
            action_probs[4] = 0.3  # flank
        else:
            # 低威胁：进攻策略
            action_probs[0] = 0.5  # attack
            action_probs[2] = 0.2  # hold
        
        # 根据任务类型调整
        task_weight = self.expert_rules['task_priority'].get(task_type, 0.5)
        action_probs = action_probs * task_weight + (1 - task_weight) * np.ones(7) / 7
        
        # 归一化
        return action_probs / np.sum(action_probs)
    
    def evaluate_leader_capability(self, drone_state: Dict) -> float:
        """评估无人机的领导能力"""
        capability_score = 0.0
        
        # 能量水平 (40%)
        energy_score = drone_state.get('energy', 0.5)
        capability_score += energy_score * 0.4
        
        # 健康度 (30%)
        health_score = drone_state.get('health', 0.5)
        capability_score += health_score * 0.3
        
        # 位置优势 (20%) - 基于高度和中心度
        position = np.array(drone_state.get('position', [0, 0, 0]))
        altitude_advantage = min(position[2] / 500.0, 1.0)  # 高度优势
        capability_score += altitude_advantage * 0.2
        
        # 威胁抵抗能力 (10%)
        threat_resistance = 1.0 - drone_state.get('threat_level', 0.5)
        capability_score += threat_resistance * 0.1
        
        return np.clip(capability_score, 0.0, 1.0)

# ====================== 分层强化学习智能体 ======================
class HierarchicalRLAgent:
    """分层强化学习智能体"""
    
    def __init__(self, config: HierarchicalRLConfig, expert_system: ExpertKnowledgeSystem):
        self.config = config
        self.expert_system = expert_system
        
        # 初始化网络
        self.high_level_policy = HighLevelPolicyNetwork(
            config.state_dim, config.action_dim, config.high_level_hidden_dim
        )
        self.high_level_target = HighLevelPolicyNetwork(
            config.state_dim, config.action_dim, config.high_level_hidden_dim  
        )
        
        self.low_level_policy = LowLevelPolicyNetwork(
            config.state_dim, config.action_dim, config.action_dim, config.low_level_hidden_dim
        )
        self.low_level_target = LowLevelPolicyNetwork(
            config.state_dim, config.action_dim, config.action_dim, config.low_level_hidden_dim
        )
        
        # 优化器
        self.high_level_optimizer = torch.optim.Adam(
            self.high_level_policy.parameters(), lr=config.high_level_lr
        )
        self.low_level_optimizer = torch.optim.Adam(
            self.low_level_policy.parameters(), lr=config.low_level_lr
        )
        
        # 经验缓冲区
        self.high_level_buffer = ReplayBuffer(config.buffer_size)
        self.low_level_buffer = ReplayBuffer(config.buffer_size)
        
        # 训练统计
        self.training_step = 0
        self.high_level_losses = []
        self.low_level_losses = []
        
        print("分层强化学习智能体初始化完成")
    
    def select_leaders(self, drone_states: Dict) -> Dict[str, LeadershipRole]:
        """基于专家经验选择领导者"""
        leadership_assignments = {}
        capability_scores = {}
        
        # 计算每个无人机的领导能力
        for drone_id, state in drone_states.items():
            capability_scores[drone_id] = self.expert_system.evaluate_leader_capability(state)
        
        # 按能力排序
        sorted_drones = sorted(capability_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 选择领导者 (取前25%作为潜在领导者)
        num_leaders = max(1, len(sorted_drones) // 4)
        
        for i, (drone_id, score) in enumerate(sorted_drones):
            if i < num_leaders and score > self.config.leader_selection_threshold:
                leadership_assignments[drone_id] = LeadershipRole.LEADER
            else:
                leadership_assignments[drone_id] = LeadershipRole.FOLLOWER
        
        print(f"选择了 {num_leaders} 个领导者: {[k for k, v in leadership_assignments.items() if v == LeadershipRole.LEADER]}")
        return leadership_assignments
    
    def high_level_decision(self, global_state: torch.Tensor, drone_states: Dict) -> Dict:
        """高层决策 - 任务分配和资源调度"""
        with torch.no_grad():
            outputs = self.high_level_policy(global_state)
        
        # 任务分配决策
        task_allocation = outputs['task_allocation']
        resource_scheduling = outputs['resource_scheduling']
        
        # 为每个无人机分配子目标
        sub_goals = {}
        drone_ids = list(drone_states.keys())
        
        for i, drone_id in enumerate(drone_ids):
            if i < len(task_allocation):
                # 将高层决策转换为子目标
                goal_vector = torch.zeros(self.config.action_dim)
                goal_vector[torch.argmax(task_allocation[i])] = 1.0
                sub_goals[drone_id] = goal_vector
            else:
                # 默认目标
                sub_goals[drone_id] = torch.zeros(self.config.action_dim)
        
        return {
            'sub_goals': sub_goals,
            'resource_allocation': resource_scheduling,
            'global_value': outputs['value']
        }
    
    def low_level_decision(self, state: torch.Tensor, goal: torch.Tensor, drone_id: str, 
                          drone_state: Dict) -> Tuple[int, float]:
        """低层决策 - 具体动作执行"""
        # 获取专家建议
        expert_probs = self.expert_system.get_expert_action_probability(
            drone_state, TaskType.TARGET_ELIMINATION
        )
        expert_probs_tensor = torch.FloatTensor(expert_probs)
        
        # 神经网络策略
        with torch.no_grad():
            nn_probs, value = self.low_level_policy(state.unsqueeze(0), goal.unsqueeze(0))
            nn_probs = nn_probs.squeeze(0)
            value = value.squeeze(0)
        
        # 专家经验与神经网络融合 (加权平均)
        expert_weight = 0.3  # 专家经验权重
        combined_probs = expert_weight * expert_probs_tensor + (1 - expert_weight) * nn_probs
        
        # ε-贪心策略
        if random.random() < self.config.low_level_epsilon:
            action = random.randint(0, len(combined_probs) - 1)
        else:
            action = torch.argmax(combined_probs).item()
        
        return action, value.item()
    
    def train_high_level(self):
        """训练高层策略网络"""
        if len(self.high_level_buffer) < self.config.batch_size:
            return
        
        # 采样批量经验
        states, actions, rewards, next_states, dones = self.high_level_buffer.sample(self.config.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # 当前Q值
        current_outputs = self.high_level_policy(states)
        current_q_values = current_outputs['value'].squeeze()
        
        # 目标Q值
        with torch.no_grad():
            next_outputs = self.high_level_target(next_states)
            next_q_values = next_outputs['value'].squeeze()
            target_q_values = rewards + (self.config.high_level_gamma * next_q_values * ~dones)
        
        # 计算损失
        q_loss = F.mse_loss(current_q_values, target_q_values)
        
        # 策略损失 (Actor-Critic)
        advantages = (target_q_values - current_q_values).detach()
        policy_loss = -torch.mean(advantages * torch.log(current_outputs['task_allocation'] + 1e-8))
        
        total_loss = q_loss + policy_loss
        
        # 反向传播
        self.high_level_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.high_level_policy.parameters(), 1.0)
        self.high_level_optimizer.step()
        
        self.high_level_losses.append(total_loss.item())
    
    def train_low_level(self):
        """训练低层策略网络"""
        if len(self.low_level_buffer) < self.config.batch_size:
            return
        
        # 采样批量经验
        states, actions, rewards, next_states, dones = self.low_level_buffer.sample(self.config.batch_size)
        
        # 这里需要从经验中提取goal信息，简化处理
        batch_size = len(states)
        dummy_goals = torch.zeros(batch_size, self.config.action_dim)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)  
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # 当前策略和价值
        current_probs, current_values = self.low_level_policy(states, dummy_goals)
        current_values = current_values.squeeze()
        
        # 目标价值
        with torch.no_grad():
            _, next_values = self.low_level_target(next_states, dummy_goals)
            next_values = next_values.squeeze()
            target_values = rewards + (self.config.low_level_gamma * next_values * ~dones)
        
        # 价值损失
        value_loss = F.mse_loss(current_values, target_values)
        
        # 策略损失
        advantages = (target_values - current_values).detach()
        action_probs = current_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -torch.mean(advantages * torch.log(action_probs + 1e-8))
        
        total_loss = value_loss + policy_loss
        
        # 反向传播
        self.low_level_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.low_level_policy.parameters(), 1.0)
        self.low_level_optimizer.step()
        
        self.low_level_losses.append(total_loss.item())
    
    def update_target_networks(self):
        """更新目标网络"""
        if self.training_step % self.config.target_update_frequency == 0:
            self.high_level_target.load_state_dict(self.high_level_policy.state_dict())
            self.low_level_target.load_state_dict(self.low_level_policy.state_dict())
            print(f"更新目标网络 (步骤: {self.training_step})")

# ====================== 分层决策集成系统 ======================
class HierarchicalDecisionSystem:
    """分层决策集成系统 - 连接原有系统与分层强化学习"""
    
    def __init__(self, combat_system, config: HierarchicalRLConfig = None):
        self.combat_system = combat_system
        self.config = config or HierarchicalRLConfig()
        
        # 初始化专家系统和强化学习智能体
        self.expert_system = ExpertKnowledgeSystem()
        self.rl_agent = HierarchicalRLAgent(self.config, self.expert_system)
        
        # 状态历史
        self.state_history = []
        self.reward_history = []
        
        print("分层决策集成系统初始化完成")
    
    def extract_global_state(self, blue_observations: Dict) -> torch.Tensor:
        """提取全局状态特征"""
        features = []
        
        # 统计特征
        num_drones = len(blue_observations)
        total_energy = sum(obs['self_state']['energy'] for obs in blue_observations.values())
        total_health = sum(obs['self_state']['health'] for obs in blue_observations.values())
        avg_threat = sum(obs['self_state']['threat_level'] for obs in blue_observations.values()) / num_drones
        
        # 位置特征
        positions = [obs['self_state']['position'] for obs in blue_observations.values()]
        center = np.mean(positions, axis=0) if positions else [0, 0, 0]
        spread = np.std(positions, axis=0) if len(positions) > 1 else [0, 0, 0]
        
        # 敌情特征  
        total_enemies = sum(len(obs['enemies']) for obs in blue_observations.values())
        
        # 组合特征向量
        features.extend([
            num_drones / 10.0,  # 归一化
            total_energy / num_drones,
            total_health / num_drones,
            avg_threat,
            *center,
            *spread,
            total_enemies / 10.0
        ])
        
        # 填充到固定维度
        while len(features) < self.config.state_dim:
            features.append(0.0)
        
        return torch.FloatTensor(features[:self.config.state_dim])
    
    def extract_local_state(self, observation: Dict, drone_id: str) -> torch.Tensor:
        """提取单个无人机的局部状态"""
        self_state = observation['self_state']
        
        features = [
            self_state['energy'],
            self_state['health'], 
            self_state['threat_level'],
            *self_state['position'],
            *self_state.get('velocity', [0, 0, 0]),
            len(observation.get('allies', [])),
            len(observation.get('enemies', []))
        ]
        
        # 最近敌人信息
        if observation.get('enemies'):
            nearest_enemy = min(observation['enemies'], 
                              key=lambda e: np.linalg.norm(np.array(e['position']) - np.array(self_state['position'])))
            enemy_pos = nearest_enemy['position']
            enemy_dist = np.linalg.norm(np.array(enemy_pos) - np.array(self_state['position']))
            features.extend([*enemy_pos, enemy_dist / 1000.0])  # 归一化距离
        else:
            features.extend([0, 0, 0, 1.0])  # 无敌人时的默认值
        
        # 填充到固定维度
        while len(features) < self.config.state_dim:
            features.append(0.0)
            
        return torch.FloatTensor(features[:self.config.state_dim])
    
    def get_hierarchical_decisions(self) -> set:
        """获取分层决策结果"""
        print("\n--- 开始分层强化学习决策 ---")
        
        # 1. 获取观测数据
        blue_observations = self.combat_system.environment.get_all_observations('blue')
        if not blue_observations:
            return set()
        
        # 2. 选择领导者
        blue_states = self.combat_system.environment.blue_drones
        leadership_assignments = self.rl_agent.select_leaders(blue_states)
        
        # 3. 高层决策 (由领导者执行)
        global_state = self.extract_global_state(blue_observations)
        high_level_output = self.rl_agent.high_level_decision(global_state, blue_states)
        
        # 4. 低层决策 (所有单位执行具体动作)
        final_decisions = set()
        action_mapping = ['attack', 'evade', 'hold', 'retreat', 'flank', 'climb', 'dive']
        
        for drone_id, obs in blue_observations.items():
            # 提取局部状态
            local_state = self.extract_local_state(obs, drone_id)
            
            # 获取子目标
            sub_goal = high_level_output['sub_goals'].get(drone_id, torch.zeros(self.config.action_dim))
            
            # 低层决策
            action_idx, value = self.rl_agent.low_level_decision(
                local_state, sub_goal, drone_id, obs['self_state']
            )
            
            # 转换为策略字符串
            strategy = action_mapping[action_idx]
            
            # 确定参数 (如果需要目标)
            strategies_requiring_target = {'attack', 'flank'}
            if strategy in strategies_requiring_target and obs.get('enemies'):
                # 选择最近的敌人作为目标
                nearest_enemy = min(obs['enemies'], 
                                  key=lambda e: np.linalg.norm(np.array(e['position']) - np.array(obs['self_state']['position'])))
                params = nearest_enemy['id']
            else:
                params = None
            
            decision = (drone_id, strategy, params)
            final_decisions.add(decision)
            
            role = leadership_assignments.get(drone_id, LeadershipRole.FOLLOWER)
            print(f"{role.value} {drone_id}: {strategy} (价值评估: {value:.3f})")
        
        # 5. 存储状态用于训练
        self.state_history.append((global_state, blue_observations))
        
        return final_decisions
    
    def update_rewards_and_train(self, drone_states_before: Dict, drone_states_after: Dict):
        """根据执行结果更新奖励并训练网络"""
        if len(self.state_history) < 2:
            return
            
        # 计算奖励
        reward = self.calculate_reward(drone_states_before, drone_states_after)
        self.reward_history.append(reward)
        
        print(f"当前奖励: {reward:.3f}")
        
        # 构造训练样本 (简化版)
        if len(self.state_history) >= 2 and len(self.reward_history) >= 1:
            prev_global_state = self.state_history[-2][0]
            curr_global_state = self.state_history[-1][0] 
            
            # 添加高层经验
            self.rl_agent.high_level_buffer.push(
                prev_global_state.numpy(),
                0,  # 简化的动作
                reward,
                curr_global_state.numpy(),
                False  # 简化的done标志
            )
            
            # 训练网络
            self.rl_agent.train_high_level()
            self.rl_agent.train_low_level()
            
            # 更新训练步数
            self.rl_agent.training_step += 1
            self.rl_agent.update_target_networks()
    
    def calculate_reward(self, states_before: Dict, states_after: Dict) -> float:
        """计算奖励函数"""
        reward = 0.0
        
        # 生存奖励
        survived_drones = len(states_after)
        total_drones = len(states_before)
        survival_rate = survived_drones / max(total_drones, 1)
        reward += survival_rate * 10.0
        
        # 能量效率奖励
        if states_after:
            avg_energy_after = sum(drone['energy'] for drone in states_after.values()) / len(states_after)
            reward += avg_energy_after * 5.0
        
        # 任务完成奖励 (基于敌人消灭情况)
        enemy_states_before = self.combat_system.environment.red_drones
        enemy_states_after = len([e for e in enemy_states_before.values() if e.get('health', 1.0) > 0])
        enemies_eliminated = len(enemy_states_before) - enemy_states_after
        reward += enemies_eliminated * 20.0
        
        # 协同性奖励 (基于集群紧密度)
        if len(states_after) > 1:
            positions = [np.array(drone['position']) for drone in states_after.values()]
            center = np.mean(positions, axis=0)
            avg_distance_to_center = np.mean([np.linalg.norm(pos - center) for pos in positions])
            cohesion_reward = max(0, 300 - avg_distance_to_center) / 300 * 3.0
            reward += cohesion_reward
        
        return reward
    
    def get_training_statistics(self) -> Dict:
        """获取训练统计信息"""
        return {
            'high_level_losses': self.rl_agent.high_level_losses[-100:],  # 最近100步
            'low_level_losses': self.rl_agent.low_level_losses[-100:],
            'avg_reward': np.mean(self.reward_history[-50:]) if self.reward_history else 0.0,
            'training_steps': self.rl_agent.training_step,
            'buffer_sizes': {
                'high_level': len(self.rl_agent.high_level_buffer),
                'low_level': len(self.rl_agent.low_level_buffer)
            }
        }

# ====================== 增强版机动集成器 ======================
class HierarchicalManeuverIntegrator:
    """分层决策与机动系统集成器"""
    
    def __init__(self, combat_system, maneuver_config=None, rl_config=None):
        from AircraftManeuvering import AircraftManeuvering
        
        self.combat_system = combat_system
        self.maneuvering = AircraftManeuvering(maneuver_config)
        
        # 初始化分层决策系统
        self.hierarchical_decision_system = HierarchicalDecisionSystem(
            combat_system, rl_config
        )
        
        # 仿真参数
        self.simulation_time = 0.0
        self.time_step = 0.1
        self.training_interval = 5  # 每5步训练一次
        self.episode_count = 0
        
        print("分层决策-机动集成系统初始化完成")
    
    def run_hierarchical_simulation(self, simulation_duration: float = 30.0, 
                                  training_episodes: int = 10):
        """运行分层强化学习仿真"""
        print(f"\n=== 开始分层强化学习仿真 ===")
        print(f"仿真持续时间: {simulation_duration}s, 训练回合数: {training_episodes}")
        
        for episode in range(training_episodes):
            print(f"\n--- 第 {episode + 1} 回合 ---")
            self.episode_count = episode + 1
            self.simulation_time = 0.0
            
            # 重置环境 (重新生成随机态势)
            self._reset_environment()
            
            step_count = 0
            while self.simulation_time < simulation_duration:
                step_count += 1
                print(f"\n时间: {self.simulation_time:.1f}s (步骤 {step_count})")
                
                # 记录执行前状态
                blue_states_before = self.combat_system.environment.blue_drones.copy()
                
                # 1. 分层强化学习决策
                hierarchical_decisions = self.hierarchical_decision_system.get_hierarchical_decisions()
                
                if not hierarchical_decisions:
                    print("仿真结束：无可用单位")
                    break
                
                # 2. 执行机动
                blue_states = self.combat_system.environment.blue_drones
                red_states = self.combat_system.environment.red_drones
                
                updated_blue_states = self.maneuvering.execute_maneuvers(
                    hierarchical_decisions, blue_states, red_states, self.time_step
                )
                
                # 3. 更新环境
                self.combat_system.environment.blue_drones = updated_blue_states
                
                # 4. 计算奖励并训练
                if step_count % self.training_interval == 0:
                    self.hierarchical_decision_system.update_rewards_and_train(
                        blue_states_before, updated_blue_states
                    )
                
                # 5. 显示状态
                if step_count % 10 == 0:  # 每10步显示一次详细状态
                    self._print_detailed_status(updated_blue_states, step_count)
                else:
                    self._print_brief_status(updated_blue_states)
                
                # 6. 检查终止条件
                if not updated_blue_states or not red_states:
                    print("仿真结束：一方全部消失")
                    break
                
                self.simulation_time += self.time_step
            
            # 回合结束统计
            self._print_episode_summary(episode + 1)
        
        print(f"\n=== 分层强化学习仿真完成 ===")
        self._print_final_statistics()
    
    def _reset_environment(self):
        """重置环境到初始状态"""
        # 重新初始化无人机状态
        self.combat_system.environment.blue_drones = self.combat_system.environment._init_drones(5, 'blue')
        self.combat_system.environment.red_drones = self.combat_system.environment._init_drones(5, 'red')
        print("环境已重置")
    
    def _print_detailed_status(self, drone_states: Dict, step: int):
        """打印详细状态信息"""
        print(f"\n--- 详细状态报告 (步骤 {step}) ---")
        
        for drone_id, state in drone_states.items():
            pos = state['position']
            vel = state.get('velocity', [0, 0, 0])
            speed = np.linalg.norm(vel) if hasattr(vel, '__len__') else 0
            
            print(f"{drone_id}: 位置({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) "
                  f"速度:{speed:.1f}m/s 能量:{state['energy']:.2f} "
                  f"健康:{state['health']:.2f} 策略:{state.get('strategy', 'unknown')}")
        
        # 显示训练统计
        training_stats = self.hierarchical_decision_system.get_training_statistics()
        print(f"训练统计: 步数={training_stats['training_steps']}, "
              f"平均奖励={training_stats['avg_reward']:.3f}")
    
    def _print_brief_status(self, drone_states: Dict):
        """打印简要状态"""
        active_drones = len(drone_states)
        avg_energy = sum(d['energy'] for d in drone_states.values()) / max(active_drones, 1)
        avg_health = sum(d['health'] for d in drone_states.values()) / max(active_drones, 1)
        
        print(f"活跃:{active_drones} 平均能量:{avg_energy:.2f} 平均健康:{avg_health:.2f}")
    
    def _print_episode_summary(self, episode: int):
        """打印回合总结"""
        print(f"\n--- 第 {episode} 回合总结 ---")
        
        blue_survivors = len(self.combat_system.environment.blue_drones)
        red_survivors = len(self.combat_system.environment.red_drones)
        
        print(f"蓝方存活: {blue_survivors}/5")
        print(f"红方存活: {red_survivors}/5") 
        
        # 训练统计
        training_stats = self.hierarchical_decision_system.get_training_statistics()
        if training_stats['high_level_losses']:
            recent_hl_loss = training_stats['high_level_losses'][-1]
            print(f"最近高层损失: {recent_hl_loss:.4f}")
        
        if training_stats['low_level_losses']:
            recent_ll_loss = training_stats['low_level_losses'][-1]
            print(f"最近低层损失: {recent_ll_loss:.4f}")
        
        print(f"累计奖励: {training_stats['avg_reward']:.3f}")
    
    def _print_final_statistics(self):
        """打印最终统计"""
        print("\n--- 最终训练统计 ---")
        
        training_stats = self.hierarchical_decision_system.get_training_statistics()
        
        print(f"总训练步数: {training_stats['training_steps']}")
        print(f"高层经验缓冲区: {training_stats['buffer_sizes']['high_level']}")
        print(f"低层经验缓冲区: {training_stats['buffer_sizes']['low_level']}")
        
        if training_stats['high_level_losses']:
            avg_hl_loss = np.mean(training_stats['high_level_losses'])
            print(f"平均高层损失: {avg_hl_loss:.4f}")
        
        if training_stats['low_level_losses']:
            avg_ll_loss = np.mean(training_stats['low_level_losses'])
            print(f"平均低层损失: {avg_ll_loss:.4f}")
        
        print(f"最终平均奖励: {training_stats['avg_reward']:.3f}")

# ====================== 使用示例 ======================
def create_hierarchical_system():
    """创建分层强化学习系统的工厂函数"""
    from DroneCombatSystem import DroneCombatSystem
    from AircraftManeuvering import ManeuverConfig
    
    # 创建作战系统
    combat_system = DroneCombatSystem(red_invasion_mode='concentrated')
    
    # 配置参数
    maneuver_config = ManeuverConfig(
        max_speed=80.0,
        max_climb_rate=15.0, 
        combat_radius=200.0,
        evasion_radius=250.0
    )
    
    rl_config = HierarchicalRLConfig(
        high_level_lr=0.001,
        low_level_lr=0.0005,
        high_level_epsilon=0.1,
        low_level_epsilon=0.2,
        state_dim=32,
        action_dim=16
    )
    
    # 创建分层集成器
    hierarchical_integrator = HierarchicalManeuverIntegrator(
        combat_system, maneuver_config, rl_config
    )
    
    return hierarchical_integrator

if __name__ == "__main__":
    # 演示分层强化学习系统
    print("=== 分层强化学习多弹协同制导系统 ===")
    
    # 创建系统
    hierarchical_system = create_hierarchical_system()
    
    # 运行训练仿真
    hierarchical_system.run_hierarchical_simulation(
        simulation_duration=20.0,
        training_episodes=5
    )
    
    print("\n系统演示完成")