import torch
import numpy as np
import math

# 地球半径常量 (km) - 使用球面近似模型
EARTH_RADIUS = 6371.0

def llh_to_ned(lat, lon, alt, ref_llh=(0, 0, 0)):
    """
    经纬高转NED坐标系（球面近似）
    
    Args:
        lat (float): 纬度 (度)
        lon (float): 经度 (度)
        alt (float): 高度 (米，海拔高度)
        ref_llh (tuple): 参考点 (纬度, 经度, 高度)，默认为(0, 0, 0)
    
    Returns:
        tuple: (北向距离(km), 天向高度(米), 东向距离(km))
    """
    ref_lat, ref_lon, ref_alt = ref_llh
    
    # 转换为弧度进行计算
    lat_rad = math.radians(lat)
    ref_lat_rad = math.radians(ref_lat)
    
    # 计算北向距离 (km)
    north = EARTH_RADIUS * math.radians(lat - ref_lat)
    
    # 计算东向距离 (km) - 考虑纬度对经度距离的影响
    east = EARTH_RADIUS * math.radians(lon - ref_lon) * math.cos(lat_rad)
    
    # 高度差 (米)
    up = alt - ref_alt
    
    return north, up, east

def heading_180_to_360(input_heading):
    """
    航向角转换：从-180~180度转换到0~360度
    
    Args:
        input_heading (float): 输入航向角 (-180~180度)
                              0表示正北，90表示正东，-90表示正西
    
    Returns:
        float: 输出航向角 (0~360度)
               0表示正北，90表示正东，270表示正西
    """
    output_heading = input_heading % 360
    return output_heading

def ned_to_llh(north, east, up, ref_llh=(0, 0, 0)):
    """
    NED坐标转经纬高坐标（球面近似）
    
    Args:
        north (float): 北向距离 (km)
        east (float): 东向距离 (km)
        up (float): 天向高度 (米)
        ref_llh (tuple): 参考点 (纬度, 经度, 高度)，默认为(0, 0, 0)
    
    Returns:
        tuple: (纬度(度), 经度(度), 高度(米))
    """
    ref_lat, ref_lon, ref_alt = ref_llh
    
    # 计算纬度
    lat = ref_lat + math.degrees(north / EARTH_RADIUS)
    
    # 计算经度 - 考虑纬度对经度距离的影响
    lat_rad = math.radians(lat)
    lon = ref_lon + math.degrees(east / (EARTH_RADIUS * math.cos(lat_rad)))
    
    # 计算高度
    alt = ref_alt + up
    
    return lat, lon, alt

def heading_360_to_180(input_heading):
    """
    航向角转换：从0~360度转换到-180~180度
    
    Args:
        input_heading (float): 输入航向角 (0~360度)
    
    Returns:
        float: 输出航向角 (-180~180度)
    """
    output_heading = (input_heading + 180) % 360 - 180
    return output_heading

def inverse_transform_actions(actions, ref_llh=(0, 0, 0)):
    """
    将专家系统输出的动作转换为环境所需格式
    
    Args:
        actions (List[dict]): 专家系统输出的动作列表
                             位置格式：[北向距离(km), 东向距离(km)]
                             航向角格式：0~360度
        ref_llh (tuple): 参考点 (纬度, 经度, 高度)，默认为(0, 0, 0)
    
    Returns:
        List[dict]: 转换后的动作列表
                   位置格式：[纬度, 经度]
                   航向角格式：0~360度
    """
    transformed_actions = []
    for act in actions:
        transformed = act.copy()
        
        # 转换位置：NED坐标转经纬度
        if 'position' in act:
            north, east = act['position']
            lat, lon, _ = ned_to_llh(north, east, 0, ref_llh)  # 高度保持不变
            transformed['position'] = [lat, lon]
        
        # # 转换航向：0~360度转-180~180度
        # if 'heading' in act:
        #     transformed['heading'] = heading_360_to_180(act['heading'])
        
        transformed_actions.append(transformed)
    return transformed_actions

def transform_state(state):
    """
    输入数据预处理：转换状态字典中的坐标和航向角
    
    Args:
        state (dict): 输入状态字典，包含我方飞机、敌方飞机和导弹信息
                     位置格式：[纬度, 经度]
                     航向角格式：-180~180度
    
    Returns:
        dict: 转换后的状态字典
              位置格式：[北向距离(km), 东向距离(km)]  # x-z平面
              航向角格式：0~360度
    """
    # 深拷贝状态字典以避免修改原始数据
    transformed_state = {
        'our_aircrafts': [],
        'enemies': [],
        'missiles': []
    }
    
    # 转换我方飞机数据
    for aircraft in state.get('our_aircrafts', []):
        transformed_aircraft = aircraft.copy()
        if 'position' in aircraft and 'height' in aircraft:
            lat, lon = aircraft['position']
            north, up, east = llh_to_ned(lat, lon, aircraft['height'])
            transformed_aircraft['position'] = np.array([north, east])  # x-z平面 (北-东)
        
        if 'heading' in aircraft:
            transformed_aircraft['heading'] = heading_180_to_360(aircraft['heading'])
        
        transformed_state['our_aircrafts'].append(transformed_aircraft)
    
    # 转换敌方飞机数据
    for enemy in state.get('enemies', []):
        transformed_enemy = enemy.copy()
        if 'position' in enemy and 'height' in enemy:
            lat, lon = enemy['position']
            north, up, east = llh_to_ned(lat, lon, enemy['height'])
            transformed_enemy['position'] = np.array([north, east])  # x-z平面 (北-东)
        
        if 'heading' in enemy:
            transformed_enemy['heading'] = heading_180_to_360(enemy['heading'])
        
        transformed_state['enemies'].append(transformed_enemy)
    
    # 转换导弹数据
    for missile in state.get('missiles', []):
        transformed_missile = missile.copy()
        if 'position' in missile and 'height' in missile:
            lat, lon = missile['position']
            north, up, east = llh_to_ned(lat, lon, missile['height'])
            transformed_missile['position'] = [north , east ]  # x-z平面 (北-东)
        
        transformed_state['missiles'].append(transformed_missile)
    
    return transformed_state

def normalize(x, min_v, max_v):
    """Min-max归一化，支持 numpy 或 torch.Tensor"""
    if isinstance(x, np.ndarray):
        return (x - min_v) / (max_v - min_v + 1e-8)
    else:  # torch.Tensor
        return (x - torch.from_numpy(min_v).to(x.device)) / (torch.from_numpy(max_v - min_v + 1e-8).to(x.device))

def denormalize(x, min_v, max_v):
    """Min-max反归一化，支持 numpy 或 torch.Tensor"""
    if isinstance(x, np.ndarray):
        return x * (max_v - min_v) + min_v
    else:  # torch.Tensor
        return x * (torch.from_numpy(max_v - min_v).to(x.device)) + torch.from_numpy(min_v).to(x.device)
# 示例使用方法
if __name__ == "__main__":
    # 示例1：单个坐标转换
    print("=== 坐标转换示例 ===")
    lat, lon, alt = 30.0, 120.0, 5000  # 纬度30度，经度120度，高度5000米
    north, up, east = llh_to_ned(lat, lon, alt)
    print(f"输入: 纬度{lat}°, 经度{lon}°, 高度{alt}m")
    print(f"输出: 北向{north:.2f}km, 天向{up}m, 东向{east:.2f}km")
    
    # 示例2：航向角转换
    print("\n=== 航向角转换示例 ===")
    headings = [-180, -90, 0, 90, 180]
    for h in headings:
        converted = heading_180_to_360(h)
        print(f"{h}° -> {converted}°")
    
    # 示例3：完整状态转换
    print("\n=== 状态转换示例 ===")
    sample_state = {
        'our_aircrafts': [
            {
                'speed': 250.0,
                'height': 5000.0,
                'heading': -90.0,  # 正西方向
                'position': np.array([30.0, 120.0])  # 纬度30度，经度120度
            }
        ],
        'enemies': [
            {
                'speed': 300.0,
                'height': 6000.0,
                'heading': 180.0,  # 正南方向
                'position': np.array([35.0, 125.0])  # 纬度35度，经度125度
            }
        ],
        'missiles': [
            {
                'position': [32.0, 122.0],  # 纬度32度，经度122度
                'height': 3000.0,
                'speed': 800.0,
                'target': 1
            }
        ]
    }
    
    transformed = transform_state(sample_state)
    print("转换后的我方飞机位置:", transformed['our_aircrafts'][0]['position'])
    print("转换后的我方飞机航向:", transformed['our_aircrafts'][0]['heading'])
    print("转换后的敌方飞机位置:", transformed['enemies'][0]['position'])
    print("转换后的敌方飞机航向:", transformed['enemies'][0]['heading'])
    print("转换后的导弹位置:", transformed['missiles'][0]['position'])


"""
工具函数模块
包含TD3算法所需的基本工具函数
"""
import torch
import numpy as np
import torch.nn.functional as F
import pickle

def identity(x):
    """恒等函数"""
    return x


def expand_action_dimensions(action_array, num_agents, original_action_dim, target_action_dim):
    """
    将3维动作扩展为4维动作
    转换规则：
    - 如果第一位出现-1，则全部为0
    - 当前三位在正常范围内时，第四位一直置1
    
    Args:
        action_array: 原始动作数组，形状为(num_agents * original_action_dim,)
        num_agents: 智能体数量
        original_action_dim: 原始动作维度（通常为3）
        target_action_dim: 目标动作维度（通常为4）
    
    Returns:
        expanded_action: 扩展后的动作数组，形状为(num_agents * target_action_dim,)
    """
    if target_action_dim != 4 or original_action_dim != 3:
        raise ValueError("当前只支持从3维扩展到4维")
    
    # 将动作数组重构为(num_agents, original_action_dim)
    action_reshaped = action_array.reshape(num_agents, original_action_dim)
    
    # 初始化扩展后的动作数组
    expanded_actions = np.zeros((num_agents, target_action_dim), dtype=np.float32)
    
    for agent_id in range(num_agents):
        agent_action = action_reshaped[agent_id]
        
        # 检查第一维是否为-1
        if agent_action[0] == -1.0:
            # 如果第一位出现-1，则全部为0
            expanded_actions[agent_id, 0] = 0.0
            expanded_actions[agent_id, 1] = 0.0
            expanded_actions[agent_id, 2] = 0.0
            expanded_actions[agent_id, 3] = 0.0
        else:
            # 当前三位在正常范围内时，第四位一直置1
            expanded_actions[agent_id, 0] = agent_action[0]
            expanded_actions[agent_id, 1] = agent_action[1]
            expanded_actions[agent_id, 2] = agent_action[2]
            expanded_actions[agent_id, 3] = 1.0
    
    # 重新展平为一维数组
    return expanded_actions.flatten()


def soft_update_from_to(source, target, tau):
    """
    软更新目标网络参数
    target = (1-tau) * target + tau * source
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def fanin_init(tensor):
    """
    Fan-in初始化
    """
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def get_device():
    """
    获取当前设备
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局设备设置
_use_gpu = torch.cuda.is_available()
device = get_device()


def to_tensor(data, device=None):
    """
    将数据转换为tensor
    """
    if device is None:
        device = get_device()
    
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float().to(device)
    elif isinstance(data, torch.Tensor):
        return data.float().to(device)
    else:
        return torch.tensor(data).float().to(device)


def to_numpy(tensor):
    """
    将tensor转换为numpy数组
    """
    return tensor.cpu().detach().numpy()


class ReplayBuffer:
    """
    全局经验回放缓冲区 - 支持多智能体集中式训练
    """
    def __init__(self, capacity, global_obs_dim, global_action_dim, global_reward_dim):
        self.capacity = capacity
        self.global_obs_dim = global_obs_dim
        self.global_action_dim = global_action_dim
        
        # 初始化缓冲区 - 存储全局经验
        self.global_observations = np.zeros((capacity, global_obs_dim), dtype=np.float32)
        self.global_actions = np.zeros((capacity, global_action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, global_reward_dim), dtype=np.float32)  # 修改为2维以支持多智能体奖励
        self.next_global_observations = np.zeros((capacity, global_obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.prev_global_actions = np.zeros((capacity, global_action_dim), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, global_obs, global_action, reward, next_global_obs, done, prev_global_action=None):
        """添加全局经验"""
        self.global_observations[self.ptr] = global_obs
        self.global_actions[self.ptr] = global_action
        self.rewards[self.ptr] = reward
        self.next_global_observations[self.ptr] = next_global_obs
        self.dones[self.ptr] = done
        
        # 处理前一步全局动作
        if prev_global_action is not None:
            self.prev_global_actions[self.ptr] = prev_global_action
        else:
            self.prev_global_actions[self.ptr] = np.zeros(self.global_action_dim, dtype=np.float32)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """采样批次数据"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            'global_observations': to_tensor(self.global_observations[indices]),
            'global_actions': to_tensor(self.global_actions[indices]),
            'rewards': to_tensor(self.rewards[indices]),
            'next_global_observations': to_tensor(self.next_global_observations[indices]),
            'dones': to_tensor(self.dones[indices]),
            'prev_global_actions': to_tensor(self.prev_global_actions[indices])
        }
        
        return batch
    
    def __len__(self):
        return self.size


def get_supervised_data(expert_data_path, obs_padding_dim=128, num_agents=2, action_dim_per_agent=4):
    """
    从专家数据pkl文件中提取监督学习所需的数据
    支持动作维度扩展：将3维动作转换为4维动作，第4维作为特殊标志位
    支持奖励数据的智能体级处理：
    - 旧格式：单个全局奖励，复制给所有智能体
    - 新格式：每个智能体的独立奖励列表，根据num_agents进行截断或补全
    
    Args:
        expert_data_path: pkl文件路径
        obs_padding_dim: 观测数据填充后的维度（默认128）
        num_agents: 智能体个数（默认2）
        action_dim_per_agent: 每个智能体的动作维度（默认4，从3扩展）
        
    Returns:
        obs_data: 填充到指定维度的观测数据数组，形状为(n_samples, obs_padding_dim)
        action_data: 动作数据数组，形状为(n_samples, num_agents * action_dim_per_agent)
        prev_action_data: 前一步动作数据数组，形状为(n_samples, num_agents * action_dim_per_agent)
        reward_data: 奖励数据数组，形状为(n_samples, num_agents)
    """
    # 原始动作维度（从数据中读取的）
    original_action_dim_per_agent = 3
    # 计算原始总动作维度
    original_total_action_dim = num_agents * original_action_dim_per_agent
    # 计算扩展后总动作维度
    total_action_dim = num_agents * action_dim_per_agent
    
    try:
        # 加载pkl文件
        with open(expert_data_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, list):
            raise ValueError("pkl文件应包含字典列表格式的数据")
        
        # 初始化列表存储数据
        obs_list = []
        action_list = []
        prev_action_list = []
        reward_list = []
        
        # 处理每个样本
        for i, sample in enumerate(data):
            if not isinstance(sample, dict):
                raise ValueError(f"样本 {i} 应为字典格式")
            
            # 检查必需的键
            required_keys = ['input_obs', 'output_actions', 'input_last_action', 'input_reward']
            for key in required_keys:
                if key not in sample:
                    raise ValueError(f"样本 {i} 缺少必需的键: {key}")
            
            # 提取并处理obs数据
            obs = sample['input_obs']
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs, dtype=np.float32)
            
            # 填充或截断obs到指定维度
            if len(obs) > obs_padding_dim:
                obs = obs[:obs_padding_dim]
            else:
                obs = np.pad(obs, (0, obs_padding_dim - len(obs)), 'constant', constant_values=0)
            
            obs_list.append(obs)
            
            # 提取动作数据
            action = sample['output_actions']
            if not isinstance(action, np.ndarray):
                action = np.array(action, dtype=np.float32)
            
            # 检查原始动作维度并进行转换
            if len(action) == original_total_action_dim:
                # 需要从3维转换为4维
                expanded_action = expand_action_dimensions(action, num_agents, original_action_dim_per_agent, action_dim_per_agent)
                action_list.append(expanded_action)
            elif len(action) == total_action_dim:
                # 已经是4维，直接使用
                action_list.append(action)
            else:
                raise ValueError(f"样本 {i} 动作维度不匹配: 期望{original_total_action_dim}或{total_action_dim}, 实际{len(action)}")
            
            # 提取前一步动作数据
            prev_action = sample['input_last_action']
            if not isinstance(prev_action, np.ndarray):
                prev_action = np.array(prev_action, dtype=np.float32)
            
            # 检查前一步动作维度并进行转换
            if len(prev_action) == original_total_action_dim:
                # 需要从3维转换为4维
                expanded_prev_action = expand_action_dimensions(prev_action, num_agents, original_action_dim_per_agent, action_dim_per_agent)
                prev_action_list.append(expanded_prev_action)
            elif len(prev_action) == total_action_dim:
                # 已经是4维，直接使用
                prev_action_list.append(prev_action)
            else:
                raise ValueError(f"样本 {i} 前一步动作维度不匹配: 期望{original_total_action_dim}或{total_action_dim}, 实际{len(prev_action)}")
            
            # 提取奖励数据 - 支持单个全局奖励和每个智能体的奖励列表
            reward = sample['input_reward']
            
            if isinstance(reward, (list, np.ndarray)):
                # 新格式：奖励列表（每个智能体一个奖励）
                reward_array = np.array(reward, dtype=np.float32)
                
                if len(reward_array) > num_agents:
                    # 截断到前num_agents个奖励
                    agent_rewards = reward_array[:num_agents]
                elif len(reward_array) < num_agents:
                    # 用0.0补全到num_agents长度
                    agent_rewards = np.pad(reward_array, (0, num_agents - len(reward_array)), 'constant', constant_values=0.0)
                else:
                    # 长度正好匹配
                    agent_rewards = reward_array
            else:
                # 旧格式：单个全局奖励，复制到所有智能体
                single_reward = float(reward)
                agent_rewards = np.full(num_agents, single_reward, dtype=np.float32)
            
            reward_list.append(agent_rewards)
        
        # 转换为numpy数组
        obs_data = np.array(obs_list, dtype=np.float32)
        action_data = np.array(action_list, dtype=np.float32)
        prev_action_data = np.array(prev_action_list, dtype=np.float32)
        reward_data = np.array(reward_list, dtype=np.float32)  # 形状为(n_samples, num_agents)
        
        print(f"成功加载数据: {len(data)} 个样本")
        print(f"obs_data 形状: {obs_data.shape}")
        print(f"action_data 形状: {action_data.shape}")
        print(f"prev_action_data 形状: {prev_action_data.shape}")
        print(f"reward_data 形状: {reward_data.shape}")
        
        return obs_data, action_data, prev_action_data, reward_data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到文件: {expert_data_path}")
    except Exception as e:
        raise RuntimeError(f"处理数据时发生错误: {str(e)}")
