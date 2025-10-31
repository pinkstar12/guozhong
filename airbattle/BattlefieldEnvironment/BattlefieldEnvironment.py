import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from typing import Dict, Tuple, Optional

class BattlefieldEnvironment:
    """
    三维战场态势生成器 (高级随机版)。
    负责生成一个高度随机、充满变数的战场初始快照。
    """

    def __init__(self, blue_drones=5, red_drones=5, area_size=(1000, 1000, 500), red_invasion_mode='concentrated'):
        self.area_size = area_size
        self.red_invasion_mode = red_invasion_mode
        self.blue_drones = self._init_drones(blue_drones, 'blue')
        self.red_drones = self._init_drones(red_drones, 'red')
        print(f"高度随机的三维战场态势已生成: 蓝方{len(self.blue_drones)}架 vs 红方{len(self.red_drones)}架 (入侵模式: {self.red_invasion_mode})")

    def _init_drones(self, count, team):
        """根据队伍和入侵模式，在随机化区域内初始化无人机群"""
        drones = {}
        width, length, height = self.area_size
        for i in range(count):
            position = np.zeros(3)
            if team == 'blue':
                # 蓝方在己方半场的一个广大矩形区域内随机部署
                deploy_area_x = width * 0.8
                deploy_area_y = length * 0.4
                pos_x = np.random.uniform((width - deploy_area_x) / 2, width - (width - deploy_area_x) / 2)
                pos_y = np.random.uniform(0, deploy_area_y)
                pos_z = np.random.uniform(height * 0.2, height * 0.8)
                position = np.array([pos_x, pos_y, pos_z])
            elif team == 'red':
                # 红方从另一侧半场，以更随机的方式入侵
                deploy_area_x = width * 0.9
                deploy_area_y = length * 0.4
                pos_x = np.random.uniform((width - deploy_area_x) / 2, width - (width - deploy_area_x) / 2)
                pos_y = np.random.uniform(length - deploy_area_y, length)
                pos_z = np.random.uniform(height * 0.2, height * 0.8)
                position = np.array([pos_x, pos_y, pos_z])

            # 为无人机生成包含随机性的初始状态属性
            drones[f"{team}_{i}"] = {
                'id': f"{team}_{i}",
                'team': team,
                'position': position,
                'velocity': np.zeros(3),
                'health': random.uniform(0.9, 1.0), # 初始健康度略有浮动
                'energy': random.uniform(0.7, 1.0), # 初始能量随机
                'strategy': 'hold',
                'target': None,
                'altitude': position[2],
                'threat_level': random.uniform(0.1, 0.9) # 初始威胁值随机性更强
            }
        return drones

    def get_all_observations(self, team):
        """获取指定阵营所有单位的观测数据"""
        drones_to_observe = self.blue_drones if team == 'blue' else self.red_drones
        return {drone_id: self.get_observations(drone_id) for drone_id in drones_to_observe}

    def get_observations(self, drone_id):
        """为单个单位生成其观测数据"""
        drone = self.blue_drones.get(drone_id) or self.red_drones.get(drone_id)
        if not drone: return None
        observation = {'self_state': drone.copy(), 'allies': [], 'enemies': []}
        allies_team = self.blue_drones if drone['team'] == 'blue' else self.red_drones
        for id, ally in allies_team.items():
            if id != drone_id:
                observation['allies'].append(ally.copy())
        enemies_team = self.red_drones if drone['team'] == 'blue' else self.blue_drones
        for id, enemy in enemies_team.items():
            observation['enemies'].append(enemy.copy())
        return observation

# ===============================cfy：添加 GATv2 模型类==================================
class DroneGNN(nn.Module):
    """
    无人机图神经网络模型
    使用GATv2进行信息融合，通过注意力机制对邻居状态进行加权聚合
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, dropout=0.1):
        super(DroneGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.heads = heads
        
        # 第一层GAT：多头注意力聚合1跳邻居信息
        self.gat1 = GATv2Conv(
            input_dim, 
            hidden_dim, 
            heads=heads, 
            dropout=dropout,
            edge_dim=1,  # 边特征维度（距离信息）
            concat=True  # 拼接多头输出
        )
        
        # 第二层GAT：聚合2跳邻居信息并输出融合特征
        self.gat2 = GATv2Conv(
            hidden_dim * heads, 
            output_dim, 
            heads=1, 
            dropout=dropout,
            edge_dim=1,
            concat=False  # 不拼接，直接平均
        )
        
        # 跳跃连接
        self.skip_connection = nn.Linear(input_dim, output_dim)
        
        # 归一化
        self.layer_norm = nn.LayerNorm(output_dim)  # 确保所有无人机状态指标（位置/电量/速度）在相同尺度比较
        self.dropout = nn.Dropout(dropout)          # 模拟传感器随机故障，以p=dropout概率将某些神经元输出置为0
        
        print(f"DroneGNN初始化: {input_dim} -> {hidden_dim}*{heads} -> {output_dim}")

    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播：通过两层GATv2实现信息融合
        
        Args:
            x: 节点特征矩阵 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            e.g.
            edge_index = [
                [源节点1, 源节点2, ..., 源节点N],  # 第0行：边的起点
                [目标节点1, 目标节点2, ..., 目标节点N]  # 第1行：边的终点
            ]

            edge_attr: 边属性（距离信息） [num_edges, 1]
            e.g.
            # 边索引：3条边 (0→1, 1→2, 2→0)
            edge_index = torch.tensor([
                [0, 1, 2],  # 源节点
                [1, 2, 0]   # 目标节点
            ], dtype=torch.long)

            # 边属性：每条边的距离（单位：米）
            edge_attr = torch.tensor([
                [250.0],  # 边 0→1 的距离
                [300.0],  # 边 1→2 的距离
                [150.0]   # 边 2→0 的距离
            ])
            
        
        Returns:
            融合后的节点特征 [num_nodes, output_dim]
        """

        # 保存原始输入用于跳跃连接
        skip = self.skip_connection(x)
        
        # 第一层GAT
        x1 = self.gat1(x, edge_index, edge_attr)
        x1 = F.relu(x1)  # 使用ReLU激活函数，对第一层GATv2的输出进行非线性变换。如果没有激活函数，多层神经网络只是线性变换的堆叠（无论多少层，最终等效于一个线性变换）
        x1 = self.dropout(x1)
        
        # 第二层GAT
        x2 = self.gat2(x1, edge_index, edge_attr)
        
        # 跳跃连接 + 层归一化
        output = self.layer_norm(x2 + skip)
        
        return output  # output：[num_nodes, feature_dim]

    @staticmethod
    def build_adjacency_graph(observations: Dict, radius: float = 250.0, max_neighbors: int = 3) -> Tuple[Data, list]:
        """
        构建稀疏图 - 限制每个节点的最大邻居数
        """
        drone_ids = list(observations.keys())
        features = []
        positions = []
        
        # 提取特征
        for drone_id in drone_ids:
            drone = observations[drone_id]['self_state']
            velocity = drone.get('velocity', [0, 0, 0])
            if not hasattr(velocity, '__getitem__'):
                velocity = [0, 0, 0]
            
            feat = [
                drone['health'], drone['energy'], drone['threat_level'],
                drone['position'][0], drone['position'][1], drone['position'][2],
                velocity[0], velocity[1], velocity[2]
            ]
            features.append(feat)
            positions.append(drone['position'])
        
        # 构建受限邻接矩阵
        edge_index = []
        edge_distances = []
        
        for i in range(len(drone_ids)):
            # 计算到所有其他节点的距离
            distances = []
            for j in range(len(drone_ids)):
                if i != j:
                    pos_i = np.array(positions[i])
                    pos_j = np.array(positions[j])
                    dist = np.linalg.norm(pos_i - pos_j)
                    if dist < radius:
                        distances.append((j, dist))
            
            # 只保留最近的max_neighbors个邻居
            distances.sort(key=lambda x: x[1])
            neighbors = distances[:max_neighbors]
            
            for neighbor_idx, dist in neighbors:
                edge_index.append([i, neighbor_idx])
                normalized_dist = 1.0 - (dist / radius)  # 归一化
                edge_distances.append([normalized_dist])
        """
        e.g.
        edge_index = [
            [0, 1], [0, 2],  # drone0的边
            [1, 0], [1, 3],   # drone1的边
            [2, 0], [2, 3],   # drone2的边
            [3, 1], [3, 2]    # drone3的边
        ]
        edge_distances = [
            [0.4], [0.2],     # drone0的边属性
            [0.4], [0.4],     # drone1的边属性
            [0.2], [0.2],     # drone2的边属性
            [0.4], [0.4]      # drone3的边属性
        ]
        """
        
        # 转换为张量
        x = torch.tensor(features, dtype=torch.float)
        """
        e.g.
        features = [
            [0.1, 0.8],  # drone0: 速度=0.1, 电量=0.8
            [0.3, 0.6],  # drone1
            [0.5, 0.4],  # drone2
            [0.7, 0.2]   # drone3
        ]
        x = tensor([[0.1, 0.8],
                    [0.3, 0.6],
                    [0.5, 0.4],
                    [0.7, 0.2]])
        """

        if edge_index:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            """
            e.g.
            同上述例子：
            edge_index_tensor = tensor([[0, 0, 1, 1, 2, 2, 3, 3],
                                    [1, 2, 0, 3, 0, 3, 1, 2]])
            """
            edge_attr = torch.tensor(edge_distances, dtype=torch.float)
            """
            e.g.
            同上述例子：
            edge_attr = tensor([[0.4], [0.2], [0.4], [0.4], [0.2], [0.2], [0.4], [0.4]])
            """
        else:  # 如果没有边，创建空的张量
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr)
        
        print(f"构建稀疏图: 节点数={len(drone_ids)}, 边数={len(edge_index)}, 最大邻居数={max_neighbors}")
        
        return data, drone_ids

    @staticmethod  # 静态方法，不需要实例化
    def train_gnn_model(data: Data,
                       input_dim: int = 9, hidden_dim: int = 32,
                       output_dim: int = 16, epochs: int = 100,
                       pretrained_model: Optional['DroneGNN'] = None) -> 'DroneGNN':
        """
        训练GNN模型，使用多样性保持损失训练
        
        Args:
            data: 图数据
            labels: 策略标签
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
            epochs: 训练轮数
            pretrained_model: 可选的预训练模型，用于增量更新
        Returns:
            训练好的模型
        """
        # 如果已经存在预训练模型且结构匹配，则直接复用
        model = pretrained_model
        if model is not None:
            if getattr(model, 'input_dim', None) != input_dim or getattr(model, 'output_dim', None) != output_dim:
                # 特征维度已经改变，无法复用旧模型
                model = None
            else:
                print("复用已有的DroneGNN模型进行增量训练。")

        # 如果没有可复用模型，则重新初始化
        if model is None:
            model = DroneGNN(input_dim, hidden_dim, output_dim, heads=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
        
        print(f"开始训练 (多样性保持): {epochs}轮")
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()  # 清空模型参数的梯度缓存

            output = model(data.x, data.edge_index, data.edge_attr)
            """
            PyTorch的 nn.Module 类重写了 __call__ 方法，
            当调用 model(...) 时，实际上调用的是 model.__call__(...)
            __call__ 方法内部会调用 forward(...) 方法
            """
            
            # 计算损失
            loss = torch.tensor(0.0, device=data.x.device, requires_grad=True)
            
            # 1. 邻居一致性损失 (较小权重)：如果两个节点在图中是邻居，它们的输出特征应该尽量相似。
            if data.edge_index.shape[1] > 0:
                """
                data.edge_index.shape → [2, num_edges]（2：每条边包含两个节点索引）
                data.edge_index.shape[1] → num_edges
                """
                edge_src = data.edge_index[0]
                edge_dst = data.edge_index[1]
                neighbor_loss = F.mse_loss(output[edge_src], output[edge_dst])
                """
                output[edge_src] → 每条边的源节点的输出特征
                """
                loss = loss + 0.3 * neighbor_loss
            
            # 2. 多样性保持损失 (更大权重)
            pairwise_distances = []  # 存储欧几里得距离
            for i in range(output.shape[0]):
                """
                output.shape → [num_nodes, output_dim]
                """
                for j in range(i+1, output.shape[0]):
                    dist = torch.norm(output[i] - output[j])
                    pairwise_distances.append(dist)
            
            if pairwise_distances:
                diversity_loss = -torch.mean(torch.stack(pairwise_distances))
                loss = loss + 0.7 * diversity_loss
            
            # 3. 特征保持损失：保持节点之间的多样性结构。即：如果两个节点在输入特征空间中距离很远，模型输出也应该保持类似的距离
            input_diversity = []
            for i in range(data.x.shape[0]):
                """
                data.x.shape → [num_nodes, input_dim]
                """
                for j in range(i+1, data.x.shape[0]):
                    input_dist = torch.norm(data.x[i] - data.x[j])
                    input_diversity.append(input_dist)
            
            output_diversity = pairwise_distances  
            if input_diversity and output_diversity:
                diversity_preservation = F.mse_loss(
                    torch.stack(output_diversity), 
                    torch.stack(input_diversity)
                )
                loss = loss + 0.2 * diversity_preservation
            
            # 如果损失中没有可训练项，跳过反向传播
            if loss.grad_fn is None:
                continue

            loss.backward()   # 算出每个参数的“错”在哪儿、错了多少 -> 计算梯度，存储在.grad属性中
            # 梯度裁剪，限制梯度的最大范数，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # 根据“错”去调整参数
            
            if epoch % 25 == 0:
                print(f"  Epoch {epoch}: Loss={loss.item():.4f}")
        
        print("训练完成 (多样性保持)")
        return model
    
    @staticmethod
    def fuse_with_gnn(observations: Dict, strategy_predictor, radius: float = 250.0, pretrained_model=None) -> Tuple[Dict, 'DroneGNN']:
        """
        增强的信息融合 - 解决过度平滑
        """
        print("\n=== 增强信息融合 (防过度平滑) ===")
        
        # 构建稀疏图
        data, drone_ids = DroneGNN.build_adjacency_graph(observations, radius, max_neighbors=3)
        
        # 训练或微调模型
        model = DroneGNN.train_gnn_model(
            data,
            input_dim=data.x.shape[1],
            hidden_dim=32,
            output_dim=16,
            epochs=50,
            pretrained_model=pretrained_model
        )
        
        # 获取融合特征
        model.eval()     # 把模型切换到评估模式
        with torch.no_grad():
            fused_features = model(data.x, data.edge_index, data.edge_attr) # 前向传播，天然包含“从邻居节点聚合信息”功能
        
        # 验证多样性
        diversity_metrics = []  # 存储所有无人机（节点）之间输出特征的两两欧几里得距离
        for i in range(len(drone_ids)):
            for j in range(i+1, len(drone_ids)):
                dist = torch.norm(fused_features[i] - fused_features[j]).item()
                diversity_metrics.append(dist)
        
        avg_diversity = np.mean(diversity_metrics) if diversity_metrics else 0
        print(f"平均节点间距离: {avg_diversity:.4f}")
        
        # 创建增强观测数据
        enhanced_observations = {}
        for idx, drone_id in enumerate(drone_ids):
            enhanced_observations[drone_id] = {
                'self_state': observations[drone_id]['self_state'].copy(),
                'allies': observations[drone_id].get('allies', []).copy(),
                'enemies': observations[drone_id].get('enemies', []).copy(),
                'fused_feature': fused_features[idx].cpu().numpy(),
                'diversity_score': avg_diversity
            }
        
        return enhanced_observations, model
