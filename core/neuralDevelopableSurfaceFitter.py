"""
神经网络直纹面拟合模块

本模块实现基于PointNet++和编码器-解码器架构的直纹面拟合算法，用于五轴加工路径规划。
利用GPU加速提高拟合速度和精度，支持三角形和四边形分区的直纹面拟合。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
from .meshProcessor import MeshProcessor


class PointNetPPEncoder(nn.Module):
    """
    PointNet++编码器，用于处理内部点云
    """
    
    def __init__(self, in_channels=3, out_channels=512):
        super(PointNetPPEncoder, self).__init__()
        # Set Abstraction 层1
        self.sa1 = PointNetSetAbstraction(radius=0.2, k=16, in_channels=in_channels, mlp=[64, 64, 128])
        # Set Abstraction 层2
        self.sa2 = PointNetSetAbstraction(radius=0.4, k=32, in_channels=128 + 3, mlp=[128, 128, 256])
        # Set Abstraction 层3
        self.sa3 = PointNetSetAbstraction(radius=0.8, k=64, in_channels=256 + 3, mlp=[256, 256, 512])
        self.out_channels = out_channels
    
    def forward(self, xyz):
        """
        前向传播
        Args:
            xyz: 点云坐标，形状为 (B, N, 3)
        Returns:
            全局特征，形状为 (B, 512)
        """
        B, N, _ = xyz.shape
        # Set Abstraction 层1
        l1_xyz, l1_features = self.sa1(xyz, None)
        # Set Abstraction 层2
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        # Set Abstraction 层3
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        # 全局最大池化
        global_features = torch.max(l3_features, dim=1)[0]
        return global_features


class PointNetSetAbstraction(nn.Module):
    """
    PointNet Set Abstraction 模块
    """
    
    def __init__(self, radius, k, in_channels, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.radius = radius
        self.k = k
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channels
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz, features):
        """
        前向传播
        Args:
            xyz: 点云坐标，形状为 (B, N, 3)
            features: 点云特征，形状为 (B, N, C)
        Returns:
            采样后的点云坐标和特征
        """
        B, N, _ = xyz.shape
        device = xyz.device
        
        # 采样中心点（使用FPS）
        centroids = self.farthest_point_sample(xyz, N // 4)
        centroid_xyz = torch.gather(xyz, 1, centroids.unsqueeze(-1).expand(-1, -1, 3))
        
        # 查找每个中心点的邻域
        idx = self.ball_query(xyz, centroid_xyz, self.radius, self.k)
        grouped_xyz = self.grouping(xyz, idx)
        grouped_xyz -= centroid_xyz.unsqueeze(2)
        
        if features is not None:
            grouped_features = self.grouping(features, idx)
            grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1)
        else:
            grouped_features = grouped_xyz
        
        # 处理邻域特征
        grouped_features = grouped_features.permute(0, 3, 1, 2)
        for i, conv in enumerate(self.mlp_convs):
            grouped_features = F.relu(self.mlp_bns[i](conv(grouped_features)))
        
        # 最大池化
        new_features = torch.max(grouped_features, dim=3)[0]
        
        return centroid_xyz, new_features.permute(0, 2, 1)
    
    def farthest_point_sample(self, xyz, npoints):
        """
        Farthest Point Sampling
        """
        B, N, _ = xyz.shape
        device = xyz.device
        centroids = torch.zeros(B, npoints, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        for i in range(npoints):
            centroids[:, i] = farthest
            centroid = xyz[torch.arange(B), farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, dim=1)[1]
        return centroids
    
    def ball_query(self, xyz, centroids, radius, k):
        """
        Ball Query
        """
        B, N, _ = xyz.shape
        npoints = centroids.shape[1]
        device = xyz.device
        idx = torch.zeros(B, npoints, k, dtype=torch.long).to(device)
        
        for b in range(B):
            for i in range(npoints):
                centroid = centroids[b, i, :]
                dist = torch.sum((xyz[b, :, :] - centroid) ** 2, dim=-1)
                _, indices = torch.topk(-dist, k)
                idx[b, i, :] = indices
        
        return idx
    
    def grouping(self, points, idx):
        """
        Grouping points
        """
        B, N, C = points.shape
        npoints = idx.shape[1]
        k = idx.shape[2]
        device = points.device
        
        idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
        idx = idx + idx_base
        idx = idx.view(-1)
        
        grouped_points = points.view(B * N, -1)[idx, :]
        grouped_points = grouped_points.view(B, npoints, k, C)
        
        return grouped_points


class EdgeEncoder(nn.Module):
    """
    边缘点列编码器
    """
    
    def __init__(self, in_channels=3, out_channels=512):
        super(EdgeEncoder, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_channels)
        )
    
    def forward(self, edges):
        """
        前向传播
        Args:
            edges: 边缘点列，形状为 (B, K, L, 3)，其中K是边数，L是每条边的点数
        Returns:
            边缘全局特征，形状为 (B, 512)
        """
        B, K, L, _ = edges.shape
        edge_features = []
        
        for i in range(K):
            edge = edges[:, i, :, :].reshape(B * L, 3)
            edge_feat = self.edge_mlp(edge).reshape(B, L, 64).permute(0, 2, 1)
            edge_feat = self.conv1d_1(edge_feat)
            edge_feat = self.conv1d_2(edge_feat)
            edge_feat = torch.max(edge_feat, dim=2)[0]
            edge_features.append(edge_feat)
        
        # 补零到4条边
        while len(edge_features) < 4:
            edge_features.append(torch.zeros_like(edge_features[0]))
        
        edge_features = torch.cat(edge_features, dim=1)
        global_edge_features = self.fc(edge_features)
        
        return global_edge_features


class DevelopableSurfaceDecoder(nn.Module):
    """
    直纹面解码器
    """
    
    def __init__(self, in_channels=1024, M=16):
        super(DevelopableSurfaceDecoder, self).__init__()
        self.M = M
        self.fc1 = nn.Linear(in_channels, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 3 * (2 * M + 1))  # 2M个曲线控制点 + 1个顶点
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 融合特征，形状为 (B, 1024)
        Returns:
            控制点，形状为 (B, 2M+1, 3)
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))
        x = x.reshape(-1, 2 * self.M + 1, 3)
        return x


class NeuralDevelopableSurfaceFitter(nn.Module):
    """
    神经网络直纹面拟合器
    """
    
    def __init__(self, M=16):
        super(NeuralDevelopableSurfaceFitter, self).__init__()
        self.point_encoder = PointNetPPEncoder(out_channels=512)
        self.edge_encoder = EdgeEncoder(out_channels=512)
        self.feature_fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.decoder = DevelopableSurfaceDecoder(in_channels=256, M=M)
        self.M = M
    
    def forward(self, interior_points, edge_points, partition_type):
        """
        前向传播
        Args:
            interior_points: 内部点云，形状为 (B, N, 3)
            edge_points: 边缘点列，形状为 (B, K, L, 3)
            partition_type: 分区类型，形状为 (B, 2)，one-hot编码
        Returns:
            控制点，形状为 (B, 2M+1, 3)
        """
        # 提取内部点云特征
        point_features = self.point_encoder(interior_points)
        
        # 提取边缘点列特征
        edge_features = self.edge_encoder(edge_points)
        
        # 融合特征
        fused_features = torch.cat([point_features, edge_features], dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # 添加分区类型信息
        fused_features = torch.cat([fused_features, partition_type], dim=1)
        
        # 解码得到控制点
        control_points = self.decoder(fused_features)
        
        return control_points


class NeuralDevelopableSurfaceFitterWrapper:
    """
    神经网络直纹面拟合器包装器，提供与原有系统兼容的接口
    """
    
    def __init__(self, mesh: MeshProcessor, device=None):
        """
        初始化神经网络直纹面拟合器
        Args:
            mesh: 网格处理器
            device: 设备（CPU或GPU）
        """
        self.mesh = mesh
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.adjacency = mesh.adjacency
        
        # 初始化神经网络模型
        self.model = NeuralDevelopableSurfaceFitter()
        
        # 自动选择设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # 阈值参数
        self.epsilon = 0.001  # 顶点合并阈值
        self.T = 0.05  # 直纹面拟合误差阈值
    
    def fit_developable_surfaces(self, partition_labels: np.ndarray, edge_midpoints: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        拟合所有分区为直纹面
        Args:
            partition_labels: 分区标签数组
            edge_midpoints: 边缘中点数组
        Returns:
            直纹面字典，键为分区标签，值为直纹面参数
        """
        print("使用神经网络拟合直纹面...")
        
        # 1. 输入与预处理
        partitions = self._preprocess(partition_labels, edge_midpoints)
        
        # 2. 批量处理分区
        developable_surfaces = self._batch_process_partitions(partitions)
        
        return developable_surfaces
    
    def _preprocess(self, partition_labels: np.ndarray, edge_midpoints: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        预处理数据
        Args:
            partition_labels: 分区标签数组
            edge_midpoints: 边缘中点数组
        Returns:
            分区信息字典
        """
        partitions = {}
        unique_labels = np.unique(partition_labels)
        
        for label in unique_labels:
            # 获取该分区的所有顶点
            partition_vertices = np.where(partition_labels == label)[0]
            
            # 提取边界边
            boundary_edges = self._extract_boundary_edges(partition_vertices, partition_labels)
            
            # 提取内部点云
            interior_points = self._extract_interior_points(partition_vertices, boundary_edges)
            
            # 提取边缘点列
            edge_points = self._extract_edge_points(boundary_edges)
            
            # 确定分区类型
            partition_type = 'triangle' if len(boundary_edges) == 3 else 'quad'
            
            # 存储分区信息
            partitions[label] = {
                'vertices': partition_vertices,
                'boundary_edges': boundary_edges,
                'interior_points': interior_points,
                'edge_points': edge_points,
                'type': partition_type
            }
        
        return partitions
    
    def _extract_boundary_edges(self, partition_vertices: np.ndarray, partition_labels: np.ndarray) -> List[List[int]]:
        """
        提取分区的边界边
        Args:
            partition_vertices: 分区的顶点索引数组
            partition_labels: 分区标签数组
        Returns:
            边界边列表，每条边由顶点索引组成
        """
        boundary_edges = []
        vertex_set = set(partition_vertices)
        
        # 找出所有边界边（只属于一个分区的边）
        for v in partition_vertices:
            for neighbor in self.adjacency[v]:
                if partition_labels[neighbor] != partition_labels[v]:
                    # 按顺序存储边，确保一致性
                    edge = tuple(sorted([v, neighbor]))
                    if edge not in boundary_edges:
                        boundary_edges.append(list(edge))
        
        return boundary_edges
    
    def _extract_interior_points(self, partition_vertices: np.ndarray, boundary_edges: List[List[int]]) -> List[np.ndarray]:
        """
        提取分区的内部点云
        Args:
            partition_vertices: 分区的顶点索引数组
            boundary_edges: 边界边列表
        Returns:
            内部点云列表
        """
        # 收集边界顶点
        boundary_vertices = set()
        for edge in boundary_edges:
            boundary_vertices.update(edge)
        
        # 提取内部顶点
        interior_vertices = [v for v in partition_vertices if v not in boundary_vertices]
        
        # 转换为点云
        interior_points = [self.vertices[v] for v in interior_vertices]
        
        return interior_points
    
    def _extract_edge_points(self, boundary_edges: List[List[int]]) -> List[List[np.ndarray]]:
        """
        提取边缘点列
        Args:
            boundary_edges: 边界边列表
        Returns:
            边缘点列列表
        """
        edge_points = []
        for edge in boundary_edges:
            # 提取边上的点
            points = [self.vertices[v] for v in edge]
            edge_points.append(points)
        
        return edge_points
    
    def _batch_process_partitions(self, partitions: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        批量处理分区
        Args:
            partitions: 分区信息字典
        Returns:
            直纹面字典
        """
        developable_surfaces = {}
        
        # 准备批次数据
        batch_interior_points = []
        batch_edge_points = []
        batch_partition_type = []
        batch_labels = []
        
        for label, partition in partitions.items():
            # 处理内部点云
            interior_points = partition['interior_points']
            if len(interior_points) < 100:
                # 点云太少，使用传统方法
                continue
            
            # 归一化内部点云
            normalized_interior = self._normalize_points(interior_points)
            
            # 处理边缘点列
            edge_points = partition['edge_points']
            normalized_edges = []
            for edge in edge_points:
                normalized_edge = self._normalize_points(edge)
                normalized_edges.append(normalized_edge)
            
            # 填充边缘点列到固定长度
            while len(normalized_edges) < 4:
                normalized_edges.append([])
            
            # 分区类型编码
            if partition['type'] == 'triangle':
                partition_type = [1, 0]
            else:
                partition_type = [0, 1]
            
            batch_interior_points.append(normalized_interior)
            batch_edge_points.append(normalized_edges)
            batch_partition_type.append(partition_type)
            batch_labels.append(label)
        
        if batch_labels:
            # 批量推理
            with torch.no_grad():
                # 转换为张量
                interior_tensors = self._pad_and_stack(batch_interior_points)
                edge_tensors = self._pad_and_stack_edges(batch_edge_points)
                type_tensors = torch.tensor(batch_partition_type, dtype=torch.float32).to(self.device)
                
                # 推理
                control_points = self.model(interior_tensors, edge_tensors, type_tensors)
                
                # 处理结果
                for i, label in enumerate(batch_labels):
                    partition = partitions[label]
                    surface = self._process_control_points(control_points[i], partition)
                    if surface:
                        developable_surfaces[label] = surface
        
        # 处理剩余分区（使用传统方法）
        for label, partition in partitions.items():
            if label not in developable_surfaces:
                surface = self._fallback_to_traditional(partition)
                if surface:
                    developable_surfaces[label] = surface
        
        print(f"神经网络拟合完成，处理了 {len(developable_surfaces)} 个分区")
        return developable_surfaces
    
    def _normalize_points(self, points: List[np.ndarray]) -> np.ndarray:
        """
        归一化点云
        Args:
            points: 点云列表
        Returns:
            归一化后的点云
        """
        if not points:
            return np.zeros((100, 3))
        
        points = np.array(points)
        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        std[std == 0] = 1
        normalized = (points - mean) / std
        
        # 随机采样到固定数量
        if len(normalized) > 1000:
            indices = np.random.choice(len(normalized), 1000, replace=False)
            normalized = normalized[indices]
        elif len(normalized) < 100:
            # 填充到100个点
            padding = np.zeros((100 - len(normalized), 3))
            normalized = np.vstack([normalized, padding])
        
        return normalized
    
    def _pad_and_stack(self, points_list: List[np.ndarray]) -> torch.Tensor:
        """
        填充并堆叠点云
        Args:
            points_list: 点云列表
        Returns:
            堆叠后的张量
        """
        max_points = max(len(points) for points in points_list)
        batch_size = len(points_list)
        padded = np.zeros((batch_size, max_points, 3))
        
        for i, points in enumerate(points_list):
            padded[i, :len(points), :] = points
        
        return torch.tensor(padded, dtype=torch.float32).to(self.device)
    
    def _pad_and_stack_edges(self, edges_list: List[List[np.ndarray]]) -> torch.Tensor:
        """
        填充并堆叠边缘点列
        Args:
            edges_list: 边缘点列列表
        Returns:
            堆叠后的张量
        """
        batch_size = len(edges_list)
        max_edges = 4
        max_points = max(len(edge) for edges in edges_list for edge in edges)
        padded = np.zeros((batch_size, max_edges, max_points, 3))
        
        for i, edges in enumerate(edges_list):
            for j, edge in enumerate(edges):
                if j < max_edges:
                    padded[i, j, :len(edge), :] = edge
        
        return torch.tensor(padded, dtype=torch.float32).to(self.device)
    
    def _process_control_points(self, control_points: torch.Tensor, partition: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理控制点，生成直纹面
        Args:
            control_points: 控制点张量
            partition: 分区信息
        Returns:
            直纹面参数
        """
        control_points = control_points.cpu().numpy()
        
        if partition['type'] == 'triangle':
            # 三角形分区
            vertex = control_points[-1]
            curve_points = control_points[:self.M]
            
            surface = {
                'type': 'conical',
                'vertex': vertex.tolist(),
                'curve': {
                    'type': 'b-spline',
                    'control_points': curve_points.tolist()
                },
                'vertices': partition['vertices'],
                'label': partition.get('label', 0)
            }
        else:
            # 四边形分区
            curve0_points = control_points[:self.M]
            curve1_points = control_points[self.M:2*self.M]
            
            surface = {
                'type': 'developable',
                'curve0': {
                    'type': 'b-spline',
                    'control_points': curve0_points.tolist()
                },
                'curve1': {
                    'type': 'b-spline',
                    'control_points': curve1_points.tolist()
                },
                'vertices': partition['vertices'],
                'label': partition.get('label', 0)
            }
        
        return surface
    
    def _fallback_to_traditional(self, partition: Dict[str, Any]) -> Dict[str, Any]:
        """
        回退到传统方法
        Args:
            partition: 分区信息
        Returns:
            直纹面参数
        """
        # 简单实现：使用边界边作为曲线
        boundary_edges = partition['boundary_edges']
        
        if len(boundary_edges) == 3:
            # 三角形分区
            vertices = [self.vertices[v] for edge in boundary_edges for v in edge]
            vertices = list(set(map(tuple, vertices)))
            vertices = [list(v) for v in vertices]
            
            surface = {
                'type': 'conical',
                'vertex': vertices[0],
                'curve': {
                    'type': 'line',
                    'start_point': vertices[1],
                    'end_point': vertices[2]
                },
                'vertices': partition['vertices'],
                'label': partition.get('label', 0)
            }
        else:
            # 四边形分区
            edge1 = boundary_edges[0]
            edge2 = boundary_edges[1]
            
            surface = {
                'type': 'developable',
                'curve1': {
                    'type': 'line',
                    'start_point': self.vertices[edge1[0]].tolist(),
                    'end_point': self.vertices[edge1[1]].tolist()
                },
                'curve2': {
                    'type': 'line',
                    'start_point': self.vertices[edge2[0]].tolist(),
                    'end_point': self.vertices[edge2[1]].tolist()
                },
                'vertices': partition['vertices'],
                'label': partition.get('label', 0)
            }
        
        return surface
