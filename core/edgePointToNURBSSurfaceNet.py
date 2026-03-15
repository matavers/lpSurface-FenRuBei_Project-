"""
边缘点列和内部点云到NURBS直纹面参数的神经网络模块

本模块实现从分区边缘点列和内部点云直接预测NURBS直纹面参数的神经网络。
输入：
- 分区的边缘点列（有序点序列）
- 内部点云
输出：
- 两条NURBS准线的完整参数（控制点、权重、节点向量、次数）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any


class EdgeEncoder(nn.Module):
    """
    边缘点列编码器
    处理多条边缘点列，提取全局特征
    """
    
    def __init__(self, in_channels=3, out_channels=512, max_edges=4, max_points_per_edge=64):
        super(EdgeEncoder, self).__init__()
        self.max_edges = max_edges
        self.max_points_per_edge = max_points_per_edge
        
        # 每条边的点云编码
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # 1D 卷积处理有序点列
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        
        # 融合多条边的特征
        self.edge_fusion = nn.Sequential(
            nn.Linear(512 * max_edges, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_channels)
        )
    
    def forward(self, edges):
        """
        前向传播
        Args:
            edges: 边缘点列，形状为 (B, K, L, 3)
        Returns:
            边缘全局特征，形状为 (B, out_channels)
        """
        B, K, L, _ = edges.shape
        edge_features = []
        
        for i in range(K):
            edge = edges[:, i, :, :].reshape(B * L, 3)
            edge_feat = self.point_mlp(edge).reshape(B, L, 128).permute(0, 2, 1)
            edge_feat = self.conv1d_1(edge_feat)
            edge_feat = self.conv1d_2(edge_feat)
            edge_feat = torch.max(edge_feat, dim=2)[0]
            edge_features.append(edge_feat)
        
        while len(edge_features) < self.max_edges:
            edge_features.append(torch.zeros(B, 512, device=edges.device))
        
        edge_features = torch.cat(edge_features, dim=1)
        global_edge_features = self.edge_fusion(edge_features)
        
        return global_edge_features


class PointCloudEncoder(nn.Module):
    """
    内部点云编码器
    处理无序点云，提取全局特征
    """
    
    def __init__(self, in_channels=3, out_channels=512, max_points=1000):
        super(PointCloudEncoder, self).__init__()
        self.max_points = max_points
        
        # 点云特征提取
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        # 全局特征融合
        self.global_fusion = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels)
        )
    
    def forward(self, point_cloud):
        """
        前向传播
        Args:
            point_cloud: 内部点云，形状为 (B, N, 3)
        Returns:
            点云全局特征，形状为 (B, out_channels)
        """
        B, N, _ = point_cloud.shape
        
        # 处理点云
        point_feat = self.point_mlp(point_cloud.reshape(B * N, 3))
        point_feat = point_feat.reshape(B, N, 256)
        
        # 全局最大池化
        global_point_features = torch.max(point_feat, dim=1)[0]
        global_point_features = self.global_fusion(global_point_features)
        
        return global_point_features


class NURBSSurfaceDecoder(nn.Module):
    """
    NURBS直纹面解码器
    从融合特征生成两条NURBS准线的参数
    """
    
    def __init__(self, in_channels=1024, M=16, degree=3):
        super(NURBSSurfaceDecoder, self).__init__()
        self.M = M  # 每条曲线的控制点数量
        self.degree = degree  # NURBS曲线次数
        
        # 计算节点向量长度
        self.knot_vector_length = degree + M + 1
        
        # 输出参数数量
        # 每条曲线：M个控制点(3坐标) + M个权重 + 节点向量 + 次数(固定)
        output_size = 2 * (M * 3 + M + self.knot_vector_length + 1)
        
        self.fc1 = nn.Linear(in_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, output_size)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(4096)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 融合特征，形状为 (B, in_channels)
        Returns:
            NURBS参数，形状为 (B, 输出参数数量)
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class EdgePointToNURBSSurfaceNet(nn.Module):
    """
    边缘点列和内部点云到NURBS直纹面的完整网络
    """
    
    def __init__(self, M=16, degree=3):
        super(EdgePointToNURBSSurfaceNet, self).__init__()
        self.M = M
        self.degree = degree
        
        # 编码器
        self.edge_encoder = EdgeEncoder(out_channels=512)
        self.point_encoder = PointCloudEncoder(out_channels=512)
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(512 + 512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        
        # 解码器
        self.surface_decoder = NURBSSurfaceDecoder(in_channels=1024, M=M, degree=degree)
    
    def forward(self, edges, point_cloud):
        """
        前向传播
        Args:
            edges: 边缘点列，形状为 (B, K, L, 3)
            point_cloud: 内部点云，形状为 (B, N, 3)
        Returns:
            NURBS参数，形状为 (B, 输出参数数量)
        """
        # 编码边缘和点云
        edge_features = self.edge_encoder(edges)
        point_features = self.point_encoder(point_cloud)
        
        # 融合特征
        fused_features = torch.cat([edge_features, point_features], dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # 解码得到NURBS参数
        nurbs_params = self.surface_decoder(fused_features)
        
        return nurbs_params


class EdgePointToNURBSSurfaceWrapper:
    """
    边缘点列和内部点云到NURBS直纹面包装器，提供与原有系统兼容的接口
    """
    
    def __init__(self, device=None, model_path=None, M=16, degree=3):
        """
        初始化
        Args:
            device: 设备（CPU 或 GPU）
            model_path: 模型权重文件路径
            M: 每条曲线的控制点数量
            degree: NURBS曲线次数
        """
        self.M = M
        self.degree = degree
        self.knot_vector_length = degree + M + 1
        
        # 自动选择设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 创建模型
        self.model = EdgePointToNURBSSurfaceNet(M=M, degree=degree)
        self.model.to(self.device)
        
        # 加载模型权重（如果提供）
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """加载模型权重"""
        import os
        try:
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"EdgePointToNURBSSurfaceNet 模型已从 {model_path} 加载")
            else:
                print(f"警告：模型文件 {model_path} 不存在")
        except Exception as e:
            print(f"加载模型时出错：{e}")
    
    def save_model(self, model_path: str):
        """保存模型权重"""
        try:
            torch.save(self.model.state_dict(), model_path)
            print(f"模型已保存到 {model_path}")
        except Exception as e:
            print(f"保存模型时出错：{e}")
    
    def preprocess_edges(self, edge_points: List[List[np.ndarray]], max_points_per_edge=64) -> torch.Tensor:
        """
        预处理边缘点列
        Args:
            edge_points: 边缘点列列表，每条边是一个点数组
            max_points_per_edge: 每条边的最大点数
        Returns:
            预处理后的张量，形状为 (1, K, L, 3)
        """
        padded_edges = []
        # 只取前4条边，确保不超过4条
        for edge in edge_points[:4]:
            edge = np.array(edge)
            if len(edge) > max_points_per_edge:
                indices = np.linspace(0, len(edge)-1, max_points_per_edge, dtype=int)
                edge = edge[indices]
            elif len(edge) < max_points_per_edge:
                padding = np.zeros((max_points_per_edge - len(edge), 3))
                edge = np.vstack([edge, padding])
            padded_edges.append(edge)
        
        while len(padded_edges) < 4:
            padded_edges.append(np.zeros((max_points_per_edge, 3)))
        
        edge_array = np.array(padded_edges)
        edge_tensor = torch.tensor(edge_array, dtype=torch.float32).unsqueeze(0)
        
        return edge_tensor.to(self.device)
    
    def preprocess_point_cloud(self, point_cloud: List[np.ndarray], max_points=1000) -> torch.Tensor:
        """
        预处理内部点云
        Args:
            point_cloud: 内部点云列表
            max_points: 最大点数
        Returns:
            预处理后的张量，形状为 (1, N, 3)
        """
        point_cloud = np.array(point_cloud)
        
        if len(point_cloud) > max_points:
            indices = np.random.choice(len(point_cloud), max_points, replace=False)
            point_cloud = point_cloud[indices]
        elif len(point_cloud) < max_points:
            padding = np.zeros((max_points - len(point_cloud), 3))
            point_cloud = np.vstack([point_cloud, padding])
        
        point_tensor = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0)
        return point_tensor.to(self.device)
    
    def predict(self, edge_points: List[List[np.ndarray]], point_cloud: List[np.ndarray]) -> Dict[str, Any]:
        """
        预测NURBS直纹面参数
        Args:
            edge_points: 边缘点列列表
            point_cloud: 内部点云列表
        Returns:
            直纹面参数字典
        """
        self.model.eval()
        
        # 预处理
        edge_tensor = self.preprocess_edges(edge_points)
        point_tensor = self.preprocess_point_cloud(point_cloud)
        
        # 推理
        with torch.no_grad():
            nurbs_params = self.model(edge_tensor, point_tensor)
        
        nurbs_params = nurbs_params.squeeze(0).cpu().numpy()
        
        # 解析参数
        # 每条曲线的参数长度
        curve_param_length = self.M * 3 + self.M + self.knot_vector_length + 1
        
        # 分离两条曲线的参数
        curve0_params = nurbs_params[:curve_param_length]
        curve1_params = nurbs_params[curve_param_length:]
        
        # 解析第一条曲线
        curve0_control = curve0_params[:self.M*3].reshape(self.M, 3)
        curve0_weights = curve0_params[self.M*3:self.M*4]
        # 确保权重为正
        curve0_weights = np.exp(curve0_weights)  # 使用指数确保权重为正
        curve0_knots = curve0_params[self.M*4:self.M*4+self.knot_vector_length]
        # 确保节点向量非递减
        curve0_knots = np.cumsum(np.exp(curve0_knots[:-1]))  # 累积和确保非递减
        curve0_knots = np.insert(curve0_knots, 0, 0.0)  # 添加起始节点
        curve0_knots = curve0_knots / curve0_knots[-1]  # 归一化到[0,1]
        curve0_degree = int(round(curve0_params[-1]))
        curve0_degree = max(1, min(self.degree, curve0_degree))  # 限制在合理范围内
        
        # 解析第二条曲线
        curve1_control = curve1_params[:self.M*3].reshape(self.M, 3)
        curve1_weights = curve1_params[self.M*3:self.M*4]
        curve1_weights = np.exp(curve1_weights)  # 使用指数确保权重为正
        curve1_knots = curve1_params[self.M*4:self.M*4+self.knot_vector_length]
        curve1_knots = np.cumsum(np.exp(curve1_knots[:-1]))  # 累积和确保非递减
        curve1_knots = np.insert(curve1_knots, 0, 0.0)  # 添加起始节点
        curve1_knots = curve1_knots / curve1_knots[-1]  # 归一化到[0,1]
        curve1_degree = int(round(curve1_params[-1]))
        curve1_degree = max(1, min(self.degree, curve1_degree))  # 限制在合理范围内
        
        # 构建直纹面参数
        surface = {
            'type': 'developable',
            'curve0': {
                'type': 'nurbs',
                'control_points': curve0_control.tolist(),
                'weights': curve0_weights.tolist(),
                'knot_vector': curve0_knots.tolist(),
                'degree': curve0_degree
            },
            'curve1': {
                'type': 'nurbs',
                'control_points': curve1_control.tolist(),
                'weights': curve1_weights.tolist(),
                'knot_vector': curve1_knots.tolist(),
                'degree': curve1_degree
            }
        }
        
        return surface
    
    def _compute_basis_function(self, u, knot_vector, degree, i):
        """
        计算B样条基函数（Cox-de Boor递推公式）
        """
        if degree == 0:
            return 1.0 if knot_vector[i] <= u < knot_vector[i+1] else 0.0
        else:
            denominator1 = knot_vector[i+degree] - knot_vector[i]
            denominator2 = knot_vector[i+degree+1] - knot_vector[i+1]
            
            term1 = 0.0
            if denominator1 > 1e-8:
                term1 = (u - knot_vector[i]) / denominator1 * self._compute_basis_function(u, knot_vector, degree-1, i)
            
            term2 = 0.0
            if denominator2 > 1e-8:
                term2 = (knot_vector[i+degree+1] - u) / denominator2 * self._compute_basis_function(u, knot_vector, degree-1, i+1)
            
            return term1 + term2
    
    def _evaluate_nurbs_curve(self, curve, u):
        """
        评估NURBS曲线上的点
        """
        control_points = np.array(curve['control_points'])
        weights = np.array(curve['weights'])
        knot_vector = np.array(curve['knot_vector'])
        degree = curve['degree']
        
        n = len(control_points) - 1
        numerator = np.zeros(3)
        denominator = 0.0
        
        for i in range(n+1):
            basis = self._compute_basis_function(u, knot_vector, degree, i)
            weighted_basis = basis * weights[i]
            numerator += weighted_basis * control_points[i]
            denominator += weighted_basis
        
        if denominator > 1e-8:
            return numerator / denominator
        else:
            return np.zeros(3)
    
    def generate_surface_points(self, surface: Dict[str, Any], num_u: int = 32, num_v: int = 32) -> np.ndarray:
        """
        根据直纹面参数生成点云
        Args:
            surface: 直纹面参数
            num_u: u 方向采样点数
            num_v: v 方向采样点数
        Returns:
            直纹面点云 (num_u*num_v, 3)
        """
        surface_points = []
        
        for u in np.linspace(0, 1, num_u):
            # 评估两条NURBS曲线上的点
            curve0_point = self._evaluate_nurbs_curve(surface['curve0'], u)
            curve1_point = self._evaluate_nurbs_curve(surface['curve1'], u)
            
            # 沿 v 方向插值生成直纹面
            for v in np.linspace(0, 1, num_v):
                point = (1 - v) * curve0_point + v * curve1_point
                surface_points.append(point)
        
        return np.array(surface_points)


if __name__ == "__main__":
    # 测试网络
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    # 创建模型
    model = EdgePointToNURBSSurfaceNet(M=16, degree=3).to(device)
    print(f"模型参数数量：{sum(p.numel() for p in model.parameters())}")
    
    # 测试输入
    batch_size = 2
    num_edges = 4
    points_per_edge = 64
    points_per_cloud = 1000
    
    test_edges = torch.randn(batch_size, num_edges, points_per_edge, 3).to(device)
    test_point_cloud = torch.randn(batch_size, points_per_cloud, 3).to(device)
    
    # 前向传播
    output = model(test_edges, test_point_cloud)
    print(f"输入边缘形状：{test_edges.shape}")
    print(f"输入点云形状：{test_point_cloud.shape}")
    print(f"输出形状：{output.shape}")
    print(f"输出参数数量：{output.shape[1]}")
