"""
数据生成工具

用于生成直纹面训练的合成数据，包括四边形和三角形分区。
"""

import numpy as np
import os
from typing import Tuple, List, Dict, Any


class DevelopableSurfaceDataGenerator:
    """
    直纹面数据生成器
    """
    
    def __init__(self, M: int = 16, num_samples: int = 1000):
        """
        初始化数据生成器
        Args:
            M: B样条曲线的控制点数量
            num_samples: 生成的样本数量
        """
        self.M = M
        self.num_samples = num_samples
        
    def generate_quadrilateral_sample(self) -> Dict[str, Any]:
        """
        生成四边形分区样本
        Returns:
            样本字典，包含内部点云、边缘点列、角点、分区类型
        """
        # 随机生成两条B样条曲线的控制点
        # 曲线A的控制点
        curve_A_control = np.random.randn(self.M, 3) * 0.3
        # 固定端点
        curve_A_control[0] = np.array([-1.0, 0.0, 0.0])
        curve_A_control[-1] = np.array([1.0, 0.0, 0.0])
        
        # 曲线B的控制点
        curve_B_control = np.random.randn(self.M, 3) * 0.3
        curve_B_control[0] = np.array([-1.0, 1.0, 0.0])
        curve_B_control[-1] = np.array([1.0, 1.0, 0.0])
        
        # 生成曲面上的点（直纹面）
        u_values = np.linspace(0, 1, 32)
        v_values = np.linspace(0, 1, 32)
        
        surface_points = []
        for u in u_values:
            for v in v_values:
                # 计算曲线A上的点
                point_A = self._evaluate_bspline(curve_A_control, u)
                # 计算曲线B上的点
                point_B = self._evaluate_bspline(curve_B_control, u)
                # 线性插值生成直纹面点
                point = (1 - v) * point_A + v * point_B
                surface_points.append(point)
        
        surface_points = np.array(surface_points)
        
        # 采样内部点云
        interior_indices = np.random.choice(len(surface_points), min(2000, len(surface_points)), replace=False)
        interior_points = surface_points[interior_indices]
        
        # 添加高斯噪声
        noise = np.random.randn(*interior_points.shape) * 0.01
        interior_points = interior_points + noise
        
        # 提取边缘点列
        edge_A = surface_points[::32]  # v=0的边
        edge_B = surface_points[31::32]  # v=1的边
        edge_C = surface_points[:32]  # u=0的边
        edge_D = surface_points[32*31:]  # u=1的边
        
        # 添加噪声到边缘点
        edge_A = edge_A + np.random.randn(*edge_A.shape) * 0.01
        edge_B = edge_B + np.random.randn(*edge_B.shape) * 0.01
        edge_C = edge_C + np.random.randn(*edge_C.shape) * 0.01
        edge_D = edge_D + np.random.randn(*edge_D.shape) * 0.01
        
        # 提取角点
        corners = [
            curve_A_control[0],  # v00
            curve_A_control[-1],  # v01
            curve_B_control[0],  # v10
            curve_B_control[-1]   # v11
        ]
        
        return {
            'interior_points': interior_points,
            'edge_points': [edge_A, edge_B, edge_C, edge_D],
            'corners': corners,
            'curve_A_control': curve_A_control,
            'curve_B_control': curve_B_control,
            'partition_type': 'quadrilateral'
        }
    
    def generate_triangle_sample(self) -> Dict[str, Any]:
        """
        生成三角形分区样本（锥面）
        Returns:
            样本字典，包含内部点云、边缘点列、角点、分区类型
        """
        # 随机生成顶点
        vertex = np.random.randn(3) * 0.3 + np.array([0.0, 0.5, 0.5])
        
        # 随机生成一条B样条曲线
        curve_control = np.random.randn(self.M, 3) * 0.3
        curve_control[0] = np.array([-1.0, -0.5, 0.0])
        curve_control[-1] = np.array([1.0, -0.5, 0.0])
        
        # 生成锥面上的点
        u_values = np.linspace(0, 1, 32)
        v_values = np.linspace(0, 1, 32)
        
        surface_points = []
        for u in u_values:
            for v in v_values:
                curve_point = self._evaluate_bspline(curve_control, u)
                point = vertex + v * (curve_point - vertex)
                surface_points.append(point)
        
        surface_points = np.array(surface_points)
        
        # 采样内部点云
        interior_indices = np.random.choice(len(surface_points), min(2000, len(surface_points)), replace=False)
        interior_points = surface_points[interior_indices]
        
        # 添加高斯噪声
        noise = np.random.randn(*interior_points.shape) * 0.01
        interior_points = interior_points + noise
        
        # 提取边缘点列
        edge_A = surface_points[::32]  # 曲线边
        edge_B = np.array([vertex + v * (curve_control[0] - vertex) for v in np.linspace(0, 1, 32)])  # 顶点到曲线起点
        edge_C = np.array([vertex + v * (curve_control[-1] - vertex) for v in np.linspace(0, 1, 32)])  # 顶点到曲线终点
        
        # 添加噪声到边缘点
        edge_A = edge_A + np.random.randn(*edge_A.shape) * 0.01
        edge_B = edge_B + np.random.randn(*edge_B.shape) * 0.01
        edge_C = edge_C + np.random.randn(*edge_C.shape) * 0.01
        
        # 提取角点
        corners = [
            vertex,
            curve_control[0],
            curve_control[-1]
        ]
        
        return {
            'interior_points': interior_points,
            'edge_points': [edge_A, edge_B, edge_C],
            'corners': corners,
            'vertex': vertex,
            'curve_control': curve_control,
            'partition_type': 'triangle'
        }
    
    def _evaluate_bspline(self, control_points: np.ndarray, t: float) -> np.ndarray:
        """
        评估B样条曲线上的点
        Args:
            control_points: 控制点数组
            t: 参数值 [0, 1]
        Returns:
            曲线上的点
        """
        n = len(control_points)
        t = t * (n - 1)
        k = int(np.floor(t))
        if k >= n - 1:
            k = n - 2
        t_local = t - k
        
        # 二次B样条基函数
        if k == 0:
            p0, p1, p2 = control_points[0], control_points[1], control_points[2]
        elif k == n - 2:
            p0, p1, p2 = control_points[-3], control_points[-2], control_points[-1]
        else:
            p0, p1, p2 = control_points[k], control_points[k+1], control_points[k+2]
        
        # 二次B样条公式
        result = (1 - t_local)**2 / 2 * p0 + \
                 (1 - 2*t_local + t_local**2) * p1 + \
                 t_local**2 / 2 * p2
        
        return result
    
    def generate_dataset(self, num_quad: int = None, num_tri: int = None) -> List[Dict[str, Any]]:
        """
        生成数据集
        Args:
            num_quad: 四边形样本数量
            num_tri: 三角形样本数量
        Returns:
            样本列表
        """
        if num_quad is None:
            num_quad = self.num_samples // 2
        if num_tri is None:
            num_tri = self.num_samples // 2
        
        dataset = []
        
        # 生成四边形样本
        for i in range(num_quad):
            sample = self.generate_quadrilateral_sample()
            # 数据增强
            sample = self._augment_sample(sample)
            dataset.append(sample)
        
        # 生成三角形样本
        for i in range(num_tri):
            sample = self.generate_triangle_sample()
            # 数据增强
            sample = self._augment_sample(sample)
            dataset.append(sample)
        
        return dataset
    
    def _augment_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        数据增强：随机旋转、平移、缩放
        """
        # 随机旋转
        angle = np.random.uniform(0, 2 * np.pi)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        rotation_matrix = self._rotation_matrix(axis, angle)
        
        # 应用旋转
        sample['interior_points'] = sample['interior_points'] @ rotation_matrix.T
        
        for i, edge in enumerate(sample['edge_points']):
            sample['edge_points'][i] = edge @ rotation_matrix.T
        
        for i, corner in enumerate(sample['corners']):
            sample['corners'][i] = corner @ rotation_matrix.T
        
        if 'vertex' in sample:
            sample['vertex'] = sample['vertex'] @ rotation_matrix.T
        
        if 'curve_A_control' in sample:
            sample['curve_A_control'] = sample['curve_A_control'] @ rotation_matrix.T
        
        if 'curve_B_control' in sample:
            sample['curve_B_control'] = sample['curve_B_control'] @ rotation_matrix.T
        
        if 'curve_control' in sample:
            sample['curve_control'] = sample['curve_control'] @ rotation_matrix.T
        
        # 随机平移
        translation = np.random.randn(3) * 0.2
        sample['interior_points'] = sample['interior_points'] + translation
        
        for i, edge in enumerate(sample['edge_points']):
            sample['edge_points'][i] = edge + translation
        
        for i, corner in enumerate(sample['corners']):
            sample['corners'][i] = corner + translation
        
        if 'vertex' in sample:
            sample['vertex'] = sample['vertex'] + translation
        
        if 'curve_A_control' in sample:
            sample['curve_A_control'] = sample['curve_A_control'] + translation
        
        if 'curve_B_control' in sample:
            sample['curve_B_control'] = sample['curve_B_control'] + translation
        
        if 'curve_control' in sample:
            sample['curve_control'] = sample['curve_control'] + translation
        
        # 随机缩放
        scale = np.random.uniform(0.8, 1.2)
        sample['interior_points'] = sample['interior_points'] * scale
        
        for i, edge in enumerate(sample['edge_points']):
            sample['edge_points'][i] = edge * scale
        
        for i, corner in enumerate(sample['corners']):
            sample['corners'][i] = corner * scale
        
        if 'vertex' in sample:
            sample['vertex'] = sample['vertex'] * scale
        
        if 'curve_A_control' in sample:
            sample['curve_A_control'] = sample['curve_A_control'] * scale
        
        if 'curve_B_control' in sample:
            sample['curve_B_control'] = sample['curve_B_control'] * scale
        
        if 'curve_control' in sample:
            sample['curve_control'] = sample['curve_control'] * scale
        
        return sample
    
    def _rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        生成绕任意轴旋转的旋转矩阵
        """
        axis = axis / np.linalg.norm(axis)
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        
        x, y, z = axis
        
        return np.array([
            [t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
            [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
            [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
        ])
    
    def normalize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        归一化样本：将点云平移至以角点包围盒中心，并缩放至单位立方体
        """
        # 计算所有点的包围盒
        all_points = [sample['interior_points']]
        all_points.extend(sample['edge_points'])
        
        min_coords = np.min(np.vstack([np.min(ep, axis=0) for ep in all_points]), axis=0)
        max_coords = np.max(np.vstack([np.max(ep, axis=0) for ep in all_points]), axis=0)
        
        center = (min_coords + max_coords) / 2
        scale = np.max(max_coords - min_coords) / 2
        
        # 归一化
        normalized = sample.copy()
        
        normalized['interior_points'] = (sample['interior_points'] - center) / scale
        
        normalized['edge_points'] = []
        for edge in sample['edge_points']:
            normalized['edge_points'].append((edge - center) / scale)
        
        normalized['corners'] = []
        for corner in sample['corners']:
            normalized['corners'].append((corner - center) / scale)
        
        if 'vertex' in sample:
            normalized['vertex'] = (sample['vertex'] - center) / scale
        
        if 'curve_A_control' in sample:
            normalized['curve_A_control'] = (sample['curve_A_control'] - center) / scale
        
        if 'curve_B_control' in sample:
            normalized['curve_B_control'] = (sample['curve_B_control'] - center) / scale
        
        if 'curve_control' in sample:
            normalized['curve_control'] = (sample['curve_control'] - center) / scale
        
        return normalized


def save_dataset(dataset: List[Dict[str, Any]], filepath: str):
    """
    保存数据集到文件
    """
    np.save(filepath, dataset)
    print(f"数据集已保存到 {filepath}")


def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """
    从文件加载数据集
    """
    dataset = np.load(filepath, allow_pickle=True)
    print(f"数据集已从 {filepath} 加载，共 {len(dataset)} 个样本")
    return dataset


if __name__ == "__main__":
    # 生成训练数据
    generator = DevelopableSurfaceDataGenerator(M=16, num_samples=1000)
    train_dataset = generator.generate_dataset(num_quad=500, num_tri=500)
    
    # 归一化
    train_dataset = [generator.normalize_sample(sample) for sample in train_dataset]
    
    # 保存
    save_dataset(train_dataset, "data/neural/train_dataset.npy")
    
    # 生成测试数据
    test_generator = DevelopableSurfaceDataGenerator(M=16, num_samples=200)
    test_dataset = test_generator.generate_dataset(num_quad=100, num_tri=100)
    
    # 归一化
    test_dataset = [test_generator.normalize_sample(sample) for sample in test_dataset]
    
    # 保存
    save_dataset(test_dataset, "data/neural/test_dataset.npy")
    
    print("数据生成完成！")
