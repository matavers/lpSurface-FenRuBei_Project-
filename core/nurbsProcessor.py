"""
NURBS曲面处理器
处理NURBS曲面数据，计算几何属性，支持可视化
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import math
import open3d as o3d


class NURBSProcessor:
    def __init__(self, control_points: np.ndarray, knots_u: np.ndarray, knots_v: np.ndarray, degree_u: int, degree_v: int, weights: Optional[np.ndarray] = None):
        """
        初始化NURBS处理器
        Args:
            control_points: 控制点数组，形状为 (n_u+1, n_v+1, 3)
            knots_u: U方向的节点向量
            knots_v: V方向的节点向量
            degree_u: U方向的次数
            degree_v: V方向的次数
            weights: 权重数组，形状为 (n_u+1, n_v+1)
        """
        self.control_points = control_points
        self.knots_u = knots_u
        self.knots_v = knots_v
        self.degree_u = degree_u
        self.degree_v = degree_v
        
        # 如果没有提供权重，使用默认权重1
        if weights is None:
            self.weights = np.ones(control_points.shape[:2])
        else:
            self.weights = weights
        
        # 预计算基函数值
        self.precomputed_basis = {}
        
        print(f"NURBS曲面初始化完成: {control_points.shape[0]}x{control_points.shape[1]} 控制点, 次数: {degree_u}x{degree_v}")
    
    def save_nurbs_data(self, file_path: str):
        """
        保存NURBS参数到文件
        Args:
            file_path: 保存文件路径
        """
        data = {
            'control_points': self.control_points,
            'knots_u': self.knots_u,
            'knots_v': self.knots_v,
            'degree_u': self.degree_u,
            'degree_v': self.degree_v,
            'weights': self.weights
        }
        np.savez(file_path, **data)
        print(f"NURBS数据保存到: {file_path}")
    
    @classmethod
    def load_nurbs_data(cls, file_path: str):
        """
        从文件加载NURBS参数
        Args:
            file_path: 加载文件路径
        Returns:
            NURBSProcessor实例
        """
        data = np.load(file_path)
        return cls(
            data['control_points'],
            data['knots_u'],
            data['knots_v'],
            data['degree_u'],
            data['degree_v'],
            data['weights']
        )
    
    def basis_function(self, degree: int, knots: np.ndarray, i: int, u: float) -> float:
        """
        计算B样条基函数值
        Args:
            degree: 样条次数
            knots: 节点向量
            i: 基函数索引
            u: 参数值
        Returns:
            基函数值
        """
        # 检查是否已预计算
        key = (degree, tuple(knots), i, u)
        if key in self.precomputed_basis:
            return self.precomputed_basis[key]
        
        # 递归计算基函数
        if degree == 0:
            value = 1.0 if knots[i] <= u < knots[i+1] else 0.0
        else:
            denom1 = knots[i+degree] - knots[i]
            denom2 = knots[i+degree+1] - knots[i+1]
            
            term1 = 0.0
            if denom1 > 1e-8:
                term1 = (u - knots[i]) / denom1 * self.basis_function(degree-1, knots, i, u)
            
            term2 = 0.0
            if denom2 > 1e-8:
                term2 = (knots[i+degree+1] - u) / denom2 * self.basis_function(degree-1, knots, i+1, u)
            
            value = term1 + term2
        
        # 保存预计算结果
        self.precomputed_basis[key] = value
        return value
    
    def evaluate(self, u: float, v: float) -> np.ndarray:
        """
        计算NURBS曲面上指定参数点的值
        Args:
            u: U方向参数
            v: V方向参数
        Returns:
            点的坐标
        """
        # 计算U方向的基函数
        # 基函数数量 = 控制点数量 - 次数
        n_u = self.control_points.shape[0]
        basis_u = []
        for i in range(n_u - self.degree_u):
            basis_u.append(self.basis_function(self.degree_u, self.knots_u, i, u))
        
        # 计算V方向的基函数
        n_v = self.control_points.shape[1]
        basis_v = []
        for j in range(n_v - self.degree_v):
            basis_v.append(self.basis_function(self.degree_v, self.knots_v, j, v))
        
        # 计算加权和
        weighted_sum = np.zeros(3)
        weight_sum = 0.0
        
        for i in range(len(basis_u)):
            for j in range(len(basis_v)):
                weight = self.weights[i, j]
                basis = basis_u[i] * basis_v[j] * weight
                weighted_sum += self.control_points[i, j] * basis
                weight_sum += basis
        
        # 归一化
        if weight_sum > 1e-8:
            return weighted_sum / weight_sum
        else:
            return np.zeros(3)
    
    def evaluate_derivative(self, u: float, v: float, du: int = 1, dv: int = 1) -> np.ndarray:
        """
        计算NURBS曲面在指定点的导数
        Args:
            u: U方向参数
            v: V方向参数
            du: U方向导数阶数
            dv: V方向导数阶数
        Returns:
            导数向量
        """
        # 使用中心差分法计算导数
        h = 1e-6
        
        if du == 1 and dv == 0:
            return (self.evaluate(u+h, v) - self.evaluate(u-h, v)) / (2*h)
        elif du == 0 and dv == 1:
            return (self.evaluate(u, v+h) - self.evaluate(u, v-h)) / (2*h)
        elif du == 1 and dv == 1:
            return (self.evaluate(u+h, v+h) - self.evaluate(u+h, v-h) - self.evaluate(u-h, v+h) + self.evaluate(u-h, v-h)) / (4*h*h)
        elif du == 2 and dv == 0:
            return (self.evaluate(u+h, v) - 2*self.evaluate(u, v) + self.evaluate(u-h, v)) / (h*h)
        elif du == 0 and dv == 2:
            return (self.evaluate(u, v+h) - 2*self.evaluate(u, v) + self.evaluate(u, v-h)) / (h*h)
        else:
            return self.evaluate(u, v)
    
    def calculate_normal(self, u: float, v: float) -> np.ndarray:
        """
        计算NURBS曲面上指定点的法线
        Args:
            u: U方向参数
            v: V方向参数
        Returns:
            法线向量
        """
        # 计算一阶偏导数
        du = self.evaluate_derivative(u, v, du=1, dv=0)
        dv = self.evaluate_derivative(u, v, du=0, dv=1)
        
        # 计算叉积
        normal = np.cross(du, dv)
        
        # 归一化
        norm = np.linalg.norm(normal)
        if norm > 1e-8:
            return normal / norm
        else:
            return np.array([0, 0, 1])
    
    def calculate_plane_normal(self) -> np.ndarray:
        """
        计算平面的法向量（解析方法）
        Returns:
            平面法向量
        """
        # 从控制点计算平面法向量
        if self.control_points.shape[0] >= 3 and self.control_points.shape[1] >= 2:
            # 取三个不共线的控制点
            p0 = self.control_points[0, 0]
            p1 = self.control_points[1, 0]
            p2 = self.control_points[0, 1]
            
            # 计算两个向量
            v1 = p1 - p0
            v2 = p2 - p0
            
            # 计算法向量
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 1e-8:
                return normal / norm
        return np.array([0, 0, 1])
    
    def calculate_cylinder_normal(self, u: float, v: float) -> np.ndarray:
        """
        计算圆柱面的法向量（解析方法）
        Args:
            u: U方向参数
            v: V方向参数
        Returns:
            圆柱面法向量
        """
        # 圆柱面法向量指向径向外侧
        point = self.evaluate(u, v)
        # 假设圆柱面的中心轴为Z轴
        # 法向量为从Z轴指向该点的单位向量
        normal = np.array([point[0], point[1], 0])
        norm = np.linalg.norm(normal)
        if norm > 1e-8:
            return normal / norm
        return np.array([1, 0, 0])
    
    def calculate_sphere_normal(self, u: float, v: float) -> np.ndarray:
        """
        计算球面的法向量（解析方法）
        Args:
            u: U方向参数
            v: V方向参数
        Returns:
            球面法向量
        """
        # 球面法向量从球心指向该点
        point = self.evaluate(u, v)
        norm = np.linalg.norm(point)
        if norm > 1e-8:
            return point / norm
        return np.array([0, 0, 1])
    
    def calculate_cone_normal(self, u: float, v: float) -> np.ndarray:
        """
        计算圆锥面的法向量（解析方法）
        Args:
            u: U方向参数
            v: V方向参数
        Returns:
            圆锥面法向量
        """
        # 圆锥面法向量
        point = self.evaluate(u, v)
        # 假设圆锥顶点在(0, 0, height)，底面在z=0平面
        # 计算从顶点到该点的向量
        # 注意：这里需要根据实际圆锥参数调整
        # 简化实现：使用偏导数叉积
        return self.calculate_normal(u, v)
    
    def calculate_curvature(self, u: float, v: float) -> Tuple[float, float]:
        """
        计算NURBS曲面上指定点的主曲率
        Args:
            u: U方向参数
            v: V方向参数
        Returns:
            (k1, k2): 主曲率
        """
        # 计算一阶偏导数
        du = self.evaluate_derivative(u, v, du=1, dv=0)
        dv = self.evaluate_derivative(u, v, du=0, dv=1)
        
        # 计算二阶偏导数
        ddu = self.evaluate_derivative(u, v, du=2, dv=0)
        ddv = self.evaluate_derivative(u, v, du=0, dv=2)
        dudv = self.evaluate_derivative(u, v, du=1, dv=1)
        
        # 计算法线
        normal = self.calculate_normal(u, v)
        
        # 计算第一基本形式
        E = np.dot(du, du)
        F = np.dot(du, dv)
        G = np.dot(dv, dv)
        
        # 计算第二基本形式
        L = np.dot(ddu, normal)
        M = np.dot(dudv, normal)
        N = np.dot(ddv, normal)
        
        # 计算主曲率
        denominator = E*G - F*F
        if denominator < 1e-8:
            return 0.0, 0.0
        
        H = (E*N - 2*F*M + G*L) / (2*denominator)
        K = (L*N - M*M) / denominator
        
        sqrt_val = math.sqrt(max(0, H*H - K))
        k1 = H + sqrt_val
        k2 = H - sqrt_val
        
        return k1, k2
    
    def calculate_gaussian_curvature(self, u: float, v: float) -> float:
        """
        计算NURBS曲面上指定点的高斯曲率
        Args:
            u: U方向参数
            v: V方向参数
        Returns:
            高斯曲率
        """
        k1, k2 = self.calculate_curvature(u, v)
        return k1 * k2
    
    def calculate_mean_curvature(self, u: float, v: float) -> float:
        """
        计算NURBS曲面上指定点的平均曲率
        Args:
            u: U方向参数
            v: V方向参数
        Returns:
            平均曲率
        """
        k1, k2 = self.calculate_curvature(u, v)
        return (k1 + k2) / 2
    
    def generate_mesh(self, resolution_u: int = 50, resolution_v: int = 50) -> o3d.geometry.TriangleMesh:
        """
        生成NURBS曲面的网格表示
        Args:
            resolution_u: U方向分辨率
            resolution_v: V方向分辨率
        Returns:
            Open3D网格对象
        """
        vertices = []
        faces = []
        
        # 生成顶点
        for i in range(resolution_u + 1):
            u = i / resolution_u
            for j in range(resolution_v + 1):
                v = j / resolution_v
                vertices.append(self.evaluate(u, v))
        
        # 生成面
        for i in range(resolution_u):
            for j in range(resolution_v):
                idx = i * (resolution_v + 1) + j
                faces.append([idx, idx + 1, idx + resolution_v + 1])
                faces.append([idx + 1, idx + resolution_v + 2, idx + resolution_v + 1])
        
        # 创建网格
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(faces))
        mesh.compute_vertex_normals()
        
        return mesh
    
    def visualize(self, resolution_u: int = 50, resolution_v: int = 50):
        """
        可视化NURBS曲面
        Args:
            resolution_u: U方向分辨率
            resolution_v: V方向分辨率
        """
        mesh = self.generate_mesh(resolution_u, resolution_v)
        
        # 可视化控制点
        control_points = o3d.geometry.PointCloud()
        control_points.points = o3d.utility.Vector3dVector(self.control_points.reshape(-1, 3))
        control_points.paint_uniform_color([1, 0, 0])
        
        # 可视化控制网格
        control_lines = []
        n_u, n_v, _ = self.control_points.shape
        for i in range(n_u):
            for j in range(n_v - 1):
                control_lines.append([i * n_v + j, i * n_v + j + 1])
        for j in range(n_v):
            for i in range(n_u - 1):
                control_lines.append([i * n_v + j, (i + 1) * n_v + j])
        
        control_mesh = o3d.geometry.LineSet()
        control_mesh.points = control_points.points
        control_mesh.lines = o3d.utility.Vector2iVector(control_lines)
        control_mesh.paint_uniform_color([0, 1, 0])
        
        # 显示
        o3d.visualization.draw_geometries([mesh, control_points, control_mesh])
    
    @staticmethod
    def create_cylinder(radius: float = 1.0, height: float = 2.0) -> 'NURBSProcessor':
        """
        创建圆柱面
        Args:
            radius: 半径
            height: 高度
        Returns:
            NURBSProcessor实例
        """
        # 标准圆柱侧面的NURBS参数
        # 次数
        degree_u = 3  # U向次数（圆周方向）
        degree_v = 1  # V向次数（高度方向）
        
        # 控制点 - 9x2网格
        # U向：9个点定义一个完整的圆
        # V向：2个点定义高度的起点和终点
        control_points = np.zeros((9, 2, 3))
        
        # U向控制点（圆形截面）
        u_points = [
            [radius, 0, 0],       # P0
            [radius, radius, 0],   # P1
            [0, radius, 0],        # P2
            [-radius, radius, 0],  # P3
            [-radius, 0, 0],       # P4
            [-radius, -radius, 0], # P5
            [0, -radius, 0],       # P6
            [radius, -radius, 0],  # P7
            [radius, 0, 0]         # P8 (与P0重合，实现闭合)
        ]
        
        # 填充控制点网格
        for i in range(9):
            # 底部控制点 (z=0)
            control_points[i, 0] = u_points[i]
            # 顶部控制点 (z=height)
            control_points[i, 1] = [u_points[i][0], u_points[i][1], height]
        
        # 权重
        # U向权重序列: [1, √2/2, 1, √2/2, 1, √2/2, 1, √2/2, 1]
        k = math.cos(math.pi / 4)  # √2/2 ≈ 0.7071
        weights = np.zeros((9, 2))
        for i in range(9):
            weight = 1.0 if i % 2 == 0 else k
            weights[i, 0] = weight
            weights[i, 1] = weight
        
        # 节点向量
        # U向节点向量 (圆周方向)
        knots_u = np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1])
        # V向节点向量 (高度方向)
        knots_v = np.array([0, 0, 1, 1])
        
        return NURBSProcessor(control_points, knots_u, knots_v, degree_u, degree_v, weights)
    
    @staticmethod
    def create_sphere(radius: float = 1.0, resolution: int = 20) -> 'NURBSProcessor':
        """
        创建球面
        Args:
            radius: 半径
            resolution: 控制点分辨率
        Returns:
            NURBSProcessor实例
        """
        # 创建控制点
        control_points = np.zeros((resolution, resolution, 3))
        for i in range(resolution):
            theta = math.pi * i / (resolution - 1)
            for j in range(resolution):
                phi = 2 * math.pi * j / (resolution - 1)
                x = radius * math.sin(theta) * math.cos(phi)
                y = radius * math.sin(theta) * math.sin(phi)
                z = radius * math.cos(theta)
                control_points[i, j] = [x, y, z]
        
        # 创建节点向量
        knots_u = np.linspace(0, 1, resolution - 3)
        knots_v = np.linspace(0, 1, resolution - 3)
        
        # 扩展节点向量以满足次数要求
        knots_u = np.concatenate([[0, 0, 0], knots_u, [1, 1, 1]])
        knots_v = np.concatenate([[0, 0, 0], knots_v, [1, 1, 1]])
        
        return NURBSProcessor(control_points, knots_u, knots_v, 3, 3)
    
    @staticmethod
    def create_cone(radius: float = 1.0, height: float = 2.0, resolution_u: int = 20, resolution_v: int = 10) -> 'NURBSProcessor':
        """
        创建圆锥面
        Args:
            radius: 底面半径
            height: 高度
            resolution_u: U方向控制点数量
            resolution_v: V方向控制点数量
        Returns:
            NURBSProcessor实例
        """
        # 创建控制点
        control_points = np.zeros((resolution_u, resolution_v, 3))
        for i in range(resolution_u):
            angle = 2 * math.pi * i / (resolution_u - 1)
            for j in range(resolution_v):
                t = j / (resolution_v - 1)
                r = radius * (1 - t)
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                z = height * t
                control_points[i, j] = [x, y, z]
        
        # 创建节点向量
        knots_u = np.linspace(0, 1, resolution_u - 3)
        knots_v = np.linspace(0, 1, resolution_v - 3)
        
        # 扩展节点向量以满足次数要求
        knots_u = np.concatenate([[0, 0, 0], knots_u, [1, 1, 1]])
        knots_v = np.concatenate([[0, 0, 0], knots_v, [1, 1, 1]])
        
        return NURBSProcessor(control_points, knots_u, knots_v, 3, 3)