"""
NURBS曲面处理器
用于处理NURBS曲面，计算几何属性，生成点云和网格
"""

import numpy as np
import open3d as o3d
from typing import List, Tuple, Dict, Any


class NURBSSurfaceProcessor:
    """
    NURBS曲面处理器类
    """
    
    def __init__(self, surface=None):
        """
        初始化NURBS曲面处理器
        Args:
            surface: NURBS曲面对象
        """
        self.surface = surface
        self.points = []
        self.normals = []
        self.curvatures = []
        self.principal_curvatures = []
    
    def create_test_surface(self):
        """
        创建测试用的NURBS曲面
        Returns:
            曲面参数
        """
        # 简单的双三次贝塞尔曲面
        self.surface = {
            'ctrlpts': np.array([
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
                [[0, 1, 1], [1, 1, 2], [2, 1, 2], [3, 1, 1]],
                [[0, 2, 2], [1, 2, 3], [2, 2, 3], [3, 2, 2]],
                [[0, 3, 0], [1, 3, 0], [2, 3, 0], [3, 3, 0]]
            ]),
            'degree_u': 3,
            'degree_v': 3
        }
        return self.surface
    
    def create_cylinder_surface(self, radius=1.0, height=2.0, resolution=16):
        """
        创建圆柱面NURBS曲面
        Args:
            radius: 圆柱半径
            height: 圆柱高度
            resolution: 圆周方向的分辨率
        Returns:
            曲面参数
        """
        # 创建圆柱面的控制点
        ctrlpts = []
        
        # 圆周方向
        for i in range(resolution + 1):
            theta = 2 * np.pi * i / resolution
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            # 高度方向两个点：底部和顶部
            ctrlpts.append([[x, y, 0], [x, y, height]])
        
        # 转换为正确的形状
        ctrlpts = np.array(ctrlpts)
        
        self.surface = {
            'ctrlpts': ctrlpts,
            'degree_u': 2,  # 圆周方向度数
            'degree_v': 1   # 高度方向度数
        }
        return self.surface
    
    def create_test_surface_from_pdf(self):
        """
        从测试面NURBS.pdf文件中的数据创建测试曲面
        使用PDF中提供的圆柱面数据
        Returns:
            曲面参数
        """
        # 使用PDF中推荐的圆柱面数据（有理形式）
        # 半径 R=1，高度 H=1
        R = 1.0
        H = 1.0
        
        # 控制点（U方向4个点，V方向2个点）
        # 权重：角点权重1，中间点权重√2/2≈0.7071
        ctrlpts = np.array([
            # V=0 层（底部）
            [[R, 0, 0], [R, 0, H]],      # (0,0) 和 (0,1)
            [[R, R, 0], [R, R, H]],      # (1,0) 和 (1,1)
            [[0, R, 0], [0, R, H]],      # (2,0) 和 (2,1)
            [[-R, R, 0], [-R, R, H]]     # (3,0) 和 (3,1)
        ])
        
        # 权重
        weights = np.array([
            [1.0, 1.0],      # (0,0) 和 (0,1)
            [np.sqrt(2)/2, np.sqrt(2)/2],  # (1,0) 和 (1,1)
            [1.0, 1.0],      # (2,0) 和 (2,1)
            [np.sqrt(2)/2, np.sqrt(2)/2]   # (3,0) 和 (3,1)
        ])
        
        self.surface = {
            'ctrlpts': ctrlpts,
            'weights': weights,
            'degree_u': 2,  # 二次
            'degree_v': 1   # 线性
        }
        return self.surface
    
    def _bernstein_poly(self, n, i, t):
        """
        计算伯恩斯坦多项式
        支持标量和数组输入
        """
        from scipy.special import comb
        # 确保i是数组
        i = np.asarray(i)
        # 计算组合数
        comb_vals = comb(n, i)
        # 计算多项式值
        return comb_vals * (t ** i) * ((1 - t) ** (n - i))
    
    def evaluate(self, u, v):
        """
        计算曲面上的点
        支持有理NURBS曲面（带权重）
        """
        ctrlpts = self.surface['ctrlpts']
        
        # 获取控制点的实际形状
        num_u, num_v, _ = ctrlpts.shape
        
        # 向量化计算伯恩斯坦多项式
        i_u = np.arange(num_u)
        i_v = np.arange(num_v)
        
        b_u = self._bernstein_poly(num_u - 1, i_u, u)
        b_v = self._bernstein_poly(num_v - 1, i_v, v)
        
        # 计算外积
        b_matrix = np.outer(b_u, b_v)
        
        # 检查是否有理NURBS（带权重）
        if 'weights' in self.surface:
            weights = self.surface['weights']
            # 计算加权和
            weighted_sum = np.sum(b_matrix[:, :, np.newaxis] * ctrlpts * weights[:, :, np.newaxis], axis=(0, 1))
            # 计算权重和
            weight_sum = np.sum(b_matrix * weights)
            # 有理除法
            if weight_sum > 1e-6:
                point = weighted_sum / weight_sum
            else:
                point = np.zeros(3)
        else:
            # 非有理NURBS
            point = np.sum(b_matrix[:, :, np.newaxis] * ctrlpts, axis=(0, 1))
        
        return point
    
    def compute_normal(self, u, v):
        """
        计算曲面在参数(u, v)处的法向量
        """
        # 数值微分计算一阶偏导数
        h = 1e-6
        du = (self.evaluate(u + h, v) - self.evaluate(u - h, v)) / (2 * h)
        dv = (self.evaluate(u, v + h) - self.evaluate(u, v - h)) / (2 * h)
        
        # 计算法向量
        normal = np.cross(du, dv)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
        else:
            normal = np.array([0, 0, 1])
        
        return normal
    
    def compute_curvatures(self, u, v):
        """
        计算曲面在参数(u, v)处的高斯曲率和主曲率
        """
        # 数值微分计算一阶和二阶偏导数
        h = 1e-6
        
        # 一阶偏导数
        du = (self.evaluate(u + h, v) - self.evaluate(u - h, v)) / (2 * h)
        dv = (self.evaluate(u, v + h) - self.evaluate(u, v - h)) / (2 * h)
        
        # 二阶偏导数
        duu = (self.evaluate(u + h, v) - 2 * self.evaluate(u, v) + self.evaluate(u - h, v)) / (h ** 2)
        duv = (self.evaluate(u + h, v + h) - self.evaluate(u + h, v - h) - self.evaluate(u - h, v + h) + self.evaluate(u - h, v - h)) / (4 * h ** 2)
        dvv = (self.evaluate(u, v + h) - 2 * self.evaluate(u, v) + self.evaluate(u, v - h)) / (h ** 2)
        
        # 计算第一基本形式系数
        E = np.dot(du, du)
        F = np.dot(du, dv)
        G = np.dot(dv, dv)
        
        # 计算法向量
        normal = np.cross(du, dv)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return 0.0, 0.0, 0.0
        
        # 计算第二基本形式系数
        L = np.dot(duu, normal) / norm
        M = np.dot(duv, normal) / norm
        N = np.dot(dvv, normal) / norm
        
        # 计算高斯曲率
        K = (L * N - M**2) / (E * G - F**2)
        
        # 计算平均曲率
        H = (E * N - 2 * F * M + G * L) / (2 * (E * G - F**2))
        
        # 计算主曲率
        discriminant = H**2 - K
        if discriminant < 0:
            k1 = H
            k2 = H
        else:
            sqrt_disc = np.sqrt(discriminant)
            k1 = H + sqrt_disc
            k2 = H - sqrt_disc
        
        return K, k1, k2
    
    def sample_points(self, resolution_u=50, resolution_v=50, adaptive=False, curvature_threshold=0.1):
        """
        从曲面采样点云
        """
        if not self.surface:
            raise ValueError("曲面未设置")
        
        points = []
        normals = []
        curvatures = []
        principal_curvatures = []
        
        if adaptive:
            # 自适应采样
            # 首先进行粗采样
            coarse_u = max(10, resolution_u // 5)
            coarse_v = max(10, resolution_v // 5)
            
            # 计算粗采样点的曲率
            coarse_points = []
            coarse_curvatures = []
            
            for i in range(coarse_u + 1):
                u = i / coarse_u
                for j in range(coarse_v + 1):
                    v = j / coarse_v
                    K, k1, k2 = self.compute_curvatures(u, v)
                    coarse_points.append((u, v))
                    coarse_curvatures.append(abs(K))
            
            # 基于曲率进行自适应采样
            for i in range(resolution_u + 1):
                u = i / resolution_u
                for j in range(resolution_v + 1):
                    v = j / resolution_v
                    
                    # 找到最近的粗采样点
                    min_dist = float('inf')
                    nearest_curvature = 0.0
                    
                    for (cu, cv), curv in zip(coarse_points, coarse_curvatures):
                        dist = (u - cu)**2 + (v - cv)**2
                        if dist < min_dist:
                            min_dist = dist
                            nearest_curvature = curv
                    
                    # 如果曲率大于阈值，增加采样密度
                    if nearest_curvature > curvature_threshold:
                        # 在周围增加采样点
                        sub_res = 2
                        for su in range(sub_res):
                            for sv in range(sub_res):
                                su_val = u + (su - 0.5) / (resolution_u * sub_res)
                                sv_val = v + (sv - 0.5) / (resolution_v * sub_res)
                                if 0 <= su_val <= 1 and 0 <= sv_val <= 1:
                                    point = self.evaluate(su_val, sv_val)
                                    normal = self.compute_normal(su_val, sv_val)
                                    K, k1, k2 = self.compute_curvatures(su_val, sv_val)
                                    
                                    points.append(point)
                                    normals.append(normal)
                                    curvatures.append(K)
                                    principal_curvatures.append([k1, k2])
                    else:
                        # 正常采样
                        point = self.evaluate(u, v)
                        normal = self.compute_normal(u, v)
                        K, k1, k2 = self.compute_curvatures(u, v)
                        
                        points.append(point)
                        normals.append(normal)
                        curvatures.append(K)
                        principal_curvatures.append([k1, k2])
        else:
            # 均匀采样
            for i in range(resolution_u + 1):
                u = i / resolution_u
                for j in range(resolution_v + 1):
                    v = j / resolution_v
                    
                    point = self.evaluate(u, v)
                    normal = self.compute_normal(u, v)
                    K, k1, k2 = self.compute_curvatures(u, v)
                    
                    points.append(point)
                    normals.append(normal)
                    curvatures.append(K)
                    principal_curvatures.append([k1, k2])
        
        self.points = np.array(points)
        self.normals = np.array(normals)
        self.curvatures = np.array(curvatures)
        self.principal_curvatures = np.array(principal_curvatures)
        
        return self.points, self.normals, self.curvatures, self.principal_curvatures
    
    def create_mesh(self, points=None):
        """
        将点云转换为三角网格
        """
        if points is None:
            points = self.points
        
        if len(points) == 0:
            raise ValueError("点云为空")
        
        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 估计法向量
        pcd.estimate_normals()
        
        # 使用泊松重建创建网格
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        
        # 裁剪网格
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
        
        return mesh
    
    def visualize_points(self):
        """
        可视化采样的点云
        """
        if len(self.points) == 0:
            raise ValueError("点云为空，请先采样")
        
        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        
        # 如果有法向量，设置法向量
        if len(self.normals) == len(self.points):
            pcd.normals = o3d.utility.Vector3dVector(self.normals)
        
        # 可视化
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    # 测试NURBS曲面处理器
    processor = NURBSSurfaceProcessor()
    
    # 创建测试曲面
    surface = processor.create_test_surface()
    print("创建测试曲面完成")
    
    # 采样点云
    print("开始采样点云...")
    points, normals, curvatures, principal_curvatures = processor.sample_points(
        resolution_u=30, 
        resolution_v=30, 
        adaptive=True, 
        curvature_threshold=0.1
    )
    print(f"采样完成，得到 {len(points)} 个点")
    
    # 创建网格
    print("开始创建网格...")
    mesh = processor.create_mesh()
    print("网格创建完成")
    
    # 可视化
    print("可视化点云...")
    processor.visualize_points()
    
    print("NURBS曲面处理器测试完成")
