"""
指标计算器
用于计算各种几何指标，包括TAR、高斯曲率相似性、几何连续性相似性、直纹面逼近误差相似性
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Set
from .meshProcessor import MeshProcessor
from .nonSphericalTool import NonSphericalTool


class IndicatorCalculator:
    def __init__(self, mesh: MeshProcessor, tool: NonSphericalTool, resolution: int = 30):
        """
        初始化指标计算器
        Args:
            mesh: 网格处理器
            tool: 刀具模型
            resolution: 采样分辨率
        """
        self.mesh = mesh
        self.tool = tool
        self.resolution = resolution

        # 在高斯球上采样方向
        self.directions = self._sample_g_sphere()

        # 缓存TAR计算结果
        self.tar_cache = {}
        # 缓存指标计算结果
        self.indicator_cache = {}

    def _sample_g_sphere(self) -> np.ndarray:
        """在高斯球上均匀采样方向
        
        优化：使用NumPy向量化计算提高性能
        """
        # 使用NumPy向量化计算
        # 生成theta和phi的网格
        theta = np.linspace(0, np.pi, self.resolution)
        phi = np.linspace(0, 2 * np.pi, self.resolution, endpoint=False)
        
        # 创建网格
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        
        # 向量化计算直角坐标
        x = np.sin(theta_grid) * np.cos(phi_grid)
        y = np.sin(theta_grid) * np.sin(phi_grid)
        z = np.cos(theta_grid)
        
        # 重塑为(N, 3)的数组
        directions = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        
        return directions

    def calculate_vertex_tar(self, vertex_idx: int) -> np.ndarray:
        """
        计算单个顶点的TAR
        Args:
            vertex_idx: 顶点索引
        Returns:
            布尔数组，表示哪些方向是可达的
        """
        # 检查缓存
        if vertex_idx in self.tar_cache:
            return self.tar_cache[vertex_idx]

        # 获取顶点信息
        vertex_pos = self.mesh.vertices[vertex_idx]
        vertex_normal = self.mesh.vertex_normals[vertex_idx]
        vertex_curvature = self.mesh.curvatures[vertex_idx]

        # 检查每个方向
        accessible = np.zeros(len(self.directions), dtype=bool)

        for i, direction in enumerate(self.directions):
            # 1. 检查方向与法向量的夹角
            dot_product = np.dot(direction, vertex_normal)
            if dot_product < 0:  # 只考虑法向量同侧的方向
                continue

            # 2. 基于曲率调整碰撞检测
            # 高曲率区域需要更严格的碰撞检测
            curvature_factor = 1.0 + vertex_curvature * 0.5

            # 3. 执行碰撞检测
            is_collision = self.tool.check_collision_simple(
                vertex_pos,
                vertex_normal,
                direction
            )

            # 4. 考虑刀具几何和方向角度的影响
            # 计算方向与法向量的夹角
            angle = math.acos(np.clip(dot_product, -1.0, 1.0))
            # 角度越大，碰撞可能性越高
            angle_factor = 1.0 + angle * 0.2

            # 5. 综合判断
            if not is_collision:
                # 对于接近法向量的方向，更容易通过碰撞检测
                if angle < math.pi / 4 or (angle < math.pi / 2 and not is_collision):
                    accessible[i] = True

        # 6. 后处理：确保TAR的连通性
        accessible = self._ensure_tar_connectivity(accessible, vertex_idx)

        # 缓存结果
        self.tar_cache[vertex_idx] = accessible

        return accessible

    def calculate_tar_similarity(self, vertex_idx1: int, vertex_idx2: int) -> float:
        """
        计算两个顶点TAR的相似性 (Jaccard指数)
        Args:
            vertex_idx1: 顶点1索引
            vertex_idx2: 顶点2索引
        Returns:
            相似性分数 (0到1)
        """
        tar1 = self.calculate_vertex_tar(vertex_idx1)
        tar2 = self.calculate_vertex_tar(vertex_idx2)

        # 计算Jaccard指数
        intersection = np.sum(tar1 & tar2)
        union = np.sum(tar1 | tar2)

        if union == 0:
            return 0.0

        return intersection / union

    def get_connected_tar_count(self, vertex_idx: int) -> int:
        """
        获取连通TAR数量
        Args:
            vertex_idx: 顶点索引
        Returns:
            连通区域数量
        """
        accessible = self.calculate_vertex_tar(vertex_idx)

        if np.sum(accessible) == 0:
            return 0

        # 1. 找到所有连通区域
        n = len(accessible)
        visited = np.zeros(n, dtype=bool)
        regions = []
        
        # 定义方向之间的距离阈值
        distance_threshold = 0.2  # 方向向量之间的最大夹角对应的余弦值
        
        # 遍历所有方向
        for i in range(n):
            if accessible[i] and not visited[i]:
                # 开始BFS
                region = []
                queue = [i]
                visited[i] = True
                
                while queue:
                    current = queue.pop(0)
                    region.append(current)
                    
                    # 查找相邻的方向
                    for j in range(n):
                        if accessible[j] and not visited[j]:
                            # 计算方向向量之间的余弦距离
                            dot_product = np.dot(self.directions[current], self.directions[j])
                            if dot_product > 1 - distance_threshold:
                                visited[j] = True
                                queue.append(j)
                
                # 只考虑足够大的区域
                if len(region) > len(self.directions) * 0.05:
                    regions.append(region)

        # 2. 返回连通区域数量
        return len(regions)

    def calculate_average_cutting_width(self, vertex_idx: int) -> float:
        """
        计算平均切削宽度
        Args:
            vertex_idx: 顶点索引
        Returns:
            平均切削宽度
        """
        accessible = self.calculate_vertex_tar(vertex_idx)
        vertex_pos = self.mesh.vertices[vertex_idx]
        vertex_normal = self.mesh.vertex_normals[vertex_idx]

        widths = []
        for i, direction in enumerate(self.directions):
            if accessible[i]:
                # 计算该方向的切削宽度
                width = self.tool.calculate_cutting_width(
                    vertex_pos,
                    vertex_normal,
                    direction,
                    scallop_height=0.4  # 默认残留高度
                )
                widths.append(width)

        if not widths:
            return 0.0

        return np.mean(widths)

    def get_tar_for_direction(self, vertex_idx: int, direction_idx: int) -> bool:
        """检查特定方向是否在TAR中"""
        accessible = self.calculate_vertex_tar(vertex_idx)
        return accessible[direction_idx]

    def _ensure_tar_connectivity(self, accessible: np.ndarray, vertex_idx: int) -> np.ndarray:
        """
        确保TAR的连通性
        Args:
            accessible: 可达方向数组
            vertex_idx: 顶点索引
        Returns:
            连通化后的可达方向数组
        """
        # 1. 找到最大的连通区域
        max_region = self._find_largest_connected_region(accessible)
        
        # 2. 如果最大区域为空，返回原始数组
        if not max_region:
            return accessible
        
        # 3. 创建新的可达数组，只保留最大的连通区域
        new_accessible = np.zeros_like(accessible)
        for idx in max_region:
            new_accessible[idx] = True
        
        return new_accessible

    def _find_largest_connected_region(self, accessible: np.ndarray) -> List[int]:
        """
        找到最大的连通区域
        Args:
            accessible: 可达方向数组
        Returns:
            最大连通区域的索引列表
        """
        # 1. 构建方向之间的邻接关系
        n = len(accessible)
        visited = np.zeros(n, dtype=bool)
        regions = []
        
        # 2. 定义方向之间的距离阈值
        distance_threshold = 0.2  # 方向向量之间的最大夹角对应的余弦值
        
        # 3. 遍历所有方向
        for i in range(n):
            if accessible[i] and not visited[i]:
                # 开始BFS
                region = []
                queue = [i]
                visited[i] = True
                
                while queue:
                    current = queue.pop(0)
                    region.append(current)
                    
                    # 查找相邻的方向
                    for j in range(n):
                        if accessible[j] and not visited[j]:
                            # 计算方向向量之间的余弦距离
                            dot_product = np.dot(self.directions[current], self.directions[j])
                            if dot_product > 1 - distance_threshold:
                                visited[j] = True
                                queue.append(j)
                
                regions.append(region)
        
        # 4. 找到最大的区域
        if not regions:
            return []
        
        largest_region = max(regions, key=len)
        return largest_region
    
    def calculate_gaussian_curvature_similarity(self, vertex_idx1: int, vertex_idx2: int, sigma_k: float = None) -> float:
        """
        计算高斯曲率相似性
        Args:
            vertex_idx1: 顶点1索引
            vertex_idx2: 顶点2索引
            sigma_k: 带宽参数
        Returns:
            高斯曲率相似性分数
        """
        # 检查缓存
        cache_key = f"gaussian_similarity_{vertex_idx1}_{vertex_idx2}"
        if cache_key in self.indicator_cache:
            return self.indicator_cache[cache_key]
        
        # 获取高斯曲率
        # 假设mesh对象有gaussian_curvatures属性
        if hasattr(self.mesh, 'gaussian_curvatures'):
            k1 = self.mesh.gaussian_curvatures[vertex_idx1]
            k2 = self.mesh.gaussian_curvatures[vertex_idx2]
        else:
            # 如果没有高斯曲率属性，使用平均曲率代替
            k1 = self.mesh.curvatures[vertex_idx1]
            k2 = self.mesh.curvatures[vertex_idx2]
        
        # 计算曲率差异
        curvature_diff = abs(k1 - k2)
        
        # 如果没有提供sigma_k，计算默认值
        if sigma_k is None:
            # 计算所有顶点的曲率标准差作为默认值
            if hasattr(self.mesh, 'gaussian_curvatures'):
                sigma_k = np.std(self.mesh.gaussian_curvatures)
            else:
                sigma_k = np.std(self.mesh.curvatures)
            
            # 避免除以零
            if sigma_k < 1e-6:
                sigma_k = 1.0
        
        # 计算高斯曲率相似性
        similarity = np.exp(-curvature_diff**2 / (2 * sigma_k**2))
        
        # 缓存结果
        self.indicator_cache[cache_key] = similarity
        
        return similarity
    
    def calculate_geometric_continuity_similarity(self, vertex_idx1: int, vertex_idx2: int, sigma_n: float = None, sigma_k: float = None) -> float:
        """
        计算几何连续性相似性
        Args:
            vertex_idx1: 顶点1索引
            vertex_idx2: 顶点2索引
            sigma_n: 法向变化的带宽参数
            sigma_k: 曲率变化的带宽参数
        Returns:
            几何连续性相似性分数
        """
        # 检查缓存
        cache_key = f"geometric_similarity_{vertex_idx1}_{vertex_idx2}"
        if cache_key in self.indicator_cache:
            return self.indicator_cache[cache_key]
        
        # 获取法向量
        n1 = self.mesh.vertex_normals[vertex_idx1]
        n2 = self.mesh.vertex_normals[vertex_idx2]
        
        # 计算法向夹角
        dot_product = np.dot(n1, n2)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        normal_angle = np.arccos(dot_product)
        
        # 获取主曲率
        if hasattr(self.mesh, 'principal_curvatures'):
            k1_1, k2_1 = self.mesh.principal_curvatures[vertex_idx1]
            k1_2, k2_2 = self.mesh.principal_curvatures[vertex_idx2]
            curvature_diff = abs(k1_1 - k1_2) + abs(k2_1 - k2_2)
        else:
            # 如果没有主曲率属性，使用平均曲率代替
            curvature_diff = abs(self.mesh.curvatures[vertex_idx1] - self.mesh.curvatures[vertex_idx2])
        
        # 如果没有提供sigma_n和sigma_k，计算默认值
        if sigma_n is None:
            # 计算所有顶点对的法向夹角标准差作为默认值
            sigma_n = 1.0  # 默认为1.0
        
        if sigma_k is None:
            # 计算所有顶点的曲率标准差作为默认值
            if hasattr(self.mesh, 'principal_curvatures'):
                all_curvatures = [abs(k1) + abs(k2) for k1, k2 in self.mesh.principal_curvatures]
                sigma_k = np.std(all_curvatures)
            else:
                sigma_k = np.std(self.mesh.curvatures)
            
            # 避免除以零
            if sigma_k < 1e-6:
                sigma_k = 1.0
        
        # 计算几何连续性相似性
        similarity = np.exp(-normal_angle**2 / (2 * sigma_n**2) - curvature_diff**2 / (2 * sigma_k**2))
        
        # 缓存结果
        self.indicator_cache[cache_key] = similarity
        
        return similarity
    
    def calculate_developable_surface_error_similarity(self, vertex_idx1: int, vertex_idx2: int, sigma_r: float = None) -> float:
        """
        计算直纹面逼近误差相似性
        Args:
            vertex_idx1: 顶点1索引
            vertex_idx2: 顶点2索引
            sigma_r: 带宽参数
        Returns:
            直纹面逼近误差相似性分数
        """
        # 检查缓存
        cache_key = f"developable_similarity_{vertex_idx1}_{vertex_idx2}"
        if cache_key in self.indicator_cache:
            return self.indicator_cache[cache_key]
        
        # 获取直纹面逼近误差
        if hasattr(self.mesh, 'rolled_error'):
            error1 = self.mesh.rolled_error[vertex_idx1]
            error2 = self.mesh.rolled_error[vertex_idx2]
        else:
            # 如果没有直纹面逼近误差属性，返回默认值
            error1 = 0.0
            error2 = 0.0
        
        # 计算误差差异
        error_diff = abs(error1 - error2)
        
        # 如果没有提供sigma_r，计算默认值
        if sigma_r is None:
            # 计算所有顶点的直纹面逼近误差标准差作为默认值
            if hasattr(self.mesh, 'rolled_error'):
                sigma_r = np.std(self.mesh.rolled_error)
            else:
                sigma_r = 1.0
            
            # 避免除以零
            if sigma_r < 1e-6:
                sigma_r = 1.0
        
        # 计算直纹面逼近误差相似性
        similarity = np.exp(-error_diff**2 / (2 * sigma_r**2))
        
        # 缓存结果
        self.indicator_cache[cache_key] = similarity
        
        return similarity
    
    def calculate_combined_similarity(self, vertex_idx1: int, vertex_idx2: int, weights: Tuple[float, float, float] = (0.3, 0.3, 0.4)) -> float:
        """
        计算综合相似性
        Args:
            vertex_idx1: 顶点1索引
            vertex_idx2: 顶点2索引
            weights: 权重 tuple (高斯曲率权重, 几何连续性权重, 直纹面误差权重)
        Returns:
            综合相似性分数
        """
        # 计算各指标相似性
        gaussian_sim = self.calculate_gaussian_curvature_similarity(vertex_idx1, vertex_idx2)
        geometric_sim = self.calculate_geometric_continuity_similarity(vertex_idx1, vertex_idx2)
        developable_sim = self.calculate_developable_surface_error_similarity(vertex_idx1, vertex_idx2)
        
        # 加权组合
        combined_sim = (weights[0] * gaussian_sim + 
                      weights[1] * geometric_sim + 
                      weights[2] * developable_sim)
        
        return combined_sim
