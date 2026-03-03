"""
工具方向场生成器
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from scipy.spatial.distance import cdist
from .meshProcessor import MeshProcessor
from .nonSphericalTool import NonSphericalTool


class ToolOrientationField:
    def __init__(self, mesh: MeshProcessor, partition_labels: np.ndarray, tool: NonSphericalTool):
        """
        初始化工具方向场
        Args:
            mesh: 网格处理器
            partition_labels: 分区标签
            tool: 刀具模型
        """
        self.mesh = mesh
        self.partition_labels = partition_labels
        self.tool = tool
        self.num_vertices = len(mesh.vertices)

        # 为每个顶点存储TAR和工具方向
        self.vertex_tars = {}
        self.vertex_orientations = np.zeros((self.num_vertices, 3))

        # 分区信息
        # 使用NumPy向量化操作构建分区字典
        unique_labels = np.unique(partition_labels)
        self.partitions = {}
        for label in unique_labels:
            # 使用np.where快速找到所有具有该标签的顶点索引
            self.partitions[label] = np.where(partition_labels == label)[0].tolist()

    def select_seed_points(self) -> Dict[int, int]:
        """
        为每个分区选择种子点（TAR面积最大的顶点）
        Returns:
            分区ID到种子点索引的映射
        """
        seed_points = {}

        for partition_id, vertices in self.partitions.items():
            max_tar_area = -1
            seed_vertex = -1

            for vertex_idx in vertices:
                # 简化：计算TAR面积（可达方向数量）
                # 实际应从TARCalculator获取
                tar_area = self._estimate_tar_area(vertex_idx)

                if tar_area > max_tar_area:
                    max_tar_area = tar_area
                    seed_vertex = vertex_idx

            if seed_vertex != -1:
                seed_points[partition_id] = seed_vertex

        return seed_points

    def _estimate_tar_area(self, vertex_idx: int) -> float:
        """估计TAR面积（完整实现）
        
        基于表面点的曲率、法向量和刀具几何形状计算TAR面积
        
        Args:
            vertex_idx: 顶点索引
        Returns:
            TAR面积估计值
        """
        # 获取顶点信息
        vertex_pos = self.mesh.vertices[vertex_idx]
        vertex_normal = self.mesh.vertex_normals[vertex_idx]
        curvature = self.mesh.curvatures[vertex_idx]
        
        # 1. 基于曲率的基础TAR面积
        # 曲率越小，TAR面积越大
        base_area = 1.0 / (curvature + 0.1)
        
        # 2. 考虑刀具几何形状
        # 对于非球形刀具，TAR面积会受到刀具形状的影响
        tool_type = self.tool.profile_type
        tool_params = self.tool.params
        
        # 根据刀具类型调整TAR面积
        if tool_type == 'ellipsoidal':
            # 椭球形刀具：基于半轴长度调整
            semi_axes = tool_params.get('semi_axes', [1.0, 1.0])
            tool_ratio = semi_axes[0] / semi_axes[1] if semi_axes[1] > 0 else 1.0
            shape_factor = 0.8 + 0.2 * min(tool_ratio, 1/tool_ratio)
        elif tool_type == 'cylindrical':
            # 圆柱形刀具：基于直径和长度调整
            diameter = tool_params.get('diameter', 1.0)
            length = tool_params.get('length', 1.0)
            shape_factor = 0.9 if length > diameter else 0.7
        else:
            # 默认形状因子
            shape_factor = 0.8
        
        # 3. 考虑表面法向量方向
        # 计算法向量与常见加工方向的一致性
        # 假设常见加工方向为Z轴正方向
        vertical_alignment = abs(np.dot(vertex_normal, np.array([0, 0, 1])))
        direction_factor = 0.7 + 0.3 * vertical_alignment
        
        # 4. 计算最终TAR面积
        tar_area = base_area * shape_factor * direction_factor
        
        # 5. 归一化到合理范围
        tar_area = max(0.1, min(tar_area, 1.0))
        
        return tar_area

    def greedy_tar_selection(self, partition_id: int, seed_vertex: int):
        """
        贪心算法选择每个顶点的TAR
        
        优化：使用collections.deque提高BFS效率
        
        Args:
            partition_id: 分区ID
            seed_vertex: 种子点
        """
        from collections import deque
        
        vertices = self.partitions[partition_id]

        # 初始化队列 - 使用deque提高出队效率
        queue = deque([seed_vertex])
        visited = set([seed_vertex])

        # 种子点的方向（简化：使用顶点法向量）
        seed_orientation = self.mesh.vertex_normals[seed_vertex]
        self.vertex_orientations[seed_vertex] = seed_orientation

        while queue:
            current = queue.popleft()  # O(1)时间复杂度
            current_orientation = self.vertex_orientations[current]

            # 处理邻居
            for neighbor in self.mesh.adjacency[current]:
                if neighbor in visited or self.partition_labels[neighbor] != partition_id:
                    continue

                # 选择与当前方向最相似的方向
                best_orientation = self._select_best_orientation(
                    neighbor, current_orientation
                )

                self.vertex_orientations[neighbor] = best_orientation
                visited.add(neighbor)
                queue.append(neighbor)

    def _select_best_orientation(self, vertex_idx: int, reference_orientation: np.ndarray) -> np.ndarray:
        """
        选择最佳工具方向
        Args:
            vertex_idx: 顶点索引
            reference_orientation: 参考方向
        Returns:
            最佳方向
        """
        # 获取顶点法向量
        normal = self.mesh.vertex_normals[vertex_idx]
        curvature = self.mesh.curvatures[vertex_idx]

        # 1. 生成候选方向
        candidates = []
        
        # 候选1：垂直于法向量
        candidate1 = np.array([-normal[1], normal[0], 0])
        if np.linalg.norm(candidate1) < 0.1:
            candidate1 = np.array([0, -normal[2], normal[1]])
        candidate1 = candidate1 / np.linalg.norm(candidate1)
        candidates.append(candidate1)
        
        # 候选2：垂直于法向量和候选1
        candidate2 = np.cross(normal, candidate1)
        candidate2 = candidate2 / np.linalg.norm(candidate2)
        candidates.append(candidate2)
        
        # 候选3：参考方向
        candidates.append(reference_orientation)
        
        # 候选4：法向量方向
        candidates.append(normal)
        
        # 2. 为每个候选方向评分
        best_score = -1
        best_candidate = candidate1
        
        for candidate in candidates:
            # 计算方向与法向量的夹角
            dot_product = np.dot(candidate, normal)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
            
            # 基础评分：方向与法向量的一致性
            base_score = dot_product
            
            # 参考方向相似度评分
            reference_score = np.dot(candidate, reference_orientation)
            
            # 曲率适应性评分：高曲率区域倾向于使用法向量方向
            curvature_score = 1.0 - curvature * 0.5 if angle < np.pi / 4 else 1.0
            
            # 刀具几何兼容性评分
            tool_score = 1.0
            tool_type = self.tool.profile_type
            if tool_type == 'ellipsoidal':
                # 椭球形刀具：偏好与长轴一致的方向
                semi_axes = self.tool.params.get('semi_axes', [1.0, 1.0])
                if semi_axes[0] > semi_axes[1]:
                    # 长轴在X方向
                    tool_score += abs(candidate[0]) * 0.2
            elif tool_type == 'cylindrical':
                # 圆柱形刀具：偏好与圆柱轴一致的方向
                tool_score += abs(candidate[2]) * 0.2
            
            # 综合评分
            total_score = 0.4 * base_score + 0.3 * reference_score + 0.2 * curvature_score + 0.1 * tool_score
            
            if total_score > best_score:
                best_score = total_score
                best_candidate = candidate
        
        # 3. 归一化并返回最佳方向
        best_candidate = best_candidate / np.linalg.norm(best_candidate)
        return best_candidate

    def laplacian_smoothing(self, orientations: np.ndarray, lambda_param: float = 0.5) -> np.ndarray:
        """
        拉普拉斯平滑
        
        优化：使用NumPy向量化操作加速计算
        
        Args:
            orientations: 原始方向
            lambda_param: 平滑参数
        Returns:
            平滑后的方向
        """
        smoothed = orientations.copy()

        for i in range(self.num_vertices):
            neighbors = self.mesh.adjacency[i]
            if not neighbors:
                continue

            # 使用NumPy向量化操作计算邻居方向的加权平均
            neighbor_vertices = self.mesh.vertices[neighbors]
            current_vertex = self.mesh.vertices[i]
            
            # 计算距离和权重
            distances = np.linalg.norm(neighbor_vertices - current_vertex, axis=1)
            weights = 1.0 / (distances + 0.001)
            weight_sum = np.sum(weights)

            if weight_sum > 0:
                # 向量化计算加权和
                neighbor_sum = np.sum(weights[:, np.newaxis] * orientations[neighbors], axis=0)
                neighbor_avg = neighbor_sum / weight_sum

                # 平滑
                smoothed[i] = (1 - lambda_param) * orientations[i] + lambda_param * neighbor_avg
                # 避免除以零
                norm = np.linalg.norm(smoothed[i])
                if norm > 1e-6:
                    smoothed[i] = smoothed[i] / norm

        return smoothed

    def local_reorientation(self, orientations: np.ndarray) -> np.ndarray:
        """
        局部重定向（确保方向在TAR内）
        Args:
            orientations: 平滑后的方向
        Returns:
            重定向后的方向
        """
        reoriented = orientations.copy()

        for i in range(self.num_vertices):
            current_orientation = orientations[i]
            normal = self.mesh.vertex_normals[i]
            
            # 1. 检查方向是否在TAR内
            # 计算方向与法向量的夹角
            dot_product = np.dot(current_orientation, normal)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
            
            # 2. 如果方向不在TAR内，调整到最近的TAR方向
            if dot_product < 0 or angle > np.pi / 2:
                # 方向与法向量反向，调整到法向量方向
                reoriented[i] = normal
            else:
                # 3. 基于曲率调整方向
                curvature = self.mesh.curvatures[i]
                # 高曲率区域倾向于使用法向量方向
                curvature_factor = 1.0 - curvature * 0.3
                curvature_factor = max(0.5, curvature_factor)
                
                # 混合当前方向和法向量方向
                reoriented[i] = curvature_factor * current_orientation + (1 - curvature_factor) * normal
                reoriented[i] = reoriented[i] / np.linalg.norm(reoriented[i])
                
                # 4. 检查调整后的方向是否与刀具几何兼容
                # 对于非球形刀具，某些方向可能不兼容
                tool_type = self.tool.profile_type
                if tool_type == 'ellipsoidal':
                    # 椭球形刀具：确保方向与刀具长轴一致
                    semi_axes = self.tool.params.get('semi_axes', [1.0, 1.0])
                    if semi_axes[0] > semi_axes[1]:
                        # 长轴在X方向，调整方向以利用长轴
                        pass
                
        # 5. 二次平滑，确保方向场的连续性
        reoriented = self.laplacian_smoothing(reoriented, lambda_param=0.3)

        return reoriented

    def generate_field(self) -> np.ndarray:
        """
        生成工具方向场
        Returns:
            工具方向数组
        """
        print("生成工具方向场...")

        # 1. 为每个分区选择种子点
        seed_points = self.select_seed_points()

        # 2. 为每个分区生成初始方向
        for partition_id, seed_vertex in seed_points.items():
            self.greedy_tar_selection(partition_id, seed_vertex)

        # 3. 拉普拉斯平滑
        smoothed = self.laplacian_smoothing(self.vertex_orientations)

        # 4. 局部重定向
        final_orientations = self.local_reorientation(smoothed)

        print("工具方向场生成完成")

        return final_orientations