"""
几何计算工具
"""

import numpy as np
import math
from typing import Tuple, List, Optional
from scipy.spatial import KDTree


class GeometryTools:
    @staticmethod
    def normalize_vector(v: np.ndarray) -> np.ndarray:
        """归一化向量"""
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v

    @staticmethod
    def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        """计算两个向量之间的夹角（弧度）"""
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cos_angle = dot / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return math.acos(cos_angle)

    @staticmethod
    def rotate_vector(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """绕轴旋转向量"""
        # 罗德里格旋转公式
        axis = GeometryTools.normalize_vector(axis)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        return (v * cos_a +
                np.cross(axis, v) * sin_a +
                axis * np.dot(axis, v) * (1 - cos_a))

    @staticmethod
    def project_point_to_plane(point: np.ndarray,
                               plane_point: np.ndarray,
                               plane_normal: np.ndarray) -> np.ndarray:
        """将点投影到平面上"""
        normal = GeometryTools.normalize_vector(plane_normal)
        vector = point - plane_point
        distance = np.dot(vector, normal)
        return point - distance * normal

    @staticmethod
    def compute_mesh_laplacian(vertices: np.ndarray,
                               adjacency: List[List[int]]) -> np.ndarray:
        """计算网格拉普拉斯矩阵"""
        n = len(vertices)
        L = np.zeros((n, n))

        for i in range(n):
            neighbors = adjacency[i]
            weight_sum = 0

            for j in neighbors:
                # 简单权重：1/距离
                distance = np.linalg.norm(vertices[i] - vertices[j])
                weight = 1.0 / (distance + 1e-6)

                L[i, j] = -weight
                weight_sum += weight

            L[i, i] = weight_sum

        return L

    @staticmethod
    def find_nearest_point(point: np.ndarray, points: np.ndarray) -> Tuple[int, float]:
        """找到最近的点"""
        distances = np.linalg.norm(points - point, axis=1)
        idx = np.argmin(distances)
        return idx, distances[idx]

    @staticmethod
    def smooth_scalar_field(field: np.ndarray,
                            adjacency: List[List[int]],
                            iterations: int = 10,
                            lambda_param: float = 0.5) -> np.ndarray:
        """平滑标量场"""
        smoothed = field.copy()

        for _ in range(iterations):
            new_field = smoothed.copy()

            for i in range(len(smoothed)):
                neighbors = adjacency[i]
                if not neighbors:
                    continue

                # 计算邻居平均值
                neighbor_sum = 0
                for j in neighbors:
                    neighbor_sum += smoothed[j]
                neighbor_avg = neighbor_sum / len(neighbors)

                # 平滑
                new_field[i] = (1 - lambda_param) * smoothed[i] + lambda_param * neighbor_avg

            smoothed = new_field

        return smoothed

    @staticmethod
    def compute_face_normal(vertices: np.ndarray) -> np.ndarray:
        """计算面的法向量（假设是三角形）"""
        if len(vertices) != 3:
            raise ValueError("仅支持三角形")

        v0, v1, v2 = vertices
        normal = np.cross(v1 - v0, v2 - v0)
        return GeometryTools.normalize_vector(normal)

    @staticmethod
    def barycentric_coordinates(point: np.ndarray,
                                triangle: np.ndarray) -> Tuple[float, float, float]:
        """计算点在三角形中的重心坐标"""
        v0, v1, v2 = triangle

        # 计算面积
        area_total = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        # 计算子三角形面积
        area_u = 0.5 * np.linalg.norm(np.cross(v1 - point, v2 - point))
        area_v = 0.5 * np.linalg.norm(np.cross(v2 - point, v0 - point))
        area_w = 0.5 * np.linalg.norm(np.cross(v0 - point, v1 - point))

        u = area_u / area_total
        v = area_v / area_total
        w = area_w / area_total

        return u, v, w