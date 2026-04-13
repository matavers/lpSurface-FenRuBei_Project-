"""
网格处理器
处理网格数据，计算几何属性
支持Open3D网格对象
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import math
import open3d as o3d


class MeshProcessor:
    def __init__(self, mesh):
        """
        初始化网格处理器
        Args:
            mesh: Open3D网格对象
        """
        self.mesh = mesh

        # 提取网格数据
        self._extract_mesh_data()

        # 计算几何属性
        self._compute_geometry()

        print(f"网格处理完成: {len(self.vertices)} 个顶点, {len(self.faces)} 个面")

    def _extract_mesh_data(self):
        """从Open3D对象提取网格数据"""
        # 顶点数据
        # asarray方法将输入转化为array类型
        self.vertices = np.asarray(self.mesh.vertices)
        self.vertex_normals = np.asarray(self.mesh.vertex_normals)

        # 面数据
        self.faces = np.asarray(self.mesh.triangles)
        self.face_normals = np.asarray(self.mesh.triangle_normals)

        # 构建邻接关系
        self._build_adjacency()

    def _build_adjacency(self):
        """构建顶点邻接关系
        
        优化措施：
        1. 使用NumPy向量化操作处理边数据
        2. 高效去重：使用结构化数组和np.unique进行边去重
        3. 批量处理：减少Python循环，提高处理速度
        4. 内存优化：预分配邻接表，减少动态扩容
        """
        # 预分配邻接表
        self.adjacency = [[] for _ in range(len(self.vertices))]
        
        # 使用NumPy向量化操作处理所有面
        # 为每个面生成三条边
        faces = self.faces
        n_faces = len(faces)
        
        # 创建边数组：每个面3条边，共n_faces*3条边
        edges = np.empty((n_faces * 3, 2), dtype=int)
        
        # 填充边数组
        for i, face in enumerate(faces):
            v0, v1, v2 = face
            # 生成三条边
            edges[i*3] = sorted([v0, v1])
            edges[i*3+1] = sorted([v1, v2])
            edges[i*3+2] = sorted([v2, v0])
        
        # 边去重
        # 使用结构化数组进行高效去重
        edges_struct = np.core.records.fromarrays(edges.T, dtype=[('v0', 'i4'), ('v1', 'i4')])
        unique_edges_struct, indices = np.unique(edges_struct, return_index=True)
        
        # 转换回普通数组
        unique_edges = np.array([(e[0], e[1]) for e in unique_edges_struct])
        
        # 构建邻接表和边列表
        self.edge_vertices = []
        for edge in unique_edges:
            v_start, v_end = edge
            self.adjacency[v_start].append(v_end)
            self.adjacency[v_end].append(v_start)
            self.edge_vertices.append(edge)

    def _compute_geometry(self):
        """计算几何属性"""
        # 计算顶点曲率（简化版本）
        self.curvatures = self._estimate_curvatures()
        
        # 计算主曲率
        self.principal_curvatures = self._estimate_principal_curvatures()

        # 计算面面积
        self.face_areas = self._compute_face_areas()

        # 计算顶点面积（相邻面面积的平均）
        self.vertex_areas = np.zeros(len(self.vertices))
        for i in range(len(self.vertices)):
            area_sum = 0
            count = 0
            # 找到包含该顶点的面
            for face_idx, face in enumerate(self.faces):
                if i in face:
                    area_sum += self.face_areas[face_idx]
                    count += 1
            if count > 0:
                self.vertex_areas[i] = area_sum / count
        
        # 初始化最大切削宽度和直纹面逼近误差
        self.max_cutting_widths = np.zeros(len(self.vertices))
        self.rolled_error = np.zeros(len(self.vertices))

    def _estimate_curvatures(self) -> np.ndarray:
        """估计顶点曲率（完整实现）
        
        使用离散曲率计算方法，考虑顶点的局部几何形状
        
        Returns:
            顶点曲率数组
        """
        curvatures = np.zeros(len(self.vertices))

        # 检查法线是否存在且足够
        has_normals = len(self.vertex_normals) == len(self.vertices) and np.any(self.vertex_normals)

        for i in range(len(self.vertices)):
            # 获取邻居顶点
            neighbors = self.adjacency[i]
            if len(neighbors) < 3:  # 需要至少3个邻居来计算曲率
                continue

            # 方法2: 使用顶点位置计算离散曲率
            # 计算顶点i到所有邻居的平均距离
            vertex_pos = self.vertices[i]
            neighbor_positions = self.vertices[neighbors]
            distances = np.linalg.norm(neighbor_positions - vertex_pos, axis=1)
            avg_distance = np.mean(distances)
            
            # 计算邻居顶点的中心
            centroid = np.mean(neighbor_positions, axis=0)
            
            # 计算离散曲率
            # 曲率 = 2 * 高度 / (半径^2)
            height = np.linalg.norm(centroid - vertex_pos)
            discrete_curvature = 2 * height / (avg_distance ** 2 + 1e-8)
            
            if has_normals:
                # 方法1: 使用法向量变化计算平均曲率
                normal_i = self.vertex_normals[i]
                neighbor_normals = self.vertex_normals[neighbors]
                dots = np.clip(np.dot(neighbor_normals, normal_i), -1.0, 1.0)
                angles = np.arccos(dots)
                normal_variation = np.sum(angles) / len(neighbors)
                
                # 综合两种方法的结果
                # 使用权重平均
                curvatures[i] = 0.6 * normal_variation + 0.4 * discrete_curvature
            else:
                # 只使用离散曲率
                curvatures[i] = discrete_curvature

        # 归一化
        max_curvature = np.max(curvatures)
        if max_curvature > 0:
            curvatures = curvatures / max_curvature
        
        # 平滑处理
        # 使用简单的移动平均平滑曲率值
        smoothed_curvatures = curvatures.copy()
        for i in range(len(self.vertices)):
            neighbors = self.adjacency[i]
            if neighbors:
                # 计算邻居的平均曲率
                neighbor_curvatures = curvatures[neighbors]
                avg_neighbor_curvature = np.mean(neighbor_curvatures)
                # 平滑当前顶点的曲率
                smoothed_curvatures[i] = 0.7 * curvatures[i] + 0.3 * avg_neighbor_curvature

        return smoothed_curvatures

    def _estimate_principal_curvatures(self) -> np.ndarray:
        """估计顶点主曲率
        
        Returns:
            主曲率数组，形状为 (N, 2)，其中N是顶点数
        """
        principal_curvatures = np.zeros((len(self.vertices), 2))
        
        for i in range(len(self.vertices)):
            neighbors = self.adjacency[i]
            if len(neighbors) < 3:
                continue
            
            # 使用邻域点拟合二次曲面
            vertex_pos = self.vertices[i]
            neighbor_positions = self.vertices[neighbors]
            
            # 构建方程组
            A = []
            b = []
            
            for p in neighbor_positions:
                x, y, z = p - vertex_pos
                A.append([x*x, y*y, x*y, x, y, 1])
                b.append(z)
            
            A = np.array(A)
            b = np.array(b)
            
            # 最小二乘拟合
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                a, b, c, d, e, f = coeffs
                
                # 计算曲率
                H = (a + b) / 2
                K = a * b - c * c
                
                if K >= H*H:
                    k1 = H
                    k2 = H
                else:
                    sqrt_val = np.sqrt(H*H - K)
                    k1 = H + sqrt_val
                    k2 = H - sqrt_val
                
                principal_curvatures[i] = [k1, k2]
            except:
                principal_curvatures[i] = [0, 0]
        
        return principal_curvatures

    def _compute_face_areas(self) -> np.ndarray:
        """计算面面积
        
        优化：使用NumPy向量化操作批量计算所有面的面积，提高计算效率
        """
        # 使用NumPy向量化操作批量计算所有面的面积
        faces = self.faces
        # 获取每个面的三个顶点
        v0 = self.vertices[faces[:, 0]]
        v1 = self.vertices[faces[:, 1]]
        v2 = self.vertices[faces[:, 2]]
        
        # 计算叉积并求范数
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        return areas

    def get_face_containing_vertex(self, vertex_idx: int) -> List[int]:
        """获取包含指定顶点的面索引"""
        face_indices = []
        for i, face in enumerate(self.faces):
            if vertex_idx in face:
                face_indices.append(i)
        return face_indices
    
    def calculate_max_cutting_width(self, tool):
        """计算每个顶点的最大切削宽度
        
        Args:
            tool: 刀具对象，包含刀具几何参数
        """
        from .nonSphericalTool import NonSphericalTool
        
        if not isinstance(tool, NonSphericalTool):
            raise ValueError("工具必须是NonSphericalTool类型")
        
        # 检查法线是否存在且足够
        has_normals = len(self.vertex_normals) == len(self.vertices) and np.any(self.vertex_normals)
        
        for i in range(len(self.vertices)):
            vertex_pos = self.vertices[i]
            
            # 计算最大切削宽度
            max_width = 0
            best_direction = None
            
            if has_normals:
                vertex_normal = self.vertex_normals[i]
                # 采样不同的刀具方向
                # 简化实现：只考虑法向量方向
                direction = vertex_normal
                
                # 计算该方向的切削宽度
                try:
                    width = tool.calculate_cutting_width(
                        vertex_pos,
                        vertex_normal,
                        direction,
                        scallop_height=0.4  # 默认残留高度
                    )
                    max_width = width
                    best_direction = direction
                except:
                    pass
            
            self.max_cutting_widths[i] = max_width
    
    def calculate_rolled_error(self):
        """计算每个顶点的直纹面逼近误差
        
        直纹面逼近误差反映了曲面局部偏离可展曲面的程度
        """
        # 检查法线是否存在且足够
        has_normals = len(self.vertex_normals) == len(self.vertices) and np.any(self.vertex_normals)
        
        for i in range(len(self.vertices)):
            # 获取顶点的一环邻域
            neighbors = self.adjacency[i]
            if len(neighbors) < 3:
                self.rolled_error[i] = 0
                continue
            
            # 计算邻域的平均曲率
            vertex_pos = self.vertices[i]
            neighbor_positions = self.vertices[neighbors]
            
            # 拟合直纹面
            # 简化实现：计算邻域的平均法向量
            if has_normals:
                avg_normal = np.mean(self.vertex_normals[neighbors], axis=0)
                avg_normal /= np.linalg.norm(avg_normal)
            else:
                # 如果没有法线，使用默认值
                avg_normal = np.array([0, 0, 1])
            
            # 计算直纹面的曲率
            # 简化实现：使用平均法向量计算曲率
            # 这里使用一个简化的方法，实际实现可能需要更复杂的算法
            rolled_curvature = 0
            
            # 计算原始曲面的平均曲率
            k1, k2 = self.principal_curvatures[i]
            original_curvature = (k1 + k2) / 2
            
            # 计算误差
            error = abs(original_curvature - rolled_curvature)
            self.rolled_error[i] = error