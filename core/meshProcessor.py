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
        
        # 多法向量存储，用于处理奇点处的法向量不唯一性
        # 格式：每个顶点可能有多个法向量，存储为列表
        self.multiple_normals = [[] for _ in range(len(self.vertices))]
        # 初始化：将原始法向量作为第一个法向量
        for i in range(len(self.vertices)):
            if len(self.vertex_normals) > i:
                self.multiple_normals[i].append(self.vertex_normals[i])

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
        
        # 计算主曲率和主方向
        self.principal_curvatures, self.principal_directions1, self.principal_directions2 = self._estimate_principal_curvatures()

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

    def _estimate_principal_curvatures(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        估计顶点主曲率和主方向
        
        Returns:
            tuple: (主曲率数组, 主方向1数组, 主方向2数组)
                主曲率数组形状为 (N, 2)，其中N是顶点数
                主方向数组形状为 (N, 3)，其中N是顶点数
        """
        principal_curvatures = np.zeros((len(self.vertices), 2))
        principal_directions1 = np.zeros((len(self.vertices), 3))
        principal_directions2 = np.zeros((len(self.vertices), 3))
        
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
                a, b_coeff, c, d, e, f = coeffs
                
                # 计算曲率
                H = (a + b_coeff) / 2
                K = a * b_coeff - c * c
                
                if K >= H*H:
                    k1 = H
                    k2 = H
                else:
                    sqrt_val = np.sqrt(H*H - K)
                    k1 = H + sqrt_val
                    k2 = H - sqrt_val
                
                principal_curvatures[i] = [k1, k2]
                
                # 计算主方向
                # 主方向是曲率张量的特征向量
                # 曲率张量矩阵
                curvature_tensor = np.array([[a, c], [c, b_coeff]])
                
                # 计算特征值和特征向量
                eigenvalues, eigenvectors = np.linalg.eigh(curvature_tensor)
                
                # 特征值从大到小排序
                sorted_indices = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[sorted_indices]
                eigenvectors = eigenvectors[:, sorted_indices]
                
                # 获取主方向
                # 将2D主方向转换为3D主方向
                # 使用法向量构建局部坐标系
                normal = self.vertex_normals[i]
                
                # 生成两个正交的基向量
                if abs(normal[2]) < 0.9:
                    vec1 = np.cross(normal, [0, 0, 1])
                else:
                    vec1 = np.cross(normal, [1, 0, 0])
                vec1 = vec1 / np.linalg.norm(vec1)
                vec2 = np.cross(normal, vec1)
                vec2 = vec2 / np.linalg.norm(vec2)
                
                # 将2D主方向转换为3D
                dir1_2d = eigenvectors[:, 0]
                dir2_2d = eigenvectors[:, 1]
                
                dir1_3d = dir1_2d[0] * vec1 + dir1_2d[1] * vec2
                dir2_3d = dir2_2d[0] * vec1 + dir2_2d[1] * vec2
                
                # 归一化
                dir1_3d = dir1_3d / np.linalg.norm(dir1_3d)
                dir2_3d = dir2_3d / np.linalg.norm(dir2_3d)
                
                principal_directions1[i] = dir1_3d
                principal_directions2[i] = dir2_3d
            except:
                principal_curvatures[i] = [0, 0]
                # 默认主方向
                normal = self.vertex_normals[i]
                if abs(normal[2]) < 0.9:
                    vec1 = np.cross(normal, [0, 0, 1])
                else:
                    vec1 = np.cross(normal, [1, 0, 0])
                vec1 = vec1 / np.linalg.norm(vec1)
                vec2 = np.cross(normal, vec1)
                vec2 = vec2 / np.linalg.norm(vec2)
                principal_directions1[i] = vec1
                principal_directions2[i] = vec2
        
        return principal_curvatures, principal_directions1, principal_directions2

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
    
    def add_normal(self, vertex_idx: int, normal: np.ndarray):
        """为顶点添加一个法向量
        
        Args:
            vertex_idx: 顶点索引
            normal: 法向量
        """
        if 0 <= vertex_idx < len(self.vertices):
            # 归一化法向量
            norm = np.linalg.norm(normal)
            if norm > 1e-8:
                normal = normal / norm
            # 检查是否已经存在相同的法向量
            exists = False
            for existing_normal in self.multiple_normals[vertex_idx]:
                if np.linalg.norm(existing_normal - normal) < 1e-6:
                    exists = True
                    break
            if not exists:
                self.multiple_normals[vertex_idx].append(normal)
    
    def get_normals(self, vertex_idx: int) -> List[np.ndarray]:
        """获取顶点的所有法向量
        
        Args:
            vertex_idx: 顶点索引
        Returns:
            法向量列表
        """
        if 0 <= vertex_idx < len(self.vertices):
            return self.multiple_normals[vertex_idx]
        return []
    
    def get_normal(self, vertex_idx: int, index: int = 0) -> np.ndarray:
        """
        获取顶点的指定法向量
        
        Args:
            vertex_idx: 顶点索引
            index: 法向量索引
        Returns:
            法向量
        """
        if 0 <= vertex_idx < len(self.vertices):
            normals = self.multiple_normals[vertex_idx]
            if normals:
                return normals[min(index, len(normals) - 1)]
        # 回退到原始法向量
        if 0 <= vertex_idx < len(self.vertex_normals):
            return self.vertex_normals[vertex_idx]
        return np.array([0, 0, 1])
    
    def update_mesh_normals(self):
        """
        更新Open3D网格对象的法向量
        使用multiple_normals中的第一个法向量作为网格的法向量
        """
        if hasattr(self, 'mesh') and self.mesh is not None:
            # 为每个顶点选择第一个法向量
            updated_normals = []
            for i in range(len(self.vertices)):
                normals = self.multiple_normals[i]
                if normals:
                    updated_normals.append(normals[0])
                elif i < len(self.vertex_normals):
                    updated_normals.append(self.vertex_normals[i])
                else:
                    updated_normals.append(np.array([0, 0, 1]))
            
            # 更新Open3D网格的法向量
            self.mesh.vertex_normals = o3d.utility.Vector3dVector(np.array(updated_normals))
            self.mesh.compute_triangle_normals()
            # 更新内部存储的法向量
            self.vertex_normals = np.array(updated_normals)
    
    def detect_singularities(self, curvature_threshold=1.0, normal_variation_threshold=0.5):
        """
        检测曲面上的奇点
        
        Args:
            curvature_threshold: 曲率阈值
            normal_variation_threshold: 法向量变化率阈值
        Returns:
            奇点索引列表
        """
        print("检测奇点...")
        
        singularities = []
        
        # 计算法向量变化率
        normal_variation = np.zeros(len(self.vertices))
        for i in range(len(self.vertices)):
            neighbors = self.adjacency[i]
            if len(neighbors) < 3:
                continue
            
            current_normal = self.vertex_normals[i]
            total_angle = 0.0
            for neighbor in neighbors:
                neighbor_normal = self.vertex_normals[neighbor]
                dot_product = np.dot(current_normal, neighbor_normal)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = np.arccos(dot_product)
                total_angle += angle
            normal_variation[i] = total_angle / len(neighbors)
        
        # 检测奇点
        for i in range(len(self.vertices)):
            # 条件1: 曲率超过阈值
            if hasattr(self, 'curvatures') and self.curvatures[i] > curvature_threshold:
                singularities.append(i)
                continue
            
            # 条件2: 法向量变化率超过阈值
            if normal_variation[i] > normal_variation_threshold:
                singularities.append(i)
                continue
            
            # 条件3: 主曲率异常
            if hasattr(self, 'principal_curvatures'):
                k1, k2 = self.principal_curvatures[i]
                # 检查主曲率是否异常大
                if abs(k1) > 10.0 or abs(k2) > 10.0:
                    singularities.append(i)
                    continue
                # 检查主曲率比值是否异常
                if abs(k1) > 1e-6 and abs(k2) > 1e-6:
                    ratio = max(abs(k1/k2), abs(k2/k1))
                    if ratio > 10.0:
                        singularities.append(i)
                        continue
            
            # 条件4: 拓扑结构异常（顶点度异常）
            neighbors = self.adjacency[i]
            if len(neighbors) < 3 or len(neighbors) > 8:
                singularities.append(i)
                continue
        
        print(f"检测到 {len(singularities)} 个奇点")
        return singularities
    
    def analyze_singularity_type(self, vertex_idx):
        """
        分析奇点类型
        
        Args:
            vertex_idx: 顶点索引
        Returns:
            奇点类型字符串
        """
        if vertex_idx >= len(self.vertices):
            return "invalid"
        
        # 分析曲率
        if hasattr(self, 'curvatures'):
            curvature = self.curvatures[vertex_idx]
        else:
            curvature = 0.0
        
        # 分析法向量变化率
        neighbors = self.adjacency[vertex_idx]
        normal_variation = 0.0
        if len(neighbors) >= 3:
            current_normal = self.vertex_normals[vertex_idx]
            total_angle = 0.0
            for neighbor in neighbors:
                neighbor_normal = self.vertex_normals[neighbor]
                dot_product = np.dot(current_normal, neighbor_normal)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = np.arccos(dot_product)
                total_angle += angle
            normal_variation = total_angle / len(neighbors)
        
        # 分析主曲率
        k1, k2 = 0.0, 0.0
        if hasattr(self, 'principal_curvatures'):
            k1, k2 = self.principal_curvatures[vertex_idx]
        
        # 分析拓扑结构
        degree = len(neighbors)
        
        # 判定奇点类型
        if curvature > 1.0:
            if abs(k1) > abs(k2) * 10:
                return "sharp_edge"
            elif abs(k2) > abs(k1) * 10:
                return "sharp_edge"
            else:
                return "sharp_vertex"
        elif normal_variation > 0.5:
            return "normal_discontinuity"
        elif degree < 3:
            return "low_degree"
        elif degree > 8:
            return "high_degree"
        elif abs(k1) > 10.0 or abs(k2) > 10.0:
            return "extreme_curvature"
        else:
            return "unknown"
    
    def get_singularity_info(self):
        """
        获取所有奇点的信息
        
        Returns:
            奇点信息字典
        """
        singularities = self.detect_singularities()
        singularity_info = {}
        
        for idx in singularities:
            singularity_type = self.analyze_singularity_type(idx)
            singularity_info[idx] = {
                'type': singularity_type,
                'position': self.vertices[idx].tolist(),
                'curvature': self.curvatures[idx] if hasattr(self, 'curvatures') else 0.0,
                'normal': self.vertex_normals[idx].tolist(),
                'num_normals': len(self.multiple_normals[idx])
            }
        
        return singularity_info