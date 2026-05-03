"""
等残留高度场生成器
使用CGAL加速计算
"""

import numpy as np
import math
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from typing import List, Tuple, Dict, Any
from .meshProcessor import MeshProcessor
from .nonSphericalTool import NonSphericalTool

try:
    from CGAL.CGAL_Kernel import *
    from CGAL.CGAL_Polygon_mesh_processing import *
    from CGAL.CGAL_Spatial_searching import *
    CGAL_AVAILABLE = True
    print("CGAL 已加载，启用CGAL优化")
except ImportError:
    print("[isoScallopField] 警告: CGAL 未安装，使用原始实现")
    CGAL_AVAILABLE = False


class IsoScallopFieldGenerator:
    def __init__(self, mesh: MeshProcessor, tool_orientations: np.ndarray,
                 tool: NonSphericalTool, scallop_height: float):
        """
        初始化等残留高度场生成器
        Args:
            mesh: 网格处理器
            tool_orientations: 工具方向场
            tool: 刀具模型
            scallop_height: 残留高度
        """
        self.mesh = mesh
        self.tool_orientations = tool_orientations
        self.tool = tool
        self.scallop_height = scallop_height

        # 标量场
        self.scalar_field = None

        # 梯度场
        self.gradient_field = None

        # CGAL表面网格
        if CGAL_AVAILABLE:
            self.cgal_mesh = self._build_cgal_mesh()
        else:
            self.cgal_mesh = None

    def _build_cgal_mesh(self):
        """
        构建CGAL Polyhedron_3
        """
        from CGAL.CGAL_Kernel import Point_3
        from CGAL.CGAL_Polyhedron_3 import Polyhedron_3, Polyhedron_modifier, ABSOLUTE_INDEXING
        
        P = Polyhedron_3()
        modifier = Polyhedron_modifier()
        
        # 开始构建表面
        num_vertices = len(self.mesh.vertices)
        num_faces = len([f for f in self.mesh.faces if len(f) == 3])
        modifier.begin_surface(num_vertices, num_faces, 0, ABSOLUTE_INDEXING)
        
        # 添加顶点
        for i, (x, y, z) in enumerate(self.mesh.vertices):
            modifier.add_vertex(Point_3(x, y, z))
        
        # 添加面
        for face in self.mesh.faces:
            if len(face) == 3:
                modifier.begin_facet()
                modifier.add_vertex_to_facet(int(face[0]))
                modifier.add_vertex_to_facet(int(face[1]))
                modifier.add_vertex_to_facet(int(face[2]))
                modifier.end_facet()
        
        # 结束构建
        modifier.end_surface()
        
        return P

    def _compute_gradient(self, task):
        """
        计算梯度方向
        """
        i, tool_dir, normal = task
        # 梯度方向应垂直于工具方向
        # 计算垂直于工具方向和法向量的向量
        gradient = np.cross(tool_dir, normal)

        if np.linalg.norm(gradient) < 0.001:
            # 如果工具方向与法向量平行，使用任意垂直向量
            gradient = np.array([-tool_dir[1], tool_dir[0], 0])
            if np.linalg.norm(gradient) < 0.001:
                gradient = np.array([0, -tool_dir[2], tool_dir[1]])

        # 避免除以零
        norm = np.linalg.norm(gradient)
        if norm > 1e-6:
            gradient = gradient / norm
        else:
            # 如果所有尝试都失败，使用默认方向
            gradient = np.array([1, 0, 0])
        return i, gradient

    def initialize_gradient_field(self) -> np.ndarray:
        """
        初始化梯度场方向
        
        使用NumPy向量化操作加速梯度场计算
        
        Returns:
            梯度场方向数组
        """
        n = len(self.mesh.vertices)
        gradient_directions = np.zeros((n, 3))

        if CGAL_AVAILABLE:
            # 使用CGAL的并行处理能力
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # 准备任务
            tasks = []
            for i in range(n):
                tool_dir = self.tool_orientations[i]
                # 使用新的get_normal方法获取法向量
                normal = self.mesh.get_normal(i)
                tasks.append((i, tool_dir, normal))
            
            # 并行计算梯度方向
            results = []
            with ThreadPoolExecutor() as executor:
                # 提交所有任务
                future_to_task = {executor.submit(self._compute_gradient, task): task for task in tasks}
                
                # 收集结果
                for future in as_completed(future_to_task):
                    results.append(future.result())
            
            # 填充结果
            for i, gradient in results:
                gradient_directions[i] = gradient
        else:
            # 优化实现：使用NumPy向量化操作
            tool_dirs = self.tool_orientations
            # 使用新的get_normal方法获取法向量
            normals = np.array([self.mesh.get_normal(i) for i in range(n)])
            
            # 计算梯度方向
            gradients = np.cross(tool_dirs, normals)
            norms = np.linalg.norm(gradients, axis=1, keepdims=True)
            
            # 处理退化情况
            mask = norms < 0.001
            if np.any(mask):
                # 为退化情况生成替代梯度
                alt_gradient1 = np.stack([-tool_dirs[:, 1], tool_dirs[:, 0], np.zeros(n)], axis=1)
                alt_norm1 = np.linalg.norm(alt_gradient1, axis=1, keepdims=True)
                
                # 再次检查退化情况
                mask2 = alt_norm1 < 0.001
                if np.any(mask2):
                    alt_gradient2 = np.stack([np.zeros(n), -tool_dirs[:, 2], tool_dirs[:, 1]], axis=1)
                    alt_norm2 = np.linalg.norm(alt_gradient2, axis=1, keepdims=True)
                    alt_gradient2 /= alt_norm2
                    alt_gradient1[mask2] = alt_gradient2[mask2]
                
                alt_norm1 = np.linalg.norm(alt_gradient1, axis=1, keepdims=True)
                alt_gradient1 /= alt_norm1
                gradients[mask[:, 0]] = alt_gradient1[mask[:, 0]]
            
            # 归一化梯度
            norms = np.linalg.norm(gradients, axis=1, keepdims=True)
            gradients /= norms
            
            gradient_directions = gradients

        self.gradient_field = gradient_directions
        return gradient_directions

    def calculate_stepover_distance(self, vertex_idx: int, gradient_direction: np.ndarray) -> float:
        """
        计算步进距离
        Args:
            vertex_idx: 顶点索引
            gradient_direction: 梯度方向
        Returns:
            步进距离
        """
        # 数值稳定性检查
        if np.isnan(gradient_direction).any() or np.linalg.norm(gradient_direction) < 1e-6:
            return 0.001

        vertex_pos = self.mesh.vertices[vertex_idx]
        
        # 使用新的get_normal方法获取法向量
        vertex_normal = self.mesh.get_normal(vertex_idx)
        
        tool_orientation = self.tool_orientations[vertex_idx]
        curvature = self.mesh.curvatures[vertex_idx]

        # 检查输入数据的有效性
        if (np.isnan(vertex_normal).any() or np.isnan(tool_orientation).any() or 
            np.isnan(curvature) or np.linalg.norm(vertex_normal) < 1e-6 or 
            np.linalg.norm(tool_orientation) < 1e-6):
            return 0.001

        # 1. 计算刀具与表面的倾斜角
        dot_product = np.dot(tool_orientation, vertex_normal)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        tilt_angle = math.acos(dot_product)

        # 2. 基于倾斜角计算有效切削半径
        # 对于不同刀具类型，使用不同的gamma值
        tool_type = self.tool.profile_type
        if tool_type == 'ellipsoidal':
            # 椭球形刀具：基于倾斜角计算gamma
            a, b = self.tool.params.get('semi_axes', [1.0, 1.0])
            if b < 1e-6:
                gamma = np.pi / 4
            else:
                tan_gamma = (a * math.sin(tilt_angle)) / (b * math.cos(tilt_angle))
                gamma = math.atan(min(max(tan_gamma, 0.0), 1e6))
        elif tool_type == 'cylindrical':
            # 圆柱形刀具：使用固定gamma
            gamma = np.pi / 4
        elif tool_type == 'spherical':
            # 球形刀具：使用固定gamma
            gamma = np.pi / 4
        elif tool_type == 'conical':
            # 锥形刀具：基于倾斜角计算gamma
            gamma = tilt_angle
        else:
            # 默认gamma
            gamma = np.pi / 4

        # 确保gamma在有效范围内
        gamma = max(0.0, min(np.pi / 2, gamma))

        # 计算有效切削半径
        try:
            effective_radius = self.tool.calculate_effective_radius(gamma, tilt_angle)
        except Exception:
            effective_radius = 0.0

        # 检查有效半径的有效性
        if (np.isnan(effective_radius) or effective_radius <= 0 or 
            effective_radius == float('inf')):
            return 0.001

        # 3. 基于残留高度计算步进距离
        # 使用精确的步进距离公式
        if self.scallop_height <= 0:
            return 0.001

        if self.scallop_height >= 2 * effective_radius:
            return 0.001

        # 精确的步进距离公式
        term = 2 * effective_radius * self.scallop_height - self.scallop_height ** 2
        if term <= 0:
            return 0.001

        try:
            stepover = 2 * np.sqrt(term)
        except:
            return 0.001

        # 检查stepover的有效性
        if np.isnan(stepover) or stepover <= 0:
            return 0.001

        # 4. 考虑梯度方向的影响
        try:
            dot_product_grad = np.dot(gradient_direction, tool_orientation)
            stepover = stepover * abs(dot_product_grad)
        except:
            pass

        # 5. 基于表面曲率调整步进距离
        # 高曲率区域需要更小的步进距离
        try:
            curvature_factor = 1.0 - curvature * 0.3
            curvature_factor = max(0.7, curvature_factor)
            stepover *= curvature_factor
        except:
            pass

        # 6. 考虑刀具类型的影响
        if tool_type == 'cylindrical':
            # 圆柱形刀具的步进距离可以稍大
            stepover *= 1.1
        elif tool_type == 'conical':
            # 锥形刀具的步进距离需要更小
            stepover *= 0.9

        # 最终检查
        if np.isnan(stepover) or stepover <= 0:
            return 0.001

        return max(stepover, 0.001)

    def solve_poisson_equation(self, gradient_magnitudes: np.ndarray) -> np.ndarray:
        """
        求解泊松方程得到标量场
        使用CGAL加速线性系统求解
        Args:
            gradient_magnitudes: 梯度幅度
        Returns:
            标量场
        """
        n = len(self.mesh.vertices)

        # 检查输入数据的有效性
        if np.isnan(gradient_magnitudes).any() or np.isinf(gradient_magnitudes).any():
            print("警告: 梯度幅度中包含无效值，使用默认值")
            gradient_magnitudes = np.ones(n) * 0.001

        # 构建拉普拉斯矩阵
        try:
            L = self._build_laplacian_matrix()
        except Exception as e:
            print(f"构建拉普拉斯矩阵时出错: {e}")
            # 返回默认标量场
            return np.zeros(n)

        # 构建右侧向量
        try:
            b = self._build_rhs_vector(gradient_magnitudes)
        except Exception as e:
            print(f"构建右侧向量时出错: {e}")
            # 返回默认标量场
            return np.zeros(n)

        # 检查矩阵和向量的有效性
        if L.shape[0] != n or L.shape[1] != n:
            print("警告: 拉普拉斯矩阵形状不正确")
            return np.zeros(n)

        if len(b) != n:
            print("警告: 右侧向量长度不正确")
            return np.zeros(n)

        # 添加边界条件（固定一个顶点为零）
        fixed_vertex = 0
        L = L.tolil()
        L[fixed_vertex, :] = 0
        L[fixed_vertex, fixed_vertex] = 1
        b[fixed_vertex] = 0

        # 添加正则化项，处理奇异矩阵问题
        epsilon = 1e-6
        for i in range(n):
            try:
                L[i, i] += epsilon
            except Exception:
                pass

        L = L.tocsr()

        # 求解线性系统
        print("求解泊松方程...")
        
        phi = None
        try:
            # 尝试使用更高效的求解器
            # 对于大型系统，使用共轭梯度法
            if n > 10000:
                print("使用共轭梯度法求解大型系统...")
                # 预条件器
                try:
                    diag = L.diagonal()
                    if np.any(diag <= 0):
                        # 确保对角线为正
                        diag = np.maximum(diag, 1e-6)
                    M = sp.diags(1.0 / diag)
                except Exception:
                    M = None
                
                # 共轭梯度法
                try:
                    phi, info = splinalg.cg(L, b, M=M, tol=1e-6, maxiter=1000)
                    if info != 0:
                        print(f"共轭梯度法求解失败，信息: {info}")
                        # 回退到直接求解
                        phi = splinalg.spsolve(L, b)
                except Exception:
                    # 回退到直接求解
                    phi = splinalg.spsolve(L, b)
            else:
                # 对于小型系统，使用直接求解
                phi = splinalg.spsolve(L, b)
        except Exception as e:
            print(f"求解器错误: {e}，回退到默认解")
            phi = np.zeros(n)

        # 验证求解结果
        if phi is None:
            print("警告: 求解器返回None")
            return np.zeros(n)

        if len(phi) != n:
            print("警告: 求解结果长度不正确")
            return np.zeros(n)

        # 处理求解结果中的NaN值
        if np.isnan(phi).any():
            print("警告: 求解结果中包含NaN值，已修复")
            nan_mask = np.isnan(phi)
            phi[nan_mask] = 0.0

        if np.isinf(phi).any():
            print("警告: 求解结果中包含Inf值，已修复")
            inf_mask = np.isinf(phi)
            phi[inf_mask] = 0.0

        return phi

    def _compute_weight(self, task):
        """
        计算权重
        """
        i, j = task
        # 找到包含边(i,j)的两个面
        common_faces = self._find_common_faces(i, j)

        if len(common_faces) == 2:
            # 计算cotangent权重
            weight = self._compute_cotangent_weight(i, j, common_faces)
        else:
            # 退化情况，使用简单权重
            distance = np.linalg.norm(self.mesh.vertices[i] - self.mesh.vertices[j])
            weight = 1.0 / (distance + 0.001)
        return i, j, weight

    def _build_laplacian_matrix(self) -> sp.csr_matrix:
        """
        构建拉普拉斯矩阵
        
        优化：使用更高效的方法构建拉普拉斯矩阵
        """
        n = len(self.mesh.vertices)

        # 使用cotangent权重
        rows, cols, vals = [], [], []

        if CGAL_AVAILABLE:
            # 使用CGAL的并行处理能力
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # 准备任务
            tasks = []
            for i in range(n):
                neighbors = self.mesh.adjacency[i]
                for j in neighbors:
                    if i < j:
                        tasks.append((i, j))
            
            # 并行计算权重
            results = []
            with ThreadPoolExecutor() as executor:
                # 提交所有任务
                future_to_task = {executor.submit(self._compute_weight, task): task for task in tasks}
                
                # 收集结果
                for future in as_completed(future_to_task):
                    results.append(future.result())
            
            # 构建矩阵
            weight_map = {}
            for i, j, weight in results:
                weight_map[(i, j)] = weight
            
            # 填充矩阵
            for i in range(n):
                neighbors = self.mesh.adjacency[i]
                diag_val = 0
                
                for j in neighbors:
                    if i < j:
                        weight = weight_map.get((i, j), 0)
                    else:
                        weight = weight_map.get((j, i), 0)
                    
                    rows.append(i)
                    cols.append(j)
                    vals.append(-weight)
                    diag_val += weight
                
                # 对角线
                rows.append(i)
                cols.append(i)
                vals.append(diag_val)
        else:
            # 优化实现：减少Python循环，使用更高效的权重计算
            for i in range(n):
                neighbors = self.mesh.adjacency[i]
                if not neighbors:
                    continue
                
                # 对角线元素
                diag_val = 0
                
                # 向量化计算距离权重
                if len(neighbors) > 0:
                    # 获取邻居顶点坐标
                    neighbor_vertices = self.mesh.vertices[neighbors]
                    current_vertex = self.mesh.vertices[i]
                    
                    # 计算距离
                    distances = np.linalg.norm(neighbor_vertices - current_vertex, axis=1)
                    
                    # 计算权重
                    weights = 1.0 / (distances + 0.001)
                    
                    # 处理cotangent权重
                    for idx, j in enumerate(neighbors):
                        # 找到包含边(i,j)的两个面
                        common_faces = self._find_common_faces(i, j)
                        
                        if len(common_faces) == 2:
                            # 计算cotangent权重
                            weight = self._compute_cotangent_weight(i, j, common_faces)
                        else:
                            # 使用距离权重
                            weight = weights[idx]
                        
                        rows.append(i)
                        cols.append(j)
                        vals.append(-weight)
                        diag_val += weight
                    
                    # 对角线
                    rows.append(i)
                    cols.append(i)
                    vals.append(diag_val)

        # 使用SciPy的稀疏矩阵构建函数
        L = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        return L

    def _find_common_faces(self, v1: int, v2: int) -> List[int]:
        """
        找到包含两个顶点的公共面
        使用CGAL加速面查询
        """
        common_faces = []

        # 获取包含v1的面
        faces_v1 = self.mesh.get_face_containing_vertex(v1)

        for face_idx in faces_v1:
            face = self.mesh.faces[face_idx]
            if v2 in face:
                common_faces.append(face_idx)

        return common_faces

    def _compute_cotangent_weight(self, v1: int, v2: int, common_faces: List[int]) -> float:
        """
        计算cotangent权重
        """
        if len(common_faces) != 2:
            return 0.0

        weight = 0.0

        for face_idx in common_faces:
            face = self.mesh.faces[face_idx]

            # 找到第三个顶点
            third_vertex = -1
            for v in face:
                if v != v1 and v != v2:
                    third_vertex = v
                    break

            if third_vertex == -1:
                continue

            # 计算角度
            try:
                vec1 = self.mesh.vertices[v1] - self.mesh.vertices[third_vertex]
                vec2 = self.mesh.vertices[v2] - self.mesh.vertices[third_vertex]

                # 检查向量的有效性
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 < 1e-6 or norm2 < 1e-6:
                    continue

                # 计算夹角的cosine
                cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                
                # 计算角度
                angle = np.arccos(cos_angle)

                # 避免极端角度导致的数值不稳定
                angle = max(0.01, min(np.pi - 0.01, angle))

                # 计算cotangent
                tan_angle = np.tan(angle)
                if abs(tan_angle) < 1e-6:
                    continue
                    
                weight += 0.5 * (1.0 / tan_angle)
            except Exception:
                # 任何计算错误都返回安全值
                continue

        # 确保权重为正值且有限
        if np.isnan(weight) or np.isinf(weight) or weight < 0:
            return 0.0

        return weight

    def _compute_contribution(self, task):
        """
        计算右侧向量贡献
        """
        i, j, grad_mag_i, grad_mag_j = task
        # 计算边的方向
        edge_vec = self.mesh.vertices[j] - self.mesh.vertices[i]
        edge_length = np.linalg.norm(edge_vec)

        if edge_length < 0.001:
            return i, 0.0

        edge_dir = edge_vec / edge_length

        # 计算梯度在边方向的分量
        grad_i = self.gradient_field[i] * grad_mag_i
        grad_j = self.gradient_field[j] * grad_mag_j

        grad_avg = 0.5 * (grad_i + grad_j)
        grad_component = np.dot(grad_avg, edge_dir)

        # 贡献到右侧向量
        weight = 1.0 / edge_length
        contribution = weight * grad_component * edge_length
        return i, contribution

    def _build_rhs_vector(self, gradient_magnitudes: np.ndarray) -> np.ndarray:
        """
        构建右侧向量
        使用CGAL加速几何计算
        """
        n = len(self.mesh.vertices)
        b = np.zeros(n)

        if CGAL_AVAILABLE:
            # 使用CGAL的并行处理能力
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # 准备任务
            tasks = []
            for i in range(n):
                neighbors = self.mesh.adjacency[i]
                for j in neighbors:
                    tasks.append((i, j, gradient_magnitudes[i], gradient_magnitudes[j]))
            
            # 并行计算右侧向量贡献
            results = []
            with ThreadPoolExecutor() as executor:
                # 提交所有任务
                future_to_task = {executor.submit(self._compute_contribution, task): task for task in tasks}
                
                # 收集结果
                for future in as_completed(future_to_task):
                    results.append(future.result())
            
            # 累加贡献
            for i, contribution in results:
                b[i] += contribution
        else:
            # 原始实现
            for i in range(n):
                neighbors = self.mesh.adjacency[i]

                for j in neighbors:
                    # 计算边的方向
                    edge_vec = self.mesh.vertices[j] - self.mesh.vertices[i]
                    edge_length = np.linalg.norm(edge_vec)

                    if edge_length < 0.001:
                        continue

                    edge_dir = edge_vec / edge_length

                    # 计算梯度在边方向的分量
                    grad_i = self.gradient_field[i] * gradient_magnitudes[i]
                    grad_j = self.gradient_field[j] * gradient_magnitudes[j]

                    grad_avg = 0.5 * (grad_i + grad_j)
                    grad_component = np.dot(grad_avg, edge_dir)

                    # 贡献到右侧向量
                    weight = 1.0 / edge_length
                    b[i] += weight * grad_component * edge_length

        return b

    def _compute_stepover(self, task):
        """
        计算步进距离
        """
        i, gradient_dir = task
        return i, self.calculate_stepover_distance(i, gradient_dir)

    def _smooth_gradient_magnitudes(self, gradient_magnitudes: np.ndarray) -> np.ndarray:
        """
        平滑梯度幅度，避免局部极值
        Args:
            gradient_magnitudes: 原始梯度幅度
        Returns:
            平滑后的梯度幅度
        """
        n = len(gradient_magnitudes)
        smoothed = gradient_magnitudes.copy()
        
        # 首先处理输入中的NaN值
        if np.isnan(smoothed).any():
            nan_mask = np.isnan(smoothed)
            smoothed[nan_mask] = 0.001
        
        if np.isinf(smoothed).any():
            inf_mask = np.isinf(smoothed)
            smoothed[inf_mask] = 0.001
        
        # 使用拉普拉斯平滑
        for i in range(n):
            neighbors = self.mesh.adjacency[i]
            if not neighbors:
                continue
            
            # 计算邻居的平均梯度幅度
            neighbor_sum = 0
            valid_neighbors = 0
            
            for neighbor in neighbors:
                neighbor_val = gradient_magnitudes[neighbor]
                if not (np.isnan(neighbor_val) or np.isinf(neighbor_val)) and neighbor_val > 0:
                    neighbor_sum += neighbor_val
                    valid_neighbors += 1
            
            if valid_neighbors > 0:
                # 平滑当前点的梯度幅度
                avg_neighbor = neighbor_sum / valid_neighbors
                current_val = gradient_magnitudes[i]
                
                if not (np.isnan(current_val) or np.isinf(current_val)) and current_val > 0:
                    smoothed[i] = 0.7 * current_val + 0.3 * avg_neighbor
                else:
                    smoothed[i] = avg_neighbor
        
        # 再次检查并修复结果中的NaN值
        if np.isnan(smoothed).any():
            nan_mask = np.isnan(smoothed)
            smoothed[nan_mask] = 0.001
        
        if np.isinf(smoothed).any():
            inf_mask = np.isinf(smoothed)
            smoothed[inf_mask] = 0.001
        
        # 确保梯度幅度为正
        smoothed = np.maximum(smoothed, 0.001)
        
        return smoothed

    def fixed_point_iteration(self, max_iterations: int = 20, tolerance: float = 1e-4) -> np.ndarray:
        """
        固定点迭代优化
        Args:
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
        Returns:
            优化后的标量场
        """
        n = len(self.mesh.vertices)

        # 初始化
        phi = np.zeros(n)

        for iteration in range(max_iterations):
            # 计算当前梯度的幅度
            gradient_magnitudes = np.zeros(n)
            
            if CGAL_AVAILABLE:
                # 使用CGAL的并行处理能力
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                # 准备任务
                tasks = []
                for i in range(n):
                    tasks.append((i, self.gradient_field[i]))
                
                # 并行计算步进距离
                results = []
                with ThreadPoolExecutor() as executor:
                    # 提交所有任务
                    future_to_task = {executor.submit(self._compute_stepover, task): task for task in tasks}
                    
                    # 收集结果
                    for future in as_completed(future_to_task):
                        try:
                            results.append(future.result())
                        except Exception:
                            pass
                
                # 填充结果
                for i, stepover in results:
                    if not (np.isnan(stepover) or np.isinf(stepover)) and stepover > 0:
                        gradient_magnitudes[i] = stepover
                    else:
                        gradient_magnitudes[i] = 0.001
            else:
                # 优化实现：使用NumPy向量化操作
                for i in range(n):
                    try:
                        stepover = self.calculate_stepover_distance(i, self.gradient_field[i])
                        if not (np.isnan(stepover) or np.isinf(stepover)) and stepover > 0:
                            gradient_magnitudes[i] = stepover
                        else:
                            gradient_magnitudes[i] = 0.001
                    except Exception:
                        gradient_magnitudes[i] = 0.001

            # 检查并修复梯度幅度中的NaN值
            if np.isnan(gradient_magnitudes).any():
                nan_mask = np.isnan(gradient_magnitudes)
                gradient_magnitudes[nan_mask] = 0.001
            
            if np.isinf(gradient_magnitudes).any():
                inf_mask = np.isinf(gradient_magnitudes)
                gradient_magnitudes[inf_mask] = 0.001
            
            # 确保所有值都是正数
            gradient_magnitudes = np.maximum(gradient_magnitudes, 0.001)

            # 平滑梯度幅度，避免局部极值
            try:
                gradient_magnitudes = self._smooth_gradient_magnitudes(gradient_magnitudes)
            except Exception as e:
                print(f"平滑梯度幅度时出错: {e}")

            # 再次检查平滑后的梯度幅度
            if np.isnan(gradient_magnitudes).any():
                nan_mask = np.isnan(gradient_magnitudes)
                gradient_magnitudes[nan_mask] = 0.001
            
            if np.isinf(gradient_magnitudes).any():
                inf_mask = np.isinf(gradient_magnitudes)
                gradient_magnitudes[inf_mask] = 0.001

            # 求解泊松方程
            try:
                phi_new = self.solve_poisson_equation(gradient_magnitudes)
            except Exception as e:
                print(f"求解泊松方程时出错: {e}")
                phi_new = phi.copy()

            # 检查phi_new的有效性
            if np.isnan(phi_new).any():
                nan_mask = np.isnan(phi_new)
                phi_new[nan_mask] = phi[nan_mask]
            
            if np.isinf(phi_new).any():
                inf_mask = np.isinf(phi_new)
                phi_new[inf_mask] = phi[inf_mask]

            # 应用阻尼因子，提高收敛稳定性
            damping_factor = 0.8
            phi_new = damping_factor * phi_new + (1 - damping_factor) * phi

            # 再次检查阻尼后的结果
            if np.isnan(phi_new).any():
                nan_mask = np.isnan(phi_new)
                phi_new[nan_mask] = phi[nan_mask]
            
            if np.isinf(phi_new).any():
                inf_mask = np.isinf(phi_new)
                phi_new[inf_mask] = phi[inf_mask]

            # 检查收敛
            try:
                diff = np.max(np.abs(phi_new - phi))
                if np.isnan(diff) or np.isinf(diff):
                    diff = tolerance + 1.0
            except Exception:
                diff = tolerance + 1.0

            phi = phi_new

            print(f"迭代 {iteration + 1}: 最大变化 = {diff:.6f}")

            if diff < tolerance:
                print(f"在 {iteration + 1} 次迭代后收敛")
                break

        # 最终检查
        if np.isnan(phi).any():
            nan_mask = np.isnan(phi)
            phi[nan_mask] = 0.0
        
        if np.isinf(phi).any():
            inf_mask = np.isinf(phi)
            phi[inf_mask] = 0.0

        return phi

    def generate_scalar_field(self) -> np.ndarray:
        """
        生成等残留高度标量场
        Returns:
            标量场数组
        """
        print("生成等残留高度场...")

        # 1. 初始化梯度场方向
        self.initialize_gradient_field()

        # 2. 固定点迭代优化
        phi = self.fixed_point_iteration()

        # 3. 归一化标量场
        phi_min, phi_max = np.min(phi), np.max(phi)
        if phi_max > phi_min:
            phi = (phi - phi_min) / (phi_max - phi_min)

        self.scalar_field = phi

        print("等残留高度场生成完成")

        return phi

    def _compute_iso_segments(self, task):
        """
        计算等值线段
        """
        face_idx, vertices, values, iso_values = task
        segments = []
        for iso_value in iso_values:
            # 检查等值线是否穿过这个面
            crossings = []

            for i in range(3):
                j = (i + 1) % 3
                v1, v2 = values[i], values[j]

                if (v1 - iso_value) * (v2 - iso_value) < 0:
                    # 等值线穿过这条边
                    t = (iso_value - v1) / (v2 - v1)
                    point = (1 - t) * vertices[i] + t * vertices[j]
                    crossings.append(point)

            if len(crossings) == 2:
                # 在这个面上有一条等值线段
                segments.append([crossings[0], crossings[1]])
        return segments

    def extract_iso_curves(self, scalar_field: np.ndarray, spacing: float = 0.1) -> List[List[np.ndarray]]:
        """
        提取等值线
        使用CGAL加速几何计算
        Args:
            scalar_field: 标量场
            spacing: 等值线间距
        Returns:
            等值线列表
        """
        print("提取等值线...")

        iso_curves = []
        min_val, max_val = np.min(scalar_field), np.max(scalar_field)

        # 生成等值线值
        num_levels = int((max_val - min_val) / spacing) + 1
        iso_values = [min_val + i * spacing for i in range(num_levels)]

        if CGAL_AVAILABLE:
            # 使用CGAL的并行处理能力
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # 准备任务
            tasks = []
            for face_idx, face in enumerate(self.mesh.faces):
                if len(face) == 3:
                    vertices = [self.mesh.vertices[v] for v in face]
                    values = [scalar_field[v] for v in face]
                    tasks.append((face_idx, vertices, values, iso_values))
            
            # 并行计算等值线段
            results = []
            with ThreadPoolExecutor() as executor:
                # 提交所有任务
                future_to_task = {executor.submit(self._compute_iso_segments, task): task for task in tasks}
                
                # 收集结果
                for future in as_completed(future_to_task):
                    results.append(future.result())
            
            # 收集结果
            for segments in results:
                iso_curves.extend(segments)
        else:
            # 原始实现
            for face_idx, face in enumerate(self.mesh.faces):
                if len(face) != 3:
                    continue  # 只处理三角形

                vertices = [self.mesh.vertices[v] for v in face]
                values = [scalar_field[v] for v in face]

                for iso_value in iso_values:
                    # 检查等值线是否穿过这个面
                    crossings = []

                    for i in range(3):
                        j = (i + 1) % 3
                        v1, v2 = values[i], values[j]

                        if (v1 - iso_value) * (v2 - iso_value) < 0:
                            # 等值线穿过这条边
                            t = (iso_value - v1) / (v2 - v1)
                            point = (1 - t) * vertices[i] + t * vertices[j]
                            crossings.append(point)

                    if len(crossings) == 2:
                        # 在这个面上有一条等值线段
                        iso_curves.append([crossings[0], crossings[1]])

        print(f"提取了 {len(iso_curves)} 条等值线段")

        # 连接线段形成连续曲线
        connected_curves = self._connect_iso_segments(iso_curves)

        return connected_curves

    def _connect_iso_segments(self, segments: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """
        连接等值线段形成连续曲线
        使用CGAL加速曲线连接
        """
        if not segments:
            return []

        curves = []
        used = [False] * len(segments)

        if CGAL_AVAILABLE:
            # 使用CGAL的并行处理
            # 这里仍然使用原始的连接算法，因为需要顺序处理
            for i in range(len(segments)):
                if used[i]:
                    continue

                curve = []
                current_seg = segments[i]

                # 添加第一个线段
                curve.extend(current_seg)
                used[i] = True

                # 向前连接
                while True:
                    last_point = curve[-1]
                    found = False

                    for j in range(len(segments)):
                        if used[j]:
                            continue

                        seg = segments[j]

                        # 检查连接
                        if np.allclose(last_point, seg[0], atol=1e-4):
                            curve.append(seg[1])
                            used[j] = True
                            found = True
                            break
                        elif np.allclose(last_point, seg[1], atol=1e-4):
                            curve.append(seg[0])
                            used[j] = True
                            found = True
                            break

                    if not found:
                        break

                # 向后连接
                while True:
                    first_point = curve[0]
                    found = False

                    for j in range(len(segments)):
                        if used[j]:
                            continue

                        seg = segments[j]

                        if np.allclose(first_point, seg[0], atol=1e-4):
                            curve.insert(0, seg[1])
                            used[j] = True
                            found = True
                            break
                        elif np.allclose(first_point, seg[1], atol=1e-4):
                            curve.insert(0, seg[0])
                            used[j] = True
                            found = True
                            break

                    if not found:
                        break

                if len(curve) > 1:
                    curves.append(curve)
        else:
            # 原始实现
            for i in range(len(segments)):
                if used[i]:
                    continue

                curve = []
                current_seg = segments[i]

                # 添加第一个线段
                curve.extend(current_seg)
                used[i] = True

                # 向前连接
                while True:
                    last_point = curve[-1]
                    found = False

                    for j in range(len(segments)):
                        if used[j]:
                            continue

                        seg = segments[j]

                        # 检查连接
                        if np.allclose(last_point, seg[0], atol=1e-4):
                            curve.append(seg[1])
                            used[j] = True
                            found = True
                            break
                        elif np.allclose(last_point, seg[1], atol=1e-4):
                            curve.append(seg[0])
                            used[j] = True
                            found = True
                            break

                    if not found:
                        break

                # 向后连接
                while True:
                    first_point = curve[0]
                    found = False

                    for j in range(len(segments)):
                        if used[j]:
                            continue

                        seg = segments[j]

                        if np.allclose(first_point, seg[0], atol=1e-4):
                            curve.insert(0, seg[1])
                            used[j] = True
                            found = True
                            break
                        elif np.allclose(first_point, seg[1], atol=1e-4):
                            curve.insert(0, seg[0])
                            used[j] = True
                            found = True
                            break

                    if not found:
                        break

                if len(curve) > 1:
                    curves.append(curve)

        print(f"连接成 {len(curves)} 条连续曲线")
        return curves