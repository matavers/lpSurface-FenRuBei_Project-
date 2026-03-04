"""
直纹面拟合模块

本模块实现直纹面的检测和拟合算法，用于五轴加工路径规划。
直纹面是一种由一条直线（母线）沿着两条曲线（导线）移动而生成的曲面。
在五轴加工中，直纹面拟合可以帮助生成更高效的刀具路径，特别是对于具有直纹面特征的零件，如叶轮、叶片等。
"""

import numpy as np
import open3d as o3d
from typing import List, Tuple, Dict, Any
from .meshProcessor import MeshProcessor


class DevelopableSurfaceFitter:
    """
    直纹面拟合器
    """
    
    def __init__(self, mesh: MeshProcessor):
        """
        初始化直纹面拟合器
        Args:
            mesh: 网格处理器
        """
        self.mesh = mesh
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.adjacency = mesh.adjacency
    
    def detect_developable_regions(self, partition_labels: np.ndarray) -> Dict[int, bool]:
        """
        检测每个分区是否为直纹面区域
        Args:
            partition_labels: 分区标签数组
        Returns:
            分区ID到是否为直纹面的映射
        """
        print("检测直纹面区域...")
        
        developable_regions = {}
        unique_labels = np.unique(partition_labels)
        
        for label in unique_labels:
            # 获取该分区的所有顶点
            partition_vertices = np.where(partition_labels == label)[0]
            if len(partition_vertices) < 3:
                developable_regions[label] = False
                continue
            
            # 计算该分区的直纹面可能性
            is_developable = self._is_partition_developable(partition_vertices)
            developable_regions[label] = is_developable
            
            if is_developable:
                print(f"分区 {label} 被检测为直纹面区域")
        
        return developable_regions
    
    def _is_partition_developable(self, partition_vertices: np.ndarray) -> bool:
        """
        判断一个分区是否为直纹面区域
        Args:
            partition_vertices: 分区的顶点索引数组
        Returns:
            是否为直纹面区域
        """
        # 计算分区的平均曲率
        curvatures = self.mesh.curvatures[partition_vertices]
        avg_curvature = np.mean(curvatures)
        
        # 直纹面的高斯曲率为0，平均曲率通常较小
        if avg_curvature > 0.5:  # 降低阈值，使圆柱面等直纹面能通过
            return False
        
        # 检查分区的几何形状
        # 计算分区的边界曲线
        boundary_edges = self._extract_boundary_edges(partition_vertices)
        if len(boundary_edges) != 2:
            # 直纹面通常有两条边界曲线
            return False
        
        # 检查两条边界曲线之间是否可以用直线连接
        return self._check_straight_line_connection(boundary_edges)
    
    def _extract_boundary_edges(self, partition_vertices: np.ndarray) -> List[List[int]]:
        """
        提取分区的边界曲线
        Args:
            partition_vertices: 分区的顶点索引数组
        Returns:
            边界曲线列表，每条曲线由顶点索引组成
        """
        # 构建分区的边界边
        boundary_edges = []
        vertex_set = set(partition_vertices)
        
        # 找出所有边界边（只属于一个分区的边）
        for v in partition_vertices:
            for neighbor in self.adjacency[v]:
                if neighbor not in vertex_set:
                    boundary_edges.append((v, neighbor))
        
        # 尝试提取边界曲线
        curves = []
        if boundary_edges:
            # 将边界边连接成边界曲线
            curves = self._connect_edges(boundary_edges)
        
        # 如果没有边界边或提取的曲线数量不等于2，创建两条边界曲线
        if not curves or len(curves) != 2:
            # 对于没有边界边的情况或曲线数量不符合要求的情况
            # 使用顶点的坐标来创建两条曲线
            vertices = self.vertices[partition_vertices]
            
            # 计算分区的中心点
            center = np.mean(vertices, axis=0)
            
            # 计算每个顶点到中心的距离
            distances = np.linalg.norm(vertices - center, axis=1)
            
            # 找到距离最小和最大的顶点
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            min_dist_vertices = partition_vertices[np.isclose(distances, min_dist)]
            max_dist_vertices = partition_vertices[np.isclose(distances, max_dist)]
            
            # 对顶点进行排序，形成曲线
            def sort_vertices_by_angle(vertices):
                """按角度排序顶点"""
                points = self.vertices[vertices]
                angles = []
                for point in points:
                    vec = point - center
                    angle = np.arctan2(vec[1], vec[0])
                    angles.append(angle)
                sorted_indices = np.argsort(angles)
                return vertices[sorted_indices]
            
            if len(min_dist_vertices) > 1:
                min_dist_vertices = sort_vertices_by_angle(min_dist_vertices)
            if len(max_dist_vertices) > 1:
                max_dist_vertices = sort_vertices_by_angle(max_dist_vertices)
            
            # 如果顶点数量不足，使用其他方法创建曲线
            if len(min_dist_vertices) < 2:
                # 按x坐标排序
                x_coords = vertices[:, 0]
                sorted_indices = np.argsort(x_coords)
                sorted_vertices = partition_vertices[sorted_indices]
                mid = len(sorted_vertices) // 2
                min_dist_vertices = sorted_vertices[:mid]
                max_dist_vertices = sorted_vertices[mid:]
            
            return [min_dist_vertices.tolist(), max_dist_vertices.tolist()]
        
        return curves
    
    def _connect_edges(self, edges: List[Tuple[int, int]]) -> List[List[int]]:
        """
        将边界边连接成边界曲线
        Args:
            edges: 边界边列表
        Returns:
            边界曲线列表
        """
        if not edges:
            return []
        
        # 构建边的字典
        edge_dict = {}
        for v1, v2 in edges:
            if v1 not in edge_dict:
                edge_dict[v1] = []
            if v2 not in edge_dict:
                edge_dict[v2] = []
            edge_dict[v1].append(v2)
            edge_dict[v2].append(v1)
        
        # 找到起点（度为1的顶点）
        start_vertices = [v for v, neighbors in edge_dict.items() if len(neighbors) == 1]
        
        curves = []
        visited = set()
        
        for start in start_vertices:
            if start in visited:
                continue
            
            curve = [start]
            current = start
            visited.add(current)
            
            while True:
                # 找到下一个顶点
                neighbors = edge_dict[current]
                next_vertex = None
                for neighbor in neighbors:
                    if neighbor not in visited:
                        next_vertex = neighbor
                        break
                
                if next_vertex is None:
                    break
                
                curve.append(next_vertex)
                visited.add(next_vertex)
                current = next_vertex
            
            if len(curve) > 1:
                curves.append(curve)
        
        return curves
    
    def _check_straight_line_connection(self, boundary_curves: List[List[int]]) -> bool:
        """
        检查两条边界曲线之间是否可以用直线连接
        Args:
            boundary_curves: 边界曲线列表
        Returns:
            是否可以用直线连接
        """
        if len(boundary_curves) != 2:
            return False
        
        curve1, curve2 = boundary_curves
        
        # 确保两条曲线都有足够的点
        if len(curve1) < 2 or len(curve2) < 2:
            return False
        
        # 简化检查，只检查几个关键点
        # 检查起点
        v1_start = curve1[0]
        v2_start = curve2[0]
        if not self._check_line_fit(v1_start, v2_start):
            return False
        
        # 检查中点
        v1_mid = curve1[len(curve1) // 2]
        v2_mid = curve2[len(curve2) // 2]
        if not self._check_line_fit(v1_mid, v2_mid):
            return False
        
        # 检查终点
        v1_end = curve1[-1]
        v2_end = curve2[-1]
        if not self._check_line_fit(v1_end, v2_end):
            return False
        
        return True
    
    def _check_line_fit(self, v1: int, v2: int) -> bool:
        """
        检查两点之间的直线是否与曲面近似
        Args:
            v1: 第一个顶点索引
            v2: 第二个顶点索引
        Returns:
            直线是否与曲面近似
        """
        # 计算直线上的点
        p1 = self.vertices[v1]
        p2 = self.vertices[v2]
        
        # 采样直线上的点
        num_samples = 5
        for t in np.linspace(0, 1, num_samples):
            point = (1 - t) * p1 + t * p2
            
            # 找到最近的曲面点
            distances = np.linalg.norm(self.vertices - point, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_point = self.vertices[nearest_idx]
            
            # 计算距离
            distance = np.linalg.norm(point - nearest_point)
            if distance > 0.5:  # 降低阈值，使圆柱面的母线能通过检查
                return False
        
        return True
    
    def fit_developable_surface(self, partition_vertices: np.ndarray, error_threshold: float = 0.01) -> Dict[str, Any]:
        """
        拟合直纹面
        Args:
            partition_vertices: 分区的顶点索引数组
            error_threshold: 逼近误差阈值，用于确定直纹面类型
        Returns:
            直纹面的参数表示
        """
        print("拟合直纹面...")
        
        # 提取边界曲线
        boundary_curves = self._extract_boundary_edges(partition_vertices)
        if len(boundary_curves) != 2:
            return None
        
        curve1, curve2 = boundary_curves
        
        # 确定直纹面类型（直线端或尖锐端）
        curve1_type = self._determine_curve_type(curve1, error_threshold)
        curve2_type = self._determine_curve_type(curve2, error_threshold)
        
        # 拟合两条边界曲线
        curve1_fit = self._fit_curve([self.vertices[v] for v in curve1], curve1_type)
        curve2_fit = self._fit_curve([self.vertices[v] for v in curve2], curve2_type)
        
        # 生成直纹面的参数表示
        developable_surface = {
            'type': 'developable',
            'curve1': curve1_fit,
            'curve2': curve2_fit,
            'curve1_type': curve1_type,
            'curve2_type': curve2_type,
            'vertices': partition_vertices,
            'error_threshold': error_threshold
        }
        
        return developable_surface
    
    def _determine_curve_type(self, curve: List[int], error_threshold: float) -> str:
        """
        确定曲线类型（直线或尖锐）
        Args:
            curve: 曲线的顶点索引列表
            error_threshold: 逼近误差阈值
        Returns:
            曲线类型：'straight' 或 'sharp'
        """
        # 检查曲线是否为空
        if not curve:
            return 'straight'
        
        # 计算曲线的长度
        curve_points = [self.vertices[v] for v in curve]
        
        # 检查曲线点是否为空
        if len(curve_points) < 2:
            return 'straight'
        
        curve_length = 0
        for i in range(1, len(curve_points)):
            curve_length += np.linalg.norm(curve_points[i] - curve_points[i-1])
        
        # 计算曲线的直线拟合误差
        start_point = curve_points[0]
        end_point = curve_points[-1]
        line_vector = end_point - start_point
        line_length = np.linalg.norm(line_vector)
        
        # 避免除以零
        if line_length < 1e-8:
            return 'straight'
        
        max_error = 0
        for point in curve_points[1:-1]:
            # 计算点到直线的距离
            vector_to_point = point - start_point
            projection = np.dot(vector_to_point, line_vector) / line_length
            projected_point = start_point + (projection / line_length) * line_vector
            error = np.linalg.norm(point - projected_point)
            max_error = max(max_error, error)
        
        # 根据误差阈值确定曲线类型
        if max_error < error_threshold:
            return 'straight'
        else:
            # 检查是否有尖锐点
            if self._has_sharp_points(curve, error_threshold):
                return 'sharp'
            else:
                return 'curved'
    
    def _has_sharp_points(self, curve: List[int], error_threshold: float) -> bool:
        """
        检查曲线是否有尖锐点
        Args:
            curve: 曲线的顶点索引列表
            error_threshold: 逼近误差阈值
        Returns:
            是否有尖锐点
        """
        curve_points = [self.vertices[v] for v in curve]
        
        for i in range(1, len(curve_points) - 1):
            # 计算前后向量
            vec1 = curve_points[i] - curve_points[i-1]
            vec2 = curve_points[i+1] - curve_points[i]
            
            # 归一化
            vec1 /= np.linalg.norm(vec1)
            vec2 /= np.linalg.norm(vec2)
            
            # 计算夹角
            dot_product = np.dot(vec1, vec2)
            angle = np.arccos(max(-1, min(1, dot_product)))
            
            # 如果夹角小于阈值，认为是尖锐点
            if angle < np.pi / 3:  # 60度
                return True
        
        return False
    
    def _fit_curve(self, points: List[np.ndarray], curve_type: str = 'curved') -> Dict[str, Any]:
        """
        拟合曲线
        Args:
            points: 曲线上的点
            curve_type: 曲线类型：'straight', 'sharp', 'curved'
        Returns:
            曲线的参数表示
        """
        points = np.array(points)
        n = len(points)
        
        # 检查点列表是否为空
        if n == 0:
            # 返回默认直线
            return {
                'type': 'line',
                'start_point': [0.0, 0.0, 0.0],
                'end_point': [1.0, 0.0, 0.0]
            }
        
        # 检查点列表是否只有一个点
        if n == 1:
            # 返回从该点到自身的直线
            point = points[0]
            return {
                'type': 'line',
                'start_point': point.tolist(),
                'end_point': point.tolist()
            }
        
        # 参数化
        t = np.linspace(0, 1, n)
        
        if curve_type == 'straight':
            # 直线拟合
            start_point = points[0]
            end_point = points[-1]
            return {
                'type': 'line',
                'start_point': start_point.tolist(),
                'end_point': end_point.tolist()
            }
        else:
            # 使用多项式拟合
            degree = min(3, n - 1)  # 使用3次多项式或更低
            
            # 拟合x, y, z坐标
            coeffs_x = np.polyfit(t, points[:, 0], degree)
            coeffs_y = np.polyfit(t, points[:, 1], degree)
            coeffs_z = np.polyfit(t, points[:, 2], degree)
            
            return {
                'type': 'polynomial',
                'degree': degree,
                'coeffs_x': coeffs_x.tolist(),
                'coeffs_y': coeffs_y.tolist(),
                'coeffs_z': coeffs_z.tolist()
            }
    
    def generate_tool_paths_for_developable(self, surface: Dict[str, Any], tool) -> List[Dict[str, Any]]:
        """
        为直纹面生成刀具路径
        Args:
            surface: 直纹面的参数表示
            tool: 刀具对象
        Returns:
            刀具路径列表
        """
        print("为直纹面生成刀具路径...")
        
        paths = []
        
        # 生成参数化的刀具路径
        num_paths = 20  # 路径数量
        
        for i in range(num_paths):
            t = i / (num_paths - 1)
            
            # 计算当前参数下的母线
            path_points = []
            path_orientations = []
            
            # 采样母线上的点
            num_points = 50
            for s in np.linspace(0, 1, num_points):
                # 计算母线上的点
                point = self._evaluate_developable(surface, t, s)
                
                # 计算刀具方向
                orientation = self._calculate_tool_orientation(surface, t, s)
                
                path_points.append(point)
                path_orientations.append(orientation)
            
            paths.append({
                'points': path_points,
                'orientations': path_orientations,
                'type': 'developable_path'
            })
        
        return paths
    
    def _evaluate_developable(self, surface: Dict[str, Any], t: float, s: float) -> np.ndarray:
        """
        计算直纹面上的点
        Args:
            surface: 直纹面的参数表示
            t: 曲线参数
            s: 母线参数
        Returns:
            直纹面上的点
        """
        # 计算两条曲线上的点
        p1 = self._evaluate_curve(surface['curve1'], t)
        p2 = self._evaluate_curve(surface['curve2'], t)
        
        # 计算母线上的点
        return (1 - s) * p1 + s * p2
    
    def _evaluate_curve(self, curve: Dict[str, Any], t: float) -> np.ndarray:
        """
        计算曲线上的点
        Args:
            curve: 曲线的参数表示
            t: 参数
        Returns:
            曲线上的点
        """
        if curve['type'] == 'polynomial':
            x = np.polyval(curve['coeffs_x'], t)
            y = np.polyval(curve['coeffs_y'], t)
            z = np.polyval(curve['coeffs_z'], t)
            return np.array([x, y, z])
        elif curve['type'] == 'line':
            start_point = np.array(curve['start_point'])
            end_point = np.array(curve['end_point'])
            return (1 - t) * start_point + t * end_point
        else:
            raise ValueError(f"不支持的曲线类型: {curve['type']}")
    
    def _calculate_tool_orientation(self, surface: Dict[str, Any], t: float, s: float) -> np.ndarray:
        """
        计算直纹面上点的刀具方向
        Args:
            surface: 直纹面的参数表示
            t: 曲线参数
            s: 母线参数
        Returns:
            刀具方向
        """
        # 计算母线方向
        p1 = self._evaluate_curve(surface['curve1'], t)
        p2 = self._evaluate_curve(surface['curve2'], t)
        generator_direction = p2 - p1
        
        # 计算曲线切线方向
        dt = 1e-6
        p1_dt = self._evaluate_curve(surface['curve1'], t + dt)
        p1_dt_minus = self._evaluate_curve(surface['curve1'], t - dt)
        tangent = p1_dt - p1_dt_minus
        
        # 计算法向量
        normal = np.cross(generator_direction, tangent)
        normal /= np.linalg.norm(normal)
        
        return normal
    
    def visualize_developable_assembly(self, developable_surfaces: Dict[int, Dict[str, Any]]):
        """
        可视化直纹面拼接后的原曲面
        Args:
            developable_surfaces: 直纹面字典，键为分区标签，值为直纹面参数
        """
        print("可视化直纹面拼接后的原曲面...")
        
        import open3d as o3d
        
        # 创建所有直纹面的网格
        meshes = []
        
        for label, surface in developable_surfaces.items():
            # 生成直纹面网格
            mesh = self._generate_developable_mesh(surface)
            if mesh:
                # 为每个直纹面设置不同的颜色
                color = np.random.rand(3)
                mesh.paint_uniform_color(color)
                meshes.append(mesh)
        
        # 可视化
        if meshes:
            o3d.visualization.draw_geometries(meshes, window_name="直纹面拼接后的原曲面")
        
        print("直纹面拼接可视化完成")
    
    def _generate_developable_mesh(self, surface: Dict[str, Any]) -> o3d.geometry.TriangleMesh:
        """
        生成直纹面的网格
        Args:
            surface: 直纹面的参数表示
        Returns:
            直纹面的网格对象
        """
        import open3d as o3d
        
        # 生成参数化网格
        num_curve_points = 50  # 曲线方向的点数
        num_generator_points = 10  # 母线方向的点数
        
        vertices = []
        triangles = []
        
        # 生成顶点
        for i in range(num_curve_points):
            t = i / (num_curve_points - 1)
            for j in range(num_generator_points):
                s = j / (num_generator_points - 1)
                point = self._evaluate_developable(surface, t, s)
                vertices.append(point)
        
        # 生成三角形
        for i in range(num_curve_points - 1):
            for j in range(num_generator_points - 1):
                # 计算四个顶点的索引
                idx0 = i * num_generator_points + j
                idx1 = i * num_generator_points + (j + 1)
                idx2 = (i + 1) * num_generator_points + j
                idx3 = (i + 1) * num_generator_points + (j + 1)
                
                # 添加两个三角形
                triangles.append([idx0, idx1, idx2])
                triangles.append([idx1, idx3, idx2])
        
        # 创建网格
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        
        return mesh
