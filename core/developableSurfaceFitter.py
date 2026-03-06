"""
直纹面拟合模块

本模块实现基于生长策略的封闭直纹面拟合算法，用于五轴加工路径规划。
直纹面是一种由一条直线（母线）沿着两条曲线（导线）移动而生成的曲面。
在五轴加工中，直纹面拟合可以帮助生成更高效的刀具路径，特别是对于具有直纹面特征的零件，如叶轮、叶片等。
"""

import numpy as np
import open3d as o3d
from typing import List, Tuple, Dict, Any, Set
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
        
        # 存储分区信息
        self.partitions = {}
        self.shared_edges = {}
        self.vertex_map = {}
        self.edge_midpoints = np.array([])
        
        # 阈值参数
        self.epsilon = 0.001  # 顶点合并阈值
        self.T = 100  # 直线边判定阈值
        self.max_skip_count = 5  # 最大跳过次数
    
    def fit_developable_surfaces(self, partition_labels: np.ndarray, edge_midpoints: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        拟合所有分区为直纹面
        Args:
            partition_labels: 分区标签数组
            edge_midpoints: 边缘中点数组
        Returns:
            直纹面字典，键为分区标签，值为直纹面参数
        """
        print("拟合直纹面...")
        
        # 1. 输入与预处理
        self._preprocess(partition_labels, edge_midpoints)
        
        # 2. 种子分区判定
        seed_partitions = self._identify_seed_partitions()
        
        # 3. 生长循环
        developable_surfaces = self._growth_loop(seed_partitions)
        
        # 4. 封闭性协调
        self._coordinate_shared_edges(developable_surfaces)
        
        # 5. 可视化直纹面拼接后的原曲面
        self.visualize_developable_assembly(developable_surfaces)
        
        return developable_surfaces
    
    def _preprocess(self, partition_labels: np.ndarray, edge_midpoints: np.ndarray):
        """
        预处理数据
        Args:
            partition_labels: 分区标签数组
            edge_midpoints: 边缘中点数组
        """
        print("预处理数据...")
        
        # 提取每个分区的边界边
        unique_labels = np.unique(partition_labels)
        
        # 构建边缘中点映射
        edge_midpoint_map = {}
        if edge_midpoints.size > 0:
            # 存储边缘中点
            self.edge_midpoints = edge_midpoints
        else:
            self.edge_midpoints = np.array([])
        
        for label in unique_labels:
            # 获取该分区的所有顶点
            partition_vertices = np.where(partition_labels == label)[0]
            
            # 提取边界边
            boundary_edges = self._extract_boundary_edges(partition_vertices, partition_labels)
            
            # 提取内部点云
            interior_points = self._extract_interior_points(partition_vertices, boundary_edges)
            
            # 存储分区信息
            self.partitions[label] = {
                'vertices': partition_vertices,
                'boundary_edges': boundary_edges,
                'interior_points': interior_points,
                'known_edges': set(),
                'skip_count': 0
            }
        
        # 处理共享边
        self._process_shared_edges()
        
        # 顶点合并
        self._merge_vertices()
        
        # 边类型检测
        self._detect_edge_types()
    
    def _extract_boundary_edges(self, partition_vertices: np.ndarray, partition_labels: np.ndarray) -> List[List[int]]:
        """
        提取分区的边界边
        Args:
            partition_vertices: 分区的顶点索引数组
            partition_labels: 分区标签数组
        Returns:
            边界边列表，每条边由顶点索引组成
        """
        boundary_edges = []
        vertex_set = set(partition_vertices)
        
        # 找出所有边界边（只属于一个分区的边）
        for v in partition_vertices:
            for neighbor in self.adjacency[v]:
                if partition_labels[neighbor] != partition_labels[v]:
                    # 按顺序存储边，确保一致性
                    edge = tuple(sorted([v, neighbor]))
                    if edge not in boundary_edges:
                        boundary_edges.append(list(edge))
        
        return boundary_edges
    
    def _extract_interior_points(self, partition_vertices: np.ndarray, boundary_edges: List[List[int]]) -> List[np.ndarray]:
        """
        提取分区的内部点云
        Args:
            partition_vertices: 分区的顶点索引数组
            boundary_edges: 边界边列表
        Returns:
            内部点云列表
        """
        # 收集边界顶点
        boundary_vertices = set()
        for edge in boundary_edges:
            boundary_vertices.update(edge)
        
        # 提取内部顶点
        interior_vertices = [v for v in partition_vertices if v not in boundary_vertices]
        
        # 转换为点云
        interior_points = [self.vertices[v] for v in interior_vertices]
        
        return interior_points
    
    def _process_shared_edges(self):
        """
        处理共享边
        """
        print("处理共享边...")
        
        # 收集所有边界边
        all_edges = {}
        
        for label, partition in self.partitions.items():
            for edge in partition['boundary_edges']:
                edge_key = tuple(edge)
                if edge_key not in all_edges:
                    all_edges[edge_key] = []
                all_edges[edge_key].append(label)
        
        # 识别共享边
        for edge_key, labels in all_edges.items():
            if len(labels) > 1:
                self.shared_edges[edge_key] = labels
    
    def _merge_vertices(self):
        """
        合并顶点
        """
        print("合并顶点...")
        
        # 收集所有顶点
        all_vertices = []
        for label, partition in self.partitions.items():
            all_vertices.extend(partition['vertices'])
        
        # 计算两两顶点之间的距离，合并距离小于阈值的顶点
        merged = {}
        for i, v1 in enumerate(all_vertices):
            if v1 not in merged:
                merged[v1] = v1
                for v2 in all_vertices[i+1:]:
                    if v2 not in merged:
                        distance = np.linalg.norm(self.vertices[v1] - self.vertices[v2])
                        if distance < self.epsilon:
                            merged[v2] = v1
        
        # 更新顶点映射
        self.vertex_map = merged
        
        # 更新分区的边界边
        for label, partition in self.partitions.items():
            # 更新边界边
            updated_edges = []
            for edge in partition['boundary_edges']:
                updated_edge = [self.vertex_map.get(v, v) for v in edge]
                updated_edges.append(updated_edge)
            partition['boundary_edges'] = updated_edges
    
    def _detect_edge_types(self):
        """
        检测边类型
        """
        print("检测边类型...")
        
        for label, partition in self.partitions.items():
            edge_types = {}
            for edge in partition['boundary_edges']:
                # 提取边上的点
                edge_points = [self.vertices[v] for v in edge]
                
                # 主成分分析
                if len(edge_points) > 1:
                    mean = np.mean(edge_points, axis=0)
                    centered = edge_points - mean
                    cov = np.cov(centered.T)
                    eigenvalues, _ = np.linalg.eigh(cov)
                    eigenvalues = sorted(eigenvalues, reverse=True)
                    
                    # 判定边类型
                    if eigenvalues[0] / (eigenvalues[1] + 1e-8) > self.T:
                        edge_types[tuple(edge)] = 'straight'
                    else:
                        edge_types[tuple(edge)] = 'curved'
                else:
                    edge_types[tuple(edge)] = 'straight'
            
            partition['edge_types'] = edge_types
    
    def _identify_seed_partitions(self) -> List[int]:
        """
        识别种子分区
        Returns:
            种子分区标签列表
        """
        print("识别种子分区...")
        
        seed_partitions = []
        
        # 首先尝试识别三角形和四边形分区
        for label, partition in self.partitions.items():
            boundary_edges = partition['boundary_edges']
            
            # 三角形分区
            if len(boundary_edges) == 3:
                # 检查是否构成三角形
                vertices = set()
                for edge in boundary_edges:
                    vertices.update(edge)
                if len(vertices) == 3:
                    seed_partitions.append(label)
                    # 标记边为已知
                    for edge in boundary_edges:
                        partition['known_edges'].add(tuple(edge))
            
            # 四边形分区
            elif len(boundary_edges) == 4:
                # 检查是否构成四边形
                vertices = set()
                for edge in boundary_edges:
                    vertices.update(edge)
                if len(vertices) == 4:
                    seed_partitions.append(label)
                    # 标记边为已知
                    for edge in boundary_edges:
                        partition['known_edges'].add(tuple(edge))
        
        # 如果没有识别到种子分区，选择一些分区作为种子
        if len(seed_partitions) == 0:
            print("没有识别到三角形或四边形分区，选择其他分区作为种子")
            # 按分区大小排序，选择中等大小的分区
            sorted_partitions = sorted(self.partitions.items(), key=lambda x: len(x[1]['vertices']))
            # 选择中间的几个分区作为种子
            mid_index = len(sorted_partitions) // 2
            for i in range(max(0, mid_index - 1), min(len(sorted_partitions), mid_index + 2)):
                label, partition = sorted_partitions[i]
                seed_partitions.append(label)
                # 标记边为已知
                for edge in partition['boundary_edges']:
                    partition['known_edges'].add(tuple(edge))
        
        print(f"识别到 {len(seed_partitions)} 个种子分区")
        return seed_partitions
    
    def _growth_loop(self, seed_partitions: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        生长循环
        Args:
            seed_partitions: 种子分区标签列表
        Returns:
            直纹面字典
        """
        print("生长循环...")
        
        developable_surfaces = {}
        queue = seed_partitions.copy()
        processed = set()
        
        # 首先处理队列中的分区
        while queue:
            # 取出一个分区
            label = queue.pop(0)
            if label in processed:
                continue
            processed.add(label)
            
            partition = self.partitions[label]
            
            # 统计已知边
            known_edges = partition['known_edges']
            boundary_edges = partition['boundary_edges']
            
            # 根据分区形状和已知边信息选择拟合策略
            if len(boundary_edges) == 3:
                # 三角形分区
                surface = self._fit_triangular_partition(label, partition)
            elif len(boundary_edges) == 4:
                # 四边形分区
                surface = self._fit_quadrilateral_partition(label, partition)
            else:
                # 其他情况
                surface = self._fit_general_partition(label, partition)
            
            if surface:
                developable_surfaces[label] = surface
                
                # 将新确定的边标记为已知，并更新相邻分区
                for edge in boundary_edges:
                    edge_key = tuple(edge)
                    if edge_key not in known_edges:
                        partition['known_edges'].add(edge_key)
                        
                        # 找到相邻分区
                        if edge_key in self.shared_edges:
                            for neighbor_label in self.shared_edges[edge_key]:
                                if neighbor_label != label and neighbor_label not in processed:
                                    neighbor_partition = self.partitions[neighbor_label]
                                    neighbor_partition['known_edges'].add(edge_key)
                                    if neighbor_label not in queue:
                                        queue.append(neighbor_label)
            else:
                # 增加跳过计数
                partition['skip_count'] += 1
                if partition['skip_count'] < self.max_skip_count:
                    queue.append(label)
                else:
                    # 强制拟合
                    surface = self._force_fit_partition(label, partition)
                    if surface:
                        developable_surfaces[label] = surface
        
        # 处理未处理的分区
        print(f"处理了 {len(processed)} 个分区，开始处理剩余分区...")
        for label, partition in self.partitions.items():
            if label not in processed:
                # 强制拟合
                surface = self._force_fit_partition(label, partition)
                if surface:
                    developable_surfaces[label] = surface
        
        print(f"生长循环完成，拟合了 {len(developable_surfaces)} 个直纹面")
        return developable_surfaces
    
    def _fit_triangular_partition(self, label: int, partition: Dict[str, Any]) -> Dict[str, Any]:
        """
        拟合三角形分区
        Args:
            label: 分区标签
            partition: 分区信息
        Returns:
            直纹面参数
        """
        # 实现三角形分区的拟合
        boundary_edges = partition['boundary_edges']
        known_edges = partition['known_edges']
        interior_points = partition['interior_points']
        
        # 检查已知边数量
        if len(known_edges) < 2:
            return None
        
        # 找到公共顶点
        edge_vertices = []
        for edge in known_edges:
            edge_vertices.extend(edge)
        
        # 统计顶点出现次数
        vertex_count = {}
        for v in edge_vertices:
            vertex_count[v] = vertex_count.get(v, 0) + 1
        
        # 公共顶点是出现次数为2的顶点
        common_vertex = None
        for v, count in vertex_count.items():
            if count == 2:
                common_vertex = v
                break
        
        if not common_vertex:
            return None
        
        # 找到另外两个顶点
        other_vertices = []
        for edge in known_edges:
            for v in edge:
                if v != common_vertex:
                    other_vertices.append(v)
        
        if len(other_vertices) != 2:
            return None
        
        A, B = other_vertices
        
        # 拟合第三条边
        third_edge = None
        for edge in boundary_edges:
            edge_key = tuple(edge)
            if edge_key not in known_edges:
                third_edge = edge
                break
        
        if third_edge:
            # 拟合曲线
            curve = self._fit_curve([self.vertices[v] for v in third_edge])
        else:
            # 直接连接A和B
            curve = {
                'type': 'line',
                'start_point': self.vertices[A].tolist(),
                'end_point': self.vertices[B].tolist()
            }
        
        # 生成直纹面
        surface = {
            'type': 'conical',
            'vertex': self.vertices[common_vertex].tolist(),
            'curve': curve,
            'vertices': partition['vertices'],
            'label': label
        }
        
        return surface
    
    def _fit_quadrilateral_partition(self, label: int, partition: Dict[str, Any]) -> Dict[str, Any]:
        """
        拟合四边形分区
        Args:
            label: 分区标签
            partition: 分区信息
        Returns:
            直纹面参数
        """
        # 实现四边形分区的拟合
        boundary_edges = partition['boundary_edges']
        known_edges = partition['known_edges']
        interior_points = partition['interior_points']
        
        # 检查已知边数量
        if len(known_edges) < 2:
            return None
        
        # 找到两条相对的直线边
        straight_edges = []
        for edge in known_edges:
            if partition['edge_types'].get(edge) == 'straight':
                straight_edges.append(edge)
        
        if len(straight_edges) < 2:
            return None
        
        # 假设前两条为相对边
        L0, L1 = straight_edges[:2]
        P00, P01 = L0
        P10, P11 = L1
        
        # 拟合两条曲线边
        curve0 = self._fit_curve([self.vertices[P00], self.vertices[P10]])
        curve1 = self._fit_curve([self.vertices[P01], self.vertices[P11]])
        
        # 生成直纹面
        surface = {
            'type': 'developable',
            'curve0': curve0,
            'curve1': curve1,
            'vertices': partition['vertices'],
            'label': label
        }
        
        return surface
    
    def _fit_general_partition(self, label: int, partition: Dict[str, Any]) -> Dict[str, Any]:
        """
        拟合一般分区
        Args:
            label: 分区标签
            partition: 分区信息
        Returns:
            直纹面参数
        """
        # 实现一般分区的拟合
        boundary_edges = partition['boundary_edges']
        known_edges = partition['known_edges']
        
        # 检查已知边数量
        if len(known_edges) < 2:
            return None
        
        # 简单实现：使用两条已知边作为母线
        known_edge_list = list(known_edges)
        edge1, edge2 = known_edge_list[:2]
        
        # 拟合曲线
        curve1 = self._fit_curve([self.vertices[v] for v in edge1])
        curve2 = self._fit_curve([self.vertices[v] for v in edge2])
        
        # 生成直纹面
        surface = {
            'type': 'developable',
            'curve1': curve1,
            'curve2': curve2,
            'vertices': partition['vertices'],
            'label': label
        }
        
        return surface
    
    def _force_fit_partition(self, label: int, partition: Dict[str, Any]) -> Dict[str, Any]:
        """
        强制拟合分区
        Args:
            label: 分区标签
            partition: 分区信息
        Returns:
            直纹面参数
        """
        # 实现强制拟合
        boundary_edges = partition['boundary_edges']
        
        if len(boundary_edges) < 2:
            return None
        
        # 使用前两条边作为母线
        edge1, edge2 = boundary_edges[:2]
        
        # 拟合曲线
        curve1 = self._fit_curve([self.vertices[v] for v in edge1])
        curve2 = self._fit_curve([self.vertices[v] for v in edge2])
        
        # 生成直纹面
        surface = {
            'type': 'developable',
            'curve1': curve1,
            'curve2': curve2,
            'vertices': partition['vertices'],
            'label': label
        }
        
        return surface
    
    def _fit_curve(self, points: List[np.ndarray]) -> Dict[str, Any]:
        """
        拟合曲线
        Args:
            points: 曲线上的点
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
    
    def _coordinate_shared_edges(self, developable_surfaces: Dict[int, Dict[str, Any]]):
        """
        协调共享边
        Args:
            developable_surfaces: 直纹面字典
        """
        print("协调共享边...")
        
        # 收集所有共享边的拟合结果
        edge_curves = {}
        
        for label, surface in developable_surfaces.items():
            partition = self.partitions[label]
            for edge in partition['boundary_edges']:
                edge_key = tuple(edge)
                if edge_key in self.shared_edges:
                    # 提取边的曲线
                    if surface['type'] == 'conical':
                        curve = surface['curve']
                    else:
                        # 简单实现：使用第一条曲线
                        curve = surface.get('curve1', surface.get('curve0'))
                    
                    if edge_key not in edge_curves:
                        edge_curves[edge_key] = []
                    edge_curves[edge_key].append(curve)
        
        # 计算平均曲线
        for edge_key, curves in edge_curves.items():
            if len(curves) > 1:
                # 简单实现：对控制点取平均
                avg_curve = self._average_curves(curves)
                
                # 更新相关分区的直纹面
                for label in self.shared_edges[edge_key]:
                    if label in developable_surfaces:
                        surface = developable_surfaces[label]
                        if surface['type'] == 'conical':
                            surface['curve'] = avg_curve
                        else:
                            # 简单实现：更新第一条曲线
                            if 'curve1' in surface:
                                surface['curve1'] = avg_curve
                            if 'curve0' in surface:
                                surface['curve0'] = avg_curve
    
    def _average_curves(self, curves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算平均曲线
        Args:
            curves: 曲线列表
        Returns:
            平均曲线
        """
        # 简单实现：返回第一条曲线
        return curves[0]
    
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
        if surface['type'] == 'conical':
            # 锥面
            vertex = np.array(surface['vertex'])
            curve_point = self._evaluate_curve(surface['curve'], t)
            return vertex + s * (curve_point - vertex)
        else:
            # 直纹面
            if 'curve1' in surface and 'curve2' in surface:
                p1 = self._evaluate_curve(surface['curve1'], t)
                p2 = self._evaluate_curve(surface['curve2'], t)
            elif 'curve0' in surface and 'curve1' in surface:
                p1 = self._evaluate_curve(surface['curve0'], t)
                p2 = self._evaluate_curve(surface['curve1'], t)
            else:
                # 默认返回原点
                return np.array([0.0, 0.0, 0.0])
            
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
