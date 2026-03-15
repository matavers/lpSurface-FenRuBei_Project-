"""
直纹面拟合模块（重构版 - 支持神经网络）

本模块实现基于神经网络的直纹面拟合算法，用于五轴加工路径规划。
支持两种模式：
1. 神经网络模式：使用 EdgeToSurfaceNet 直接从边缘点列预测直纹面控制点
2. 传统模式：使用几何方法拟合（fallback）

直纹面数学定义：
S(u,v) = v·c₁(u) + (1-v)·c₂(u), v ∈ [0,1]
其中 c₁(u) 和 c₂(u) 是两条 B 样条曲线（准线）
"""

import numpy as np
import open3d as o3d
from typing import List, Tuple, Dict, Any, Set
import os
from .meshProcessor import MeshProcessor


class DevelopableSurfaceFitter:
    """
    直纹面拟合器（支持神经网络）
    """
    
    def __init__(self, mesh: MeshProcessor, device=None, neural_model_path=None, use_neural=True):
        """
        初始化直纹面拟合器
        Args:
            mesh: 网格处理器
            device: 设备（CPU 或 GPU）
            neural_model_path: 神经网络模型权重路径
            use_neural: 是否使用神经网络拟合（默认 True）
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
        
        # 神经网络配置
        self.use_neural = use_neural
        self.neural_fitter = None
        
        if use_neural:
            self._init_neural_fitter(device, neural_model_path)
    
    def _init_neural_fitter(self, device, model_path):
        """初始化神经网络拟合器"""
        try:
            from .edgePointToNURBSSurfaceNet import EdgePointToNURBSSurfaceWrapper
            
            # 自动选择设备
            if device is None:
                import torch
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 查找模型路径
            if model_path is None:
                # 默认路径
                default_paths = [
                    'data/neural/checkpoints/best_nurbs_model.pth',
                    os.path.join(os.path.dirname(__file__), '..', 'data', 'neural', 'checkpoints', 'best_nurbs_model.pth')
                ]
                for path in default_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
            
            if model_path and os.path.exists(model_path):
                self.neural_fitter = EdgePointToNURBSSurfaceWrapper(device=device, model_path=model_path, M=16, degree=3)
                print(f"NURBS神经网络拟合器已初始化，模型路径：{model_path}")
            else:
                print(f"警告：未找到神经网络模型文件，将使用传统方法")
                self.use_neural = False
        except Exception as e:
            print(f"初始化神经网络拟合器失败：{e}，将使用传统方法")
            self.use_neural = False
    
    def fit_developable_surfaces(self, partition_labels: np.ndarray, edge_midpoints: np.ndarray = None) -> Dict[int, Dict[str, Any]]:
        """
        拟合所有分区为直纹面
        Args:
            partition_labels: 分区标签数组
            edge_midpoints: 边缘中点数组（可选）
        Returns:
            直纹面字典，键为分区标签，值为直纹面参数
        """
        print("拟合直纹面...")
        print(f"使用神经网络：{self.use_neural}")
        
        # 1. 输入与预处理
        self._preprocess(partition_labels, edge_midpoints)
        
        # 2. 使用神经网络或传统方法拟合
        if self.use_neural and self.neural_fitter:
            developable_surfaces = self._neural_fit_all_partitions()
        else:
            # 传统方法：种子分区判定 + 生长循环
            seed_partitions = self._identify_seed_partitions()
            developable_surfaces = self._growth_loop(seed_partitions)
        
        # 3. 封闭性协调
        self._coordinate_shared_edges(developable_surfaces)
        
        # 4. 可视化直纹面拼接后的原曲面
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
        print(f"总分区数: {len(unique_labels)}")
        
        # 构建边缘中点映射
        if edge_midpoints is not None and edge_midpoints.size > 0:
            self.edge_midpoints = edge_midpoints
            print(f"边缘中点数量: {len(edge_midpoints)}")
        else:
            self.edge_midpoints = np.array([])
            print("没有边缘中点数据")
        
        # 统计分区大小和边界边数量
        partition_stats = []
        
        for label in unique_labels[:5]:  # 只打印前5个分区的信息
            # 获取该分区的所有顶点
            partition_vertices = np.where(partition_labels == label)[0]
            
            # 提取边界边
            boundary_edges = self._extract_boundary_edges(partition_vertices, partition_labels)
            
            # 提取边缘点列（用于神经网络）
            edge_points = self._extract_edge_points(boundary_edges)
            
            # 提取内部点云（用于传统方法）
            interior_points = self._extract_interior_points(partition_vertices, boundary_edges)
            
            # 存储分区信息
            self.partitions[label] = {
                'vertices': partition_vertices,
                'boundary_edges': boundary_edges,
                'edge_points': edge_points,  # 边缘点列（神经网络输入）
                'interior_points': interior_points,
                'known_edges': set(),
                'skip_count': 0,
                'label': label
            }
            
            # 记录统计信息
            partition_stats.append({
                'label': label,
                'num_vertices': len(partition_vertices),
                'num_boundary_edges': len(boundary_edges),
                'num_edge_points': len(edge_points),
                'num_interior_points': len(interior_points)
            })
        
        # 打印前5个分区的信息
        print("前5个分区的信息:")
        for stat in partition_stats:
            print(f"分区 {stat['label']}: 顶点数={stat['num_vertices']}, 边界边数={stat['num_boundary_edges']}, 边缘点列数={stat['num_edge_points']}, 内部点云数={stat['num_interior_points']}")
        
        # 处理剩余的分区
        for label in unique_labels[5:]:
            # 获取该分区的所有顶点
            partition_vertices = np.where(partition_labels == label)[0]
            
            # 提取边界边
            boundary_edges = self._extract_boundary_edges(partition_vertices, partition_labels)
            
            # 提取边缘点列（用于神经网络）
            edge_points = self._extract_edge_points(boundary_edges)
            
            # 提取内部点云（用于传统方法）
            interior_points = self._extract_interior_points(partition_vertices, boundary_edges)
            
            # 存储分区信息
            self.partitions[label] = {
                'vertices': partition_vertices,
                'boundary_edges': boundary_edges,
                'edge_points': edge_points,  # 边缘点列（神经网络输入）
                'interior_points': interior_points,
                'known_edges': set(),
                'skip_count': 0,
                'label': label
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
                    edge = tuple(sorted([v, neighbor]))
                    if edge not in boundary_edges:
                        boundary_edges.append(list(edge))
        
        # 如果边界边数量不足，使用分区内的边
        if len(boundary_edges) < 3:
            # 从分区内的所有边中选择一些作为边界边
            all_edges = []
            for v in partition_vertices:
                for neighbor in self.adjacency[v]:
                    if neighbor in vertex_set:
                        edge = tuple(sorted([v, neighbor]))
                        if edge not in all_edges:
                            all_edges.append(list(edge))
            
            # 选择足够的边
            if all_edges:
                # 如果边数不足3，复制现有边
                while len(boundary_edges) < 3 and all_edges:
                    boundary_edges.extend(all_edges[:3 - len(boundary_edges)])
        
        # 确保至少有3条边
        if len(boundary_edges) < 3 and len(partition_vertices) >= 2:
            # 创建虚拟边
            for i in range(min(3, len(partition_vertices))):
                for j in range(i + 1, min(3, len(partition_vertices))):
                    edge = [partition_vertices[i], partition_vertices[j]]
                    edge_tuple = tuple(sorted(edge))
                    if edge_tuple not in [tuple(sorted(e)) for e in boundary_edges]:
                        boundary_edges.append(edge)
                    if len(boundary_edges) >= 3:
                        break
                if len(boundary_edges) >= 3:
                    break
        
        return boundary_edges
    
    def _extract_edge_points(self, boundary_edges: List[List[int]]) -> List[List[np.ndarray]]:
        """
        提取边缘点列（用于神经网络）
        Args:
            boundary_edges: 边界边列表
        Returns:
            边缘点列列表，每条边是一个点数组
        """
        edge_points = []
        for edge in boundary_edges:
            # 提取边上的所有顶点
            points = [self.vertices[v] for v in edge]
            
            # 如果边只有两个点，在中间插值生成更多点
            if len(points) == 2:
                num_interpolated = 10  # 插值生成10个点
                interpolated_points = []
                p0 = points[0]
                p1 = points[1]
                for i in range(num_interpolated + 1):
                    t = i / num_interpolated
                    interpolated_point = (1 - t) * p0 + t * p1
                    interpolated_points.append(interpolated_point)
                edge_points.append(interpolated_points)
            else:
                edge_points.append(points)
        return edge_points
    
    def _extract_interior_points(self, partition_vertices: np.ndarray, boundary_edges: List[List[int]]) -> List[np.ndarray]:
        """
        提取分区的内部点云（用于传统方法）
        Args:
            partition_vertices: 分区的顶点索引数组
            boundary_edges: 边界边列表
        Returns:
            内部点云列表
        """
        boundary_vertices = set()
        for edge in boundary_edges:
            boundary_vertices.update(edge)
        
        interior_vertices = [v for v in partition_vertices if v not in boundary_vertices]
        interior_points = [self.vertices[v] for v in interior_vertices]
        
        return interior_points
    
    def _neural_fit_all_partitions(self) -> Dict[int, Dict[str, Any]]:
        """
        使用神经网络拟合所有分区
        Returns:
            直纹面字典
        """
        print("使用NURBS神经网络拟合直纹面...")
        
        developable_surfaces = {}
        
        for label, partition in self.partitions.items():
            edge_points = partition['edge_points']
            interior_points = partition['interior_points']
            
            # 确保内部点云不为空
            if not interior_points:
                # 如果内部点云为空，使用边界点作为内部点
                interior_points = []
                for edge in edge_points:
                    interior_points.extend(edge)
                # 去重
                interior_points = list({tuple(p) for p in interior_points})
                # 如果仍然为空，添加一些默认点
                if not interior_points:
                    interior_points = [np.array([0.0, 0.0, 0.0])]
            
            # 确保边缘点列数量足够
            while len(edge_points) < 3:
                # 如果边缘点列不足，复制现有边
                if edge_points:
                    edge_points.append(edge_points[0])
                else:
                    # 如果没有边缘点列，创建默认边
                    edge_points.append([np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])])
            
            # 使用神经网络预测
            try:
                surface = self.neural_fitter.predict(edge_points, interior_points)
                surface['vertices'] = partition['vertices']
                surface['label'] = label
                print(f"分区 {label}: NURBS神经网络拟合成功")
            except Exception as e:
                print(f"分区 {label} 神经网络拟合失败：{e}，使用传统方法")
                surface = self._fit_partition_traditional(label, partition)
            
            if surface:
                developable_surfaces[label] = surface
        
        print(f"NURBS神经网络拟合完成，处理了 {len(developable_surfaces)} 个分区")
        return developable_surfaces
    
    def _fit_partition_traditional(self, label: int, partition: Dict[str, Any]) -> Dict[str, Any]:
        """
        传统方法拟合单个分区
        Args:
            label: 分区标签
            partition: 分区信息
        Returns:
            直纹面参数
        """
        boundary_edges = partition['boundary_edges']
        
        if len(boundary_edges) == 3:
            return self._fit_triangular_partition(label, partition)
        elif len(boundary_edges) == 4:
            return self._fit_quadrilateral_partition(label, partition)
        else:
            return self._fit_general_partition(label, partition)
    
    def _process_shared_edges(self):
        """处理共享边"""
        print("处理共享边...")
        
        all_edges = {}
        for label, partition in self.partitions.items():
            for edge in partition['boundary_edges']:
                edge_key = tuple(edge)
                if edge_key not in all_edges:
                    all_edges[edge_key] = []
                all_edges[edge_key].append(label)
        
        for edge_key, labels in all_edges.items():
            if len(labels) > 1:
                self.shared_edges[edge_key] = labels
    
    def _merge_vertices(self):
        """合并顶点"""
        print("合并顶点...")
        
        all_vertices = []
        for label, partition in self.partitions.items():
            all_vertices.extend(partition['vertices'])
        
        merged = {}
        for i, v1 in enumerate(all_vertices):
            if v1 not in merged:
                merged[v1] = v1
                for v2 in all_vertices[i+1:]:
                    if v2 not in merged:
                        distance = np.linalg.norm(self.vertices[v1] - self.vertices[v2])
                        if distance < self.epsilon:
                            merged[v2] = v1
        
        self.vertex_map = merged
        
        for label, partition in self.partitions.items():
            updated_edges = []
            for edge in partition['boundary_edges']:
                updated_edge = [self.vertex_map.get(v, v) for v in edge]
                updated_edges.append(updated_edge)
            partition['boundary_edges'] = updated_edges
    
    def _detect_edge_types(self):
        """检测边类型"""
        print("检测边类型...")
        
        for label, partition in self.partitions.items():
            edge_types = {}
            for edge in partition['boundary_edges']:
                edge_points = [self.vertices[v] for v in edge]
                
                if len(edge_points) > 1:
                    mean = np.mean(edge_points, axis=0)
                    centered = edge_points - mean
                    cov = np.cov(centered.T)
                    eigenvalues, _ = np.linalg.eigh(cov)
                    eigenvalues = sorted(eigenvalues, reverse=True)
                    
                    if eigenvalues[0] / (eigenvalues[1] + 1e-8) > self.T:
                        edge_types[tuple(edge)] = 'straight'
                    else:
                        edge_types[tuple(edge)] = 'curved'
                else:
                    edge_types[tuple(edge)] = 'straight'
            
            partition['edge_types'] = edge_types
    
    def _identify_seed_partitions(self) -> List[int]:
        """识别种子分区"""
        print("识别种子分区...")
        
        seed_partitions = []
        
        for label, partition in self.partitions.items():
            boundary_edges = partition['boundary_edges']
            
            if len(boundary_edges) == 3:
                vertices = set()
                for edge in boundary_edges:
                    vertices.update(edge)
                if len(vertices) == 3:
                    seed_partitions.append(label)
                    for edge in boundary_edges:
                        partition['known_edges'].add(tuple(edge))
            
            elif len(boundary_edges) == 4:
                vertices = set()
                for edge in boundary_edges:
                    vertices.update(edge)
                if len(vertices) == 4:
                    seed_partitions.append(label)
                    for edge in boundary_edges:
                        partition['known_edges'].add(tuple(edge))
        
        if len(seed_partitions) == 0:
            print("没有识别到三角形或四边形分区，选择其他分区作为种子")
            sorted_partitions = sorted(self.partitions.items(), key=lambda x: len(x[1]['vertices']))
            mid_index = len(sorted_partitions) // 2
            for i in range(max(0, mid_index - 1), min(len(sorted_partitions), mid_index + 2)):
                label, partition = sorted_partitions[i]
                seed_partitions.append(label)
                for edge in partition['boundary_edges']:
                    partition['known_edges'].add(tuple(edge))
        
        print(f"识别到 {len(seed_partitions)} 个种子分区")
        return seed_partitions
    
    def _growth_loop(self, seed_partitions: List[int]) -> Dict[int, Dict[str, Any]]:
        """生长循环"""
        print("生长循环...")
        
        developable_surfaces = {}
        queue = seed_partitions.copy()
        processed = set()
        
        while queue:
            label = queue.pop(0)
            if label in processed:
                continue
            processed.add(label)
            
            partition = self.partitions[label]
            boundary_edges = partition['boundary_edges']
            
            if len(boundary_edges) == 3:
                surface = self._fit_triangular_partition(label, partition)
            elif len(boundary_edges) == 4:
                surface = self._fit_quadrilateral_partition(label, partition)
            else:
                surface = self._fit_general_partition(label, partition)
            
            if surface:
                developable_surfaces[label] = surface
                
                for edge in boundary_edges:
                    edge_key = tuple(edge)
                    if edge_key not in partition['known_edges']:
                        partition['known_edges'].add(edge_key)
                        
                        if edge_key in self.shared_edges:
                            for neighbor_label in self.shared_edges[edge_key]:
                                if neighbor_label != label and neighbor_label not in processed:
                                    neighbor_partition = self.partitions[neighbor_label]
                                    neighbor_partition['known_edges'].add(edge_key)
                                    if neighbor_label not in queue:
                                        queue.append(neighbor_label)
            else:
                partition['skip_count'] += 1
                if partition['skip_count'] < self.max_skip_count:
                    queue.append(label)
                else:
                    surface = self._force_fit_partition(label, partition)
                    if surface:
                        developable_surfaces[label] = surface
        
        print(f"处理了 {len(processed)} 个分区，开始处理剩余分区...")
        for label, partition in self.partitions.items():
            if label not in processed:
                surface = self._force_fit_partition(label, partition)
                if surface:
                    developable_surfaces[label] = surface
        
        print(f"生长循环完成，拟合了 {len(developable_surfaces)} 个直纹面")
        return developable_surfaces
    
    def _fit_triangular_partition(self, label: int, partition: Dict[str, Any]) -> Dict[str, Any]:
        """拟合三角形分区"""
        boundary_edges = partition['boundary_edges']
        known_edges = partition['known_edges']
        
        if len(known_edges) < 2:
            return None
        
        edge_vertices = []
        for edge in known_edges:
            edge_vertices.extend(edge)
        
        vertex_count = {}
        for v in edge_vertices:
            vertex_count[v] = vertex_count.get(v, 0) + 1
        
        common_vertex = None
        for v, count in vertex_count.items():
            if count == 2:
                common_vertex = v
                break
        
        if not common_vertex:
            return None
        
        other_vertices = []
        for edge in known_edges:
            for v in edge:
                if v != common_vertex:
                    other_vertices.append(v)
        
        if len(other_vertices) != 2:
            return None
        
        A, B = other_vertices
        
        third_edge = None
        for edge in boundary_edges:
            edge_key = tuple(edge)
            if edge_key not in known_edges:
                third_edge = edge
                break
        
        if third_edge:
            curve = self._fit_curve([self.vertices[v] for v in third_edge])
        else:
            curve = {
                'type': 'line',
                'start_point': self.vertices[A].tolist(),
                'end_point': self.vertices[B].tolist()
            }
        
        surface = {
            'type': 'conical',
            'vertex': self.vertices[common_vertex].tolist(),
            'curve': curve,
            'vertices': partition['vertices'],
            'label': label
        }
        
        return surface
    
    def _fit_quadrilateral_partition(self, label: int, partition: Dict[str, Any]) -> Dict[str, Any]:
        """拟合四边形分区"""
        boundary_edges = partition['boundary_edges']
        known_edges = partition['known_edges']
        
        if len(known_edges) < 2:
            return None
        
        straight_edges = []
        for edge in known_edges:
            if partition['edge_types'].get(edge) == 'straight':
                straight_edges.append(edge)
        
        if len(straight_edges) < 2:
            return None
        
        L0, L1 = straight_edges[:2]
        P00, P01 = L0
        P10, P11 = L1
        
        curve0 = self._fit_curve([self.vertices[P00], self.vertices[P10]])
        curve1 = self._fit_curve([self.vertices[P01], self.vertices[P11]])
        
        surface = {
            'type': 'developable',
            'curve0': curve0,
            'curve1': curve1,
            'vertices': partition['vertices'],
            'label': label
        }
        
        return surface
    
    def _fit_general_partition(self, label: int, partition: Dict[str, Any]) -> Dict[str, Any]:
        """拟合一般分区"""
        boundary_edges = partition['boundary_edges']
        known_edges = partition['known_edges']
        
        if len(known_edges) < 2:
            return None
        
        known_edge_list = list(known_edges)
        edge1, edge2 = known_edge_list[:2]
        
        curve1 = self._fit_curve([self.vertices[v] for v in edge1])
        curve2 = self._fit_curve([self.vertices[v] for v in edge2])
        
        surface = {
            'type': 'developable',
            'curve1': curve1,
            'curve2': curve2,
            'vertices': partition['vertices'],
            'label': label
        }
        
        return surface
    
    def _force_fit_partition(self, label: int, partition: Dict[str, Any]) -> Dict[str, Any]:
        """强制拟合分区"""
        boundary_edges = partition['boundary_edges']
        
        if len(boundary_edges) < 2:
            return None
        
        edge1, edge2 = boundary_edges[:2]
        
        curve1 = self._fit_curve([self.vertices[v] for v in edge1])
        curve2 = self._fit_curve([self.vertices[v] for v in edge2])
        
        surface = {
            'type': 'developable',
            'curve1': curve1,
            'curve2': curve2,
            'vertices': partition['vertices'],
            'label': label
        }
        
        return surface
    
    def _fit_curve(self, points: List[np.ndarray]) -> Dict[str, Any]:
        """拟合曲线"""
        points = np.array(points)
        n = len(points)
        
        if n == 0:
            return {
                'type': 'line',
                'start_point': [0.0, 0.0, 0.0],
                'end_point': [1.0, 0.0, 0.0]
            }
        
        if n == 1:
            point = points[0]
            return {
                'type': 'line',
                'start_point': point.tolist(),
                'end_point': point.tolist()
            }
        
        t = np.linspace(0, 1, n)
        degree = min(3, n - 1)
        
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
        """协调共享边"""
        print("协调共享边...")
        
        edge_curves = {}
        
        for label, surface in developable_surfaces.items():
            partition = self.partitions[label]
            for edge in partition['boundary_edges']:
                edge_key = tuple(edge)
                if edge_key in self.shared_edges:
                    if surface['type'] == 'conical':
                        curve = surface['curve']
                    else:
                        curve = surface.get('curve1', surface.get('curve0'))
                    
                    if edge_key not in edge_curves:
                        edge_curves[edge_key] = []
                    edge_curves[edge_key].append(curve)
        
        for edge_key, curves in edge_curves.items():
            if len(curves) > 1:
                avg_curve = self._average_curves(curves)
                
                for label in self.shared_edges[edge_key]:
                    if label in developable_surfaces:
                        surface = developable_surfaces[label]
                        if surface['type'] == 'conical':
                            surface['curve'] = avg_curve
                        else:
                            if 'curve1' in surface:
                                surface['curve1'] = avg_curve
                            if 'curve0' in surface:
                                surface['curve0'] = avg_curve
    
    def _average_curves(self, curves: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算平均曲线"""
        return curves[0]
    
    def _compute_basis_function(self, u, knot_vector, degree, i):
        """
        计算B样条基函数（Cox-de Boor递推公式）
        """
        if degree == 0:
            return 1.0 if knot_vector[i] <= u < knot_vector[i+1] else 0.0
        else:
            denominator1 = knot_vector[i+degree] - knot_vector[i]
            denominator2 = knot_vector[i+degree+1] - knot_vector[i+1]
            
            term1 = 0.0
            if denominator1 > 1e-8:
                term1 = (u - knot_vector[i]) / denominator1 * self._compute_basis_function(u, knot_vector, degree-1, i)
            
            term2 = 0.0
            if denominator2 > 1e-8:
                term2 = (knot_vector[i+degree+1] - u) / denominator2 * self._compute_basis_function(u, knot_vector, degree-1, i+1)
            
            return term1 + term2
    
    def visualize_developable_assembly(self, developable_surfaces: Dict[int, Dict[str, Any]]):
        """可视化直纹面拼接后的原曲面"""
        print("可视化直纹面拼接后的原曲面...")
        
        meshes = []
        valid_meshes = 0
        all_vertices = []
        
        for label, surface in developable_surfaces.items():
            mesh = self._generate_developable_mesh(surface)
            if mesh:
                # 检查网格是否有效
                vertices = np.asarray(mesh.vertices)
                triangles = np.asarray(mesh.triangles)
                if len(vertices) > 0 and len(triangles) > 0:
                    # 检查顶点是否有效（非NaN和非无穷大）
                    if np.all(np.isfinite(vertices)):
                        color = np.random.rand(3)
                        mesh.paint_uniform_color(color)
                        meshes.append(mesh)
                        all_vertices.extend(vertices)
                        valid_meshes += 1
                    else:
                        print(f"分区 {label} 的网格顶点无效")
                else:
                    print(f"分区 {label} 的网格无效：顶点数={len(vertices)}, 三角形数={len(triangles)}")
        
        print(f"生成了 {valid_meshes} 个有效直纹面网格")
        
        if meshes:
            # 计算所有顶点的边界框
            all_vertices = np.array(all_vertices)
            if all_vertices.size > 0:
                min_bound = np.min(all_vertices, axis=0)
                max_bound = np.max(all_vertices, axis=0)
                center = (min_bound + max_bound) / 2
                extent = max_bound - min_bound
                print(f"网格边界框: 最小={min_bound}, 最大={max_bound}, 中心={center}, 范围={extent}")
            else:
                # 如果没有有效顶点，使用默认值
                center = np.array([0.0, 0.0, 0.0])
                extent = np.array([10.0, 10.0, 10.0])
            
            # 创建可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="直纹面拼接后的原曲面", width=1024, height=768)
            
            # 添加所有网格
            for i, mesh in enumerate(meshes):
                vis.add_geometry(mesh)
            
            # 设置渲染选项
            render_option = vis.get_render_option()
            render_option.mesh_show_back_face = True
            render_option.mesh_show_wireframe = False  # 关闭线框，避免密密麻麻的纹路
            render_option.background_color = np.array([1, 1, 1])  # 白色背景，提高可见性
            render_option.point_size = 2.0
            render_option.light_on = True  # 开启光照，提高颜色显示效果
            
            # 设置视图控制
            ctr = vis.get_view_control()
            # 简化相机设置，确保兼容性
            ctr.set_lookat(center)
            ctr.set_up([0, 1, 0])
            
            # 尝试设置相机距离，基于网格范围
            try:
                # 计算合适的相机距离
                distance = np.max(extent) * 2
                if distance < 10.0:
                    distance = 10.0
                # 使用Open3D的视角控制方法
                ctr.set_zoom(0.5)  # 缩放视图
            except Exception as e:
                print(f"设置相机参数时出错: {e}")
            
            # 手动更新渲染
            vis.poll_events()
            vis.update_renderer()
            
            # 运行可视化
            print("正在显示直纹面拼接结果...")
            print(f"网格数量: {len(meshes)}")
            print("提示：使用鼠标拖动可以旋转视角，滚轮可以缩放")
            vis.run()
            vis.destroy_window()
        else:
            print("没有生成有效的直纹面网格")
        
        print("直纹面拼接可视化完成")
    
    def _generate_developable_mesh(self, surface: Dict[str, Any]) -> o3d.geometry.TriangleMesh:
        """生成直纹面的网格"""
        num_curve_points = 20  # 减少曲线点数量
        num_generator_points = 5  # 减少生成器点数量
        
        vertices = []
        triangles = []
        
        try:
            for i in range(num_curve_points):
                t = i / (num_curve_points - 1)
                for j in range(num_generator_points):
                    s = j / (num_generator_points - 1)
                    point = self._evaluate_developable(surface, t, s)
                    # 检查点是否有效
                    if not np.any(np.isnan(point)) and not np.any(np.isinf(point)):
                        vertices.append(point)
                    else:
                        # 如果点无效，使用默认点
                        vertices.append(np.array([0.0, 0.0, 0.0]))
            
            # 确保有足够的顶点生成三角形
            if len(vertices) >= 3:
                for i in range(num_curve_points - 1):
                    for j in range(num_generator_points - 1):
                        idx0 = i * num_generator_points + j
                        idx1 = i * num_generator_points + (j + 1)
                        idx2 = (i + 1) * num_generator_points + j
                        idx3 = (i + 1) * num_generator_points + (j + 1)
                        
                        # 确保索引有效
                        if idx0 < len(vertices) and idx1 < len(vertices) and idx2 < len(vertices):
                            triangles.append([idx0, idx1, idx2])
                        if idx1 < len(vertices) and idx3 < len(vertices) and idx2 < len(vertices):
                            triangles.append([idx1, idx3, idx2])
            
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.compute_vertex_normals()
            
            return mesh
        except Exception as e:
            print(f"生成直纹面网格失败：{e}")
            # 返回一个空网格
            return o3d.geometry.TriangleMesh()
    
    def _evaluate_developable(self, surface: Dict[str, Any], t: float, s: float) -> np.ndarray:
        """计算直纹面上的点"""
        if surface['type'] == 'conical':
            vertex = np.array(surface['vertex'])
            curve_point = self._evaluate_curve(surface['curve'], t)
            return vertex + s * (curve_point - vertex)
        else:
            if 'curve1' in surface and 'curve2' in surface:
                p1 = self._evaluate_curve(surface['curve1'], t)
                p2 = self._evaluate_curve(surface['curve2'], t)
            elif 'curve0' in surface and 'curve1' in surface:
                p1 = self._evaluate_curve(surface['curve0'], t)
                p2 = self._evaluate_curve(surface['curve1'], t)
            else:
                return np.array([0.0, 0.0, 0.0])
            
            return (1 - s) * p1 + s * p2
    
    def _evaluate_curve(self, curve: Dict[str, Any], t: float) -> np.ndarray:
        """计算曲线上的点"""
        if curve['type'] == 'polynomial':
            x = np.polyval(curve['coeffs_x'], t)
            y = np.polyval(curve['coeffs_y'], t)
            z = np.polyval(curve['coeffs_z'], t)
            return np.array([x, y, z])
        elif curve['type'] == 'line':
            start_point = np.array(curve['start_point'])
            end_point = np.array(curve['end_point'])
            return (1 - t) * start_point + t * end_point
        elif curve['type'] == 'b-spline':
            # B 样条曲线评估（二次）
            control_points = np.array(curve['control_points'])
            M = len(control_points)
            
            u = t * (M - 1)
            k = int(np.floor(u))
            if k >= M - 1:
                k = M - 2
            u_local = u - k
            
            if k == 0:
                p0, p1, p2 = control_points[0], control_points[1], control_points[2]
            elif k == M - 2:
                p0, p1, p2 = control_points[-3], control_points[-2], control_points[-1]
            else:
                p0, p1, p2 = control_points[k], control_points[k+1], control_points[k+2]
            
            result = (1 - u_local)**2 / 2 * p0 + \
                     (-2*u_local**2 + 2*u_local + 1) / 2 * p1 + \
                     u_local**2 / 2 * p2
            
            return result
        elif curve['type'] == 'nurbs':
            # NURBS曲线评估
            control_points = np.array(curve['control_points'])
            weights = np.array(curve['weights'])
            knot_vector = np.array(curve['knot_vector'])
            degree = curve['degree']
            
            n = len(control_points) - 1
            numerator = np.zeros(3)
            denominator = 0.0
            
            for i in range(n+1):
                basis = self._compute_basis_function(t, knot_vector, degree, i)
                weighted_basis = basis * weights[i]
                numerator += weighted_basis * control_points[i]
                denominator += weighted_basis
            
            if denominator > 1e-8:
                return numerator / denominator
            else:
                return np.zeros(3)
        else:
            raise ValueError(f"不支持的曲线类型：{curve['type']}")
