"""
基于新指标的高级表面分区器
实现算法version2中描述的分区算法
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

from .meshProcessor import MeshProcessor


class AdvancedSurfacePartitioner:
    def __init__(self, mesh: MeshProcessor, tool, resolution=0.1):
        """
        初始化高级表面分区器
        Args:
            mesh: 网格处理器
            tool: 刀具对象
            resolution: 聚类分辨率参数，控制分区数量
        """
        self.mesh = mesh
        self.tool = tool
        self.num_vertices = len(mesh.vertices)
        self.resolution = resolution
        
        # 预计算几何特征
        self._precompute_features()

    def _precompute_features(self):
        """
        预计算顶点的几何特征
        """
        print("预计算顶点几何特征...")
        
        # 计算最大切削宽度
        self.mesh.calculate_max_cutting_width(self.tool)
        
        # 计算直纹面逼近误差
        self.mesh.calculate_rolled_error()
        
        print("顶点几何特征预计算完成")

    def _compute_local_curvature_similarity(self, i: int, j: int) -> float:
        """
        计算两个顶点之间的局部曲率相似性
        Args:
            i: 顶点1索引
            j: 顶点2索引
        Returns:
            局部曲率相似性指标，值越小表示曲率越相似
        """
        # 获取主曲率
        k1_i, k2_i = self.mesh.principal_curvatures[i]
        k1_j, k2_j = self.mesh.principal_curvatures[j]
        
        # 计算曲率差异
        curvature_diff = abs(k1_i - k1_j) + abs(k2_i - k2_j)
        
        # 计算测地距离（使用更精确的方法）
        distance = self._compute_geodesic_distance(i, j)
        if distance < 1e-8:
            distance = 1e-8
        
        # 计算局部曲率相似性
        similarity = curvature_diff / distance
        return similarity

    def _compute_geodesic_distance(self, i: int, j: int) -> float:
        """
        计算两个顶点之间的测地距离
        Args:
            i: 顶点1索引
            j: 顶点2索引
        Returns:
            测地距离
        """
        # 直接使用欧氏距离作为近似，对于局部邻域来说已经足够准确
        # 这样可以避免为每个顶点对构建完整的图，大大提高计算速度
        return np.linalg.norm(self.mesh.vertices[i] - self.mesh.vertices[j])

    def _compute_cutting_width_diff(self, i: int, j: int) -> float:
        """
        计算两个顶点之间的最大切削宽度差异
        Args:
            i: 顶点1索引
            j: 顶点2索引
        Returns:
            最大切削宽度差异
        """
        width_i = self.mesh.max_cutting_widths[i]
        width_j = self.mesh.max_cutting_widths[j]
        return abs(width_i - width_j)

    def _compute_rolled_error_diff(self, i: int, j: int) -> float:
        """
        计算两个顶点之间的直纹面逼近误差差异
        Args:
            i: 顶点1索引
            j: 顶点2索引
        Returns:
            直纹面逼近误差差异
        """
        error_i = self.mesh.rolled_error[i]
        error_j = self.mesh.rolled_error[j]
        return abs(error_i - error_j)

    def _precompute_feature_statistics(self):
        """
        预计算特征的统计信息，用于标准化和自适应权重
        """
        print("预计算特征统计信息...")
        
        # 收集所有特征值
        curvature_values = []
        width_values = []
        error_values = []
        
        for i in range(self.num_vertices):
            # 曲率特征
            k1, k2 = self.mesh.principal_curvatures[i]
            curvature_values.append(abs(k1) + abs(k2))
            
            # 切削宽度特征
            width_values.append(self.mesh.max_cutting_widths[i])
            
            # 直纹面误差特征
            error_values.append(self.mesh.rolled_error[i])
        
        # 计算统计信息
        self.curvature_mean = np.mean(curvature_values)
        self.curvature_std = np.std(curvature_values) if np.std(curvature_values) > 0 else 1.0
        
        self.width_mean = np.mean(width_values)
        self.width_std = np.std(width_values) if np.std(width_values) > 0 else 1.0
        
        self.error_mean = np.mean(error_values)
        self.error_std = np.std(error_values) if np.std(error_values) > 0 else 1.0
        
        print("特征统计信息预计算完成")

    def _build_weighted_adjacency_matrix(self) -> np.ndarray:
        """
        构建基于新指标的加权邻接矩阵
        Returns:
            加权邻接矩阵
        """
        print("构建加权邻接矩阵...")
        
        # 预计算特征统计信息
        self._precompute_feature_statistics()
        
        n = self.num_vertices
        
        # 使用稀疏矩阵表示，节省内存和计算时间
        from scipy.sparse import lil_matrix
        adjacency_matrix = lil_matrix((n, n))
        
        # 并行计算权重
        def compute_weight_batch(batch):
            results = []
            for i, j in batch:
                # 早期终止：如果两个顶点距离太远，直接跳过
                distance = np.linalg.norm(self.mesh.vertices[i] - self.mesh.vertices[j])
                if distance > 1.0:  # 增大距离阈值，保留更多边
                    continue
                
                # 计算三个指标
                curvature_sim = self._compute_local_curvature_similarity(i, j)
                width_diff = self._compute_cutting_width_diff(i, j)
                error_diff = self._compute_rolled_error_diff(i, j)
                
                # 标准化指标
                curvature_sim_norm = curvature_sim / (self.curvature_std + 1e-8)
                width_diff_norm = width_diff / (self.width_std + 1e-8)
                error_diff_norm = error_diff / (self.error_std + 1e-8)
                
                # 使用高斯核函数归一化
                sigma1 = 1.0
                sigma2 = 1.0
                sigma3 = 1.0
                
                f1 = np.exp(-curvature_sim_norm**2 / (2 * sigma1**2))
                f2 = np.exp(-width_diff_norm**2 / (2 * sigma2**2))
                f3 = np.exp(-error_diff_norm**2 / (2 * sigma3**2))
                
                # 自适应权重
                # 根据特征的重要性动态调整权重
                lambda1 = 0.3 + 0.2 * (self.curvature_std / (self.curvature_std + self.width_std + self.error_std))
                lambda2 = 0.4 + 0.2 * (self.width_std / (self.curvature_std + self.width_std + self.error_std))
                lambda3 = 0.3 + 0.2 * (self.error_std / (self.curvature_std + self.width_std + self.error_std))
                
                # 归一化权重
                total_lambda = lambda1 + lambda2 + lambda3
                lambda1 /= total_lambda
                lambda2 /= total_lambda
                lambda3 /= total_lambda
                
                weight = lambda1 * f1 + lambda2 * f2 + lambda3 * f3
                
                # 阈值过滤：只保留权重较高的边
                if weight > 0.1:
                    results.append((i, j, weight))
            return results
        
        # 动态批次大小：根据图的大小自动调整
        batch_size = min(2000, max(500, n // 10))
        tasks = []
        for i in range(n):
            neighbors = self.mesh.adjacency[i]
            for j in neighbors:
                if i < j:
                    tasks.append((i, j))
        
        # 分批次处理
        batches = [tasks[i:i+batch_size] for i in range(0, len(tasks), batch_size)]
        
        # 并行处理批次
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(compute_weight_batch, batch): batch for batch in batches}
            for future in as_completed(futures):
                results = future.result()
                for i, j, weight in results:
                    adjacency_matrix[i, j] = weight
                    adjacency_matrix[j, i] = weight
        
        # 转换为稠密矩阵
        adjacency_matrix = adjacency_matrix.toarray()
        
        # 归一化矩阵
        max_weight = np.max(adjacency_matrix)
        if max_weight > 0:
            adjacency_matrix /= max_weight
        
        print("加权邻接矩阵构建完成")
        return adjacency_matrix

    def _leiden_clustering(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        使用Leiden聚类算法进行分区
        Args:
            adjacency_matrix: 加权邻接矩阵
        Returns:
            分区标签数组
        """
        print("执行Leiden聚类...")
        
        # 构建图
        G = nx.Graph()
        G.add_nodes_from(range(self.num_vertices))
        
        # 添加边和权重
        for i in range(self.num_vertices):
            for j in range(i+1, self.num_vertices):
                if adjacency_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=adjacency_matrix[i, j])
        
        # 尝试导入leidenalg
        try:
            import leidenalg
            import igraph as ig
            
            # 转换为igraph格式
            g = ig.Graph.from_networkx(G)
            
            # 确保边权重属性存在
            if 'weight' not in g.es.attributes():
                # 提取边权重并设置
                weights = [G[u][v]['weight'] for u, v in G.edges()]
                g.es['weight'] = weights
            
            # 自适应分辨率参数
            # 根据图的大小和边权重分布自动调整resolution
            edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
            avg_weight = np.mean(edge_weights) if edge_weights else 0.5
            
            # 基于图大小和平均权重计算自适应分辨率
            # 进一步降低分辨率以大幅减少分区数量
            adaptive_resolution = self.resolution * 0.1 * (1 + 0.5 * (1 - avg_weight))
            
            # 多分辨率聚类
            # 先使用较低分辨率获取初始分区
            initial_partition = leidenalg.find_partition(
                g, 
                leidenalg.CPMVertexPartition, 
                resolution_parameter=adaptive_resolution * 0.2,
                weights="weight"
            )
            
            # 再使用较高分辨率进行细化
            refined_partition = leidenalg.find_partition(
                g, 
                leidenalg.CPMVertexPartition, 
                resolution_parameter=adaptive_resolution,
                weights="weight",
                initial_membership=initial_partition.membership
            )
            
            # 获取分区标签
            labels = np.zeros(self.num_vertices, dtype=int)
            for node, community in enumerate(refined_partition.membership):
                labels[node] = community
            
            print(f"Leiden聚类完成: {len(np.unique(labels))} 个分区")
            return labels
        except ImportError:
            print("leidenalg库未安装，使用备用聚类方法")
            # 使用备用聚类方法
            return self._alternative_clustering(G)

    def _alternative_clustering(self, G: nx.Graph) -> np.ndarray:
        """
        备用聚类方法
        Args:
            G: 网络图
        Returns:
            分区标签数组
        """
        print("使用备用聚类方法...")
        
        # 使用社区检测算法
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        
        # 获取分区标签
        labels = np.zeros(self.num_vertices, dtype=int)
        for node, community in partition.items():
            labels[node] = community
        
        print(f"备用聚类完成: {len(np.unique(labels))} 个分区")
        return labels

    def _fit_partition_boundaries(self, labels: np.ndarray) -> np.ndarray:
        """
        拟合分区边界
        Args:
            labels: 分区标签数组
        Returns:
            优化后的分区标签数组
        """
        print("拟合分区边界...")
        
        # 提取边界顶点
        boundary_vertices = []
        for i in range(self.num_vertices):
            neighbors = self.mesh.adjacency[i]
            neighbor_labels = [labels[j] for j in neighbors]
            if len(set(neighbor_labels)) > 1:
                boundary_vertices.append(i)
        
        # 特殊边界处理
        # 检测并处理尖锐边界和复杂边界
        smoothed_labels = labels.copy()
        
        # 对边界进行平滑处理
        for i in boundary_vertices:
            neighbors = self.mesh.adjacency[i]
            if neighbors:
                # 计算邻居的标签分布
                label_counts = {}
                for j in neighbors:
                    # 使用权重：距离越近，权重越大
                    distance = np.linalg.norm(self.mesh.vertices[i] - self.mesh.vertices[j])
                    weight = 1.0 / (distance + 1e-8)
                    label_counts[labels[j]] = label_counts.get(labels[j], 0) + weight
                
                # 选择权重最大的标签
                most_common_label = max(label_counts, key=label_counts.get)
                smoothed_labels[i] = most_common_label
        
        # 边界平滑参数调整
        # 进行多次平滑迭代以获得更平滑的边界
        num_iterations = 2
        for iteration in range(num_iterations):
            temp_labels = smoothed_labels.copy()
            for i in boundary_vertices:
                neighbors = self.mesh.adjacency[i]
                if neighbors:
                    label_counts = {}
                    for j in neighbors:
                        distance = np.linalg.norm(self.mesh.vertices[i] - self.mesh.vertices[j])
                        weight = 1.0 / (distance + 1e-8)
                        label_counts[temp_labels[j]] = label_counts.get(temp_labels[j], 0) + weight
                    
                    most_common_label = max(label_counts, key=label_counts.get)
                    smoothed_labels[i] = most_common_label
        
        print("分区边界拟合完成")
        return smoothed_labels

    def partition_surface(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行表面分区
        Returns:
            分区标签数组和中点边缘点数组
        """
        print("开始表面分区...")
        
        # 1. 构建加权邻接矩阵
        adjacency_matrix = self._build_weighted_adjacency_matrix()
        
        # 2. 执行Leiden聚类
        labels = self._leiden_clustering(adjacency_matrix)
        
        # 3. 应用对称性约束
        labels = self._apply_symmetry_constraints(labels)
        
        # 4. 拟合分区边界
        labels = self._fit_partition_boundaries(labels)
        
        # 5. 确保分区连通性
        labels = self._ensure_connectivity(labels)
        
        # 6. 重新编号标签为连续的整数
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
        
        # 7. 提取中点边缘
        edge_midpoints = self._extract_edge_midpoints(labels)
        
        print(f"分区完成: {len(unique_labels)} 个分区")
        
        return labels, edge_midpoints

    def _detect_symmetry(self) -> Dict[str, Any]:
        """
        检测曲面的对称性
        Returns:
            对称性信息字典
        """
        print("检测曲面对称性...")
        
        # 计算曲面的中心
        center = np.mean(self.mesh.vertices, axis=0)
        
        # 检测球对称性
        distances = np.linalg.norm(self.mesh.vertices - center, axis=1)
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        is_spherical = std_distance / avg_distance < 0.1  # 球体的距离标准差较小
        
        symmetry_info = {
            'center': center,
            'is_spherical': is_spherical,
            'avg_distance': avg_distance,
            'std_distance': std_distance
        }
        
        print(f"对称性检测完成: 球面={is_spherical}")
        return symmetry_info

    def _apply_symmetry_constraints(self, labels: np.ndarray) -> np.ndarray:
        """
        应用对称性约束到分区标签
        Args:
            labels: 分区标签数组
        Returns:
            应用对称约束后的分区标签
        """
        print("应用对称性约束...")
        
        # 检测对称性
        symmetry_info = self._detect_symmetry()
        
        if not symmetry_info['is_spherical']:
            print("非球面，跳过对称性约束")
            return labels
        
        # 对于球面，应用对称约束
        center = symmetry_info['center']
        symmetric_labels = labels.copy()
        
        # 为每个顶点找到其对称点
        for i in range(self.num_vertices):
            # 计算对称点
            vec = self.mesh.vertices[i] - center
            symmetric_point = center - vec
            
            # 找到最接近对称点的顶点
            distances = np.linalg.norm(self.mesh.vertices - symmetric_point, axis=1)
            j = np.argmin(distances)
            
            # 确保对称点的标签相同
            if distances[j] < 1e-4:  # 阈值，确保是真正的对称点
                symmetric_labels[j] = labels[i]
        
        print("对称性约束应用完成")
        return symmetric_labels

    def _ensure_connectivity(self, labels: np.ndarray) -> np.ndarray:
        """
        确保每个分区都是连通的
        Args:
            labels: 分区标签数组
        Returns:
            连通的分区标签
        """
        print("确保分区连通性...")
        
        n = self.num_vertices
        visited = np.zeros(n, dtype=bool)
        
        # 构建邻接图
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        for v in range(n):
            neighbors = self.mesh.adjacency[v]
            for neighbor in neighbors:
                if v < neighbor:
                    G.add_edge(v, neighbor)
        
        # 检查每个分区的连通性
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            vertices = np.where(labels == label)[0]
            if len(vertices) > 0:
                # 计算连通分量
                subgraph = G.subgraph(vertices)
                components = list(nx.connected_components(subgraph))
                
                # 如果有多个连通分量，将小分量重新分配
                if len(components) > 1:
                    # 找到最大的连通分量
                    main_component = max(components, key=len)
                    
                    # 重新分配小分量
                    for component in components:
                        if component != main_component:
                            # 为小分量分配新标签
                            new_label = max(unique_labels) + 1
                            unique_labels = np.append(unique_labels, new_label)
                            
                            for v in component:
                                labels[v] = new_label
                            
                            print(f"修复分区连通性: 将 {len(component)} 个顶点分配到新分区 {new_label}")
        
        print("分区连通性检查完成")
        return labels

    def _extract_edge_midpoints(self, labels: np.ndarray) -> np.ndarray:
        """
        提取分区边缘的中点
        Args:
            labels: 分区标签数组
        Returns:
            中点边缘点数组
        """
        print("提取边缘中点...")
        
        edge_pairs = {}
        vertices = self.mesh.vertices
        
        for v in range(len(vertices)):
            neighbors = self.mesh.adjacency[v]
            for neighbor in neighbors:
                if labels[neighbor] != labels[v]:
                    edge_key = tuple(sorted([v, neighbor]))
                    if edge_key not in edge_pairs:
                        edge_pairs[edge_key] = {
                            'point1': vertices[v],
                            'point2': vertices[neighbor],
                            'label1': labels[v],
                            'label2': labels[neighbor]
                        }
        
        midpoints = []
        for edge_data in edge_pairs.values():
            midpoint = (edge_data['point1'] + edge_data['point2']) / 2
            midpoints.append(midpoint)
        
        print(f"提取完成: {len(midpoints)} 个中点边缘点")
        return np.array(midpoints)
