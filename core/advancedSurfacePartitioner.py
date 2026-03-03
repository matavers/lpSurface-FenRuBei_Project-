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
        
        # 计算测地距离
        # 使用欧氏距离作为近似
        distance = np.linalg.norm(self.mesh.vertices[i] - self.mesh.vertices[j])
        if distance < 1e-8:
            distance = 1e-8
        
        # 计算局部曲率相似性
        similarity = curvature_diff / distance
        return similarity

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

    def _build_weighted_adjacency_matrix(self) -> np.ndarray:
        """
        构建基于新指标的加权邻接矩阵
        Returns:
            加权邻接矩阵
        """
        print("构建加权邻接矩阵...")
        
        n = self.num_vertices
        adjacency_matrix = np.zeros((n, n))
        
        # 并行计算权重
        def compute_weight_batch(batch):
            results = []
            for i, j in batch:
                # 计算三个指标
                curvature_sim = self._compute_local_curvature_similarity(i, j)
                width_diff = self._compute_cutting_width_diff(i, j)
                error_diff = self._compute_rolled_error_diff(i, j)
                
                # 使用高斯核函数归一化
                sigma1 = 1.0
                sigma2 = 0.1
                sigma3 = 0.1
                
                f1 = np.exp(-curvature_sim**2 / (2 * sigma1**2))
                f2 = np.exp(-width_diff**2 / (2 * sigma2**2))
                f3 = np.exp(-error_diff**2 / (2 * sigma3**2))
                
                # 加权组合
                lambda1 = 0.3
                lambda2 = 0.4
                lambda3 = 0.3
                weight = lambda1 * f1 + lambda2 * f2 + lambda3 * f3
                
                results.append((i, j, weight))
            return results
        
        # 生成任务批次
        batch_size = 1000
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
            
            # 运行Leiden算法
            partition = leidenalg.find_partition(
                g, 
                leidenalg.CPMVertexPartition, 
                resolution_parameter=self.resolution,
                weights="weight"
            )
            
            # 获取分区标签
            labels = np.zeros(self.num_vertices, dtype=int)
            for node, community in enumerate(partition.membership):
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
        
        # 对边界进行平滑处理
        # 简化实现：使用移动平均
        smoothed_labels = labels.copy()
        for i in boundary_vertices:
            neighbors = self.mesh.adjacency[i]
            if neighbors:
                # 计算邻居的标签分布
                label_counts = {}
                for j in neighbors:
                    label_counts[labels[j]] = label_counts.get(labels[j], 0) + 1
                
                # 选择最常见的标签
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
        
        # 3. 拟合分区边界
        labels = self._fit_partition_boundaries(labels)
        
        # 4. 确保分区连通性
        labels = self._ensure_connectivity(labels)
        
        # 5. 重新编号标签为连续的整数
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
        
        # 6. 提取中点边缘
        edge_midpoints = self._extract_edge_midpoints(labels)
        
        print(f"分区完成: {len(unique_labels)} 个分区")
        
        return labels, edge_midpoints

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
