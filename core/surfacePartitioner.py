"""
基于多层级几何特征的表面分区器
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

from .meshProcessor import MeshProcessor



class SurfacePartitioner:
    def __init__(self, mesh: MeshProcessor, resolution=0.1):
        """
        初始化表面分区器
        Args:
            mesh: 网格处理器
            resolution: 聚类分辨率参数，控制分区数量（值越大分区越多，值越小分区越少）
        """
        self.mesh = mesh
        self.num_vertices = len(mesh.vertices)
        self.resolution = resolution
        
        # 预计算几何特征
        self.vertex_features = self._precompute_vertex_features()

    def _precompute_vertex_features(self) -> Dict[int, Dict[str, float]]:
        """
        预计算顶点的几何特征
        Returns:
            顶点特征字典
        """
        print("预计算顶点几何特征...")
        
        features = {}
        
        # 并行计算特征
        def compute_feature(vertex_idx):
            # 计算曲率
            curvature = self.mesh.curvatures[vertex_idx]
            
            # 计算法向量特征（使用完整的法向量，而不是仅Z分量）
            normal = self.mesh.vertex_normals[vertex_idx]
            # 计算法向量的方向特征（使用球面坐标角度）
            theta = np.arctan2(normal[1], normal[0])  # 方位角
            phi = np.arccos(np.clip(normal[2], -1.0, 1.0))  # 极角
            
            # 计算边缘连续性指标
            edge_continuity = self._calculate_edge_continuity(vertex_idx)
            
            return vertex_idx, {
                'curvature': curvature,
                'normal_theta': theta,
                'normal_phi': phi,
                'edge_continuity': edge_continuity
            }
        
        # 使用线程池并行计算
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(compute_feature, i): i for i in range(self.num_vertices)}
            for future in as_completed(futures):
                vertex_idx, feature = future.result()
                features[vertex_idx] = feature
        
        print("顶点几何特征预计算完成")
        return features

    def _calculate_edge_continuity(self, vertex_idx: int) -> float:
        """
        计算顶点的边缘连续性指标
        Args:
            vertex_idx: 顶点索引
        Returns:
            边缘连续性指标，值越大表示连续性越好
        """
        neighbors = self.mesh.adjacency[vertex_idx]
        if len(neighbors) < 2:
            return 0.0
        
        # 计算邻居顶点的法向量一致性
        normal = self.mesh.vertex_normals[vertex_idx]
        neighbor_normals = self.mesh.vertex_normals[neighbors]
        dot_products = np.dot(neighbor_normals, normal)
        avg_dot_product = np.mean(dot_products)
        
        # 计算邻居顶点的曲率一致性
        curvature = self.mesh.curvatures[vertex_idx]
        neighbor_curvatures = self.mesh.curvatures[neighbors]
        curvature_diff = np.abs(neighbor_curvatures - curvature)
        avg_curvature_consistency = 1.0 / (1.0 + np.mean(curvature_diff))
        
        # 综合计算连续性指标
        continuity = 0.6 * avg_dot_product + 0.4 * avg_curvature_consistency
        return continuity

    def _compute_similarity(self, i: int, j: int) -> Dict[str, float]:
        """
        计算两个顶点之间的相似性
        Args:
            i: 顶点1索引
            j: 顶点2索引
        Returns:
            相似性指标字典
        """
        # 几何特征相似性
        feat_i = self.vertex_features[i]
        feat_j = self.vertex_features[j]
        
        # 曲率相似性
        curvature_diff = abs(feat_i['curvature'] - feat_j['curvature'])
        curvature_similarity = np.exp(-curvature_diff)
        
        # 法向量相似性（使用完整的法向量方向）
        theta_diff = min(abs(feat_i['normal_theta'] - feat_j['normal_theta']), 2*np.pi - abs(feat_i['normal_theta'] - feat_j['normal_theta']))
        phi_diff = abs(feat_i['normal_phi'] - feat_j['normal_phi'])
        normal_similarity = np.exp(-(theta_diff + phi_diff))
        
        # 边缘连续性相似性
        continuity_diff = abs(feat_i['edge_continuity'] - feat_j['edge_continuity'])
        continuity_similarity = np.exp(-continuity_diff)
        
        return {
            'curvature': curvature_similarity,
            'normal': normal_similarity,
            'continuity': continuity_similarity
        }

    def _build_similarity_matrix(self) -> np.ndarray:
        """
        构建相似性矩阵
        Returns:
            相似性矩阵
        """
        print("构建相似性矩阵...")
        
        n = self.num_vertices
        similarity_matrix = np.zeros((n, n))
        
        # 并行计算相似性
        def compute_similarity_batch(batch):
            results = []
            for i, j in batch:
                sim = self._compute_similarity(i, j)
                # 加权组合相似性 - 增加边缘连续性的权重
                weight = np.array([0.3, 0.3, 0.4])  # 调整权重，连续性权重最高
                sim_values = np.array([sim['curvature'], sim['normal'], sim['continuity']])
                combined_sim = np.dot(weight, sim_values)
                results.append((i, j, combined_sim))
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
            futures = {executor.submit(compute_similarity_batch, batch): batch for batch in batches}
            for future in as_completed(futures):
                results = future.result()
                for i, j, sim in results:
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # 归一化矩阵
        max_sim = np.max(similarity_matrix)
        if max_sim > 0:
            similarity_matrix /= max_sim
        
        print("相似性矩阵构建完成")
        return similarity_matrix

    def _adaptive_partition(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        自适应分区算法
        Args:
            similarity_matrix: 相似性矩阵
        Returns:
            分区标签数组
        """
        print("执行自适应分区...")
        
        n = self.num_vertices
        labels = np.zeros(n, dtype=int)
        label_counter = 0
        
        # 计算每个顶点的复杂度
        complexity = np.zeros(n)
        for i in range(n):
            # 复杂度 = 曲率 + 法向量变化率 + 边缘连续性
            feat = self.vertex_features[i]
            complexity[i] = feat['curvature'] + abs(feat['normal_theta']) + abs(feat['normal_phi']) + (1.0 - feat['edge_continuity'])
        
        # 按复杂度排序，从复杂区域开始分区
        sorted_vertices = np.argsort(-complexity)
        
        # 分区大小限制
        min_partition_size = 100
        max_partition_size = 400  # 增加最大分区大小
        
        visited = np.zeros(n, dtype=bool)
        
        for seed in sorted_vertices:
            if not visited[seed]:
                # 扩展分区
                partition = self._expand_partition(seed, similarity_matrix, visited, min_partition_size, max_partition_size)
                
                # 分配标签
                labels[partition] = label_counter
                label_counter += 1
                
                print(f"创建分区 {label_counter}: {len(partition)} 个顶点")
        
        print(f"分区完成: {label_counter} 个分区")
        return labels

    def _expand_partition(self, seed: int, similarity_matrix: np.ndarray, visited: np.ndarray, min_size: int, max_size: int) -> List[int]:
        """
        从种子顶点扩展分区
        Args:
            seed: 种子顶点
            similarity_matrix: 相似性矩阵
            visited: 访问标记
            min_size: 最小分区大小
            max_size: 最大分区大小
        Returns:
            分区顶点列表
        """
        partition = []
        queue = [(seed, 1.0)]  # (顶点, 优先级)
        visited[seed] = True
        
        while queue and len(partition) < max_size:
            # 按优先级排序
            queue.sort(key=lambda x: x[1], reverse=True)
            current, priority = queue.pop(0)
            partition.append(current)
            
            # 扩展邻居
            neighbors = self.mesh.adjacency[current]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    # 计算邻居的优先级
                    sim = similarity_matrix[current, neighbor]
                    # 考虑几何特征的优先级
                    feat = self.vertex_features[neighbor]
                    geo_priority = 1.0 + feat['curvature']
                    total_priority = sim * geo_priority
                    
                    queue.append((neighbor, total_priority))
                    visited[neighbor] = True
        
        # 确保最小分区大小
        if len(partition) < min_size and len(queue) > 0:
            # 从队列中添加更多顶点
            queue.sort(key=lambda x: x[1], reverse=True)
            while len(partition) < min_size and queue:
                current, _ = queue.pop(0)
                partition.append(current)
        
        return partition

    def _optimize_partitions(self, labels: np.ndarray) -> np.ndarray:
        """
        优化分区，确保分区质量
        Args:
            labels: 初始分区标签
        Returns:
            优化后的分区标签
        """
        print("优化分区...")
        
        n = self.num_vertices
        unique_labels = np.unique(labels)
        
        # 计算每个分区的统计信息
        partition_stats = {}
        for label in unique_labels:
            vertices = np.where(labels == label)[0]
            if len(vertices) > 0:
                # 计算分区的平均几何特征
                avg_curvature = np.mean([self.vertex_features[v]['curvature'] for v in vertices])
                avg_normal_theta = np.mean([self.vertex_features[v]['normal_theta'] for v in vertices])
                avg_normal_phi = np.mean([self.vertex_features[v]['normal_phi'] for v in vertices])
                avg_edge_continuity = np.mean([self.vertex_features[v]['edge_continuity'] for v in vertices])
                
                partition_stats[label] = {
                    'size': len(vertices),
                    'curvature': avg_curvature,
                    'normal_theta': avg_normal_theta,
                    'normal_phi': avg_normal_phi,
                    'edge_continuity': avg_edge_continuity
                }
        
        # 调整边界顶点
        for i in range(n):
            neighbors = self.mesh.adjacency[i]
            neighbor_labels = [labels[j] for j in neighbors if j != i]
            
            if neighbor_labels:
                # 计算邻居标签的分布
                label_counts = {}
                for lbl in neighbor_labels:
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
                
                # 找到最常见的邻居标签
                most_common = max(label_counts, key=label_counts.get)
                
                # 如果当前顶点的标签与大多数邻居不同，考虑调整
                current_label = labels[i]
                if most_common != current_label:
                    # 计算当前顶点与两个分区的相似度
                    current_stats = partition_stats[current_label]
                    target_stats = partition_stats[most_common]
                    
                    # 计算相似度
                    current_sim = self._calculate_vertex_partition_similarity(i, current_stats)
                    target_sim = self._calculate_vertex_partition_similarity(i, target_stats)
                    
                    # 如果与目标分区更相似，调整标签
                    if target_sim > current_sim:
                        labels[i] = most_common
        
        print("分区优化完成")
        return labels

    def _calculate_vertex_partition_similarity(self, vertex_idx: int, partition_stats: Dict[str, float]) -> float:
        """
        计算顶点与分区的相似度
        Args:
            vertex_idx: 顶点索引
            partition_stats: 分区统计信息
        Returns:
            相似度
        """
        feat = self.vertex_features[vertex_idx]
        
        # 计算特征差异
        curvature_diff = abs(feat['curvature'] - partition_stats['curvature'])
        normal_theta_diff = abs(feat['normal_theta'] - partition_stats['normal_theta'])
        normal_phi_diff = abs(feat['normal_phi'] - partition_stats['normal_phi'])
        continuity_diff = abs(feat['edge_continuity'] - partition_stats['edge_continuity'])
        
        # 计算相似度
        similarity = np.exp(-(curvature_diff + normal_theta_diff + normal_phi_diff + continuity_diff))
        return similarity

    def _ensure_connectivity(self, labels: np.ndarray) -> np.ndarray:
        """
        确保每个分区都是连通的
        Args:
            labels: 分区标签
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

    def extract_edge_midpoints(self, labels: np.ndarray) -> np.ndarray:
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

    def partition_surface(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行表面分区
        Returns:
            分区标签数组和中点边缘点数组
        """
        print("开始表面分区...")
        
        # 1. 构建相似性矩阵
        similarity_matrix = self._build_similarity_matrix()
        
        # 2. 执行自适应分区
        labels = self._adaptive_partition(similarity_matrix)
        
        # 3. 优化分区
        labels = self._optimize_partitions(labels)
        
        # 4. 确保分区连通性
        labels = self._ensure_connectivity(labels)
        
        # 5. 重新编号标签为连续的整数
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
        
        # 6. 提取中点边缘
        edge_midpoints = self.extract_edge_midpoints(labels)
        
        print(f"分区完成: {len(unique_labels)} 个分区")
        
        return labels, edge_midpoints
