"""
基于新指标的高级表面分区器
实现算法version2中描述的分区算法
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import cKDTree
import time

from .meshProcessor import MeshProcessor
from .indicatorCalculator import IndicatorCalculator


class AdvancedSurfacePartitioner:
    def __init__(self, mesh: MeshProcessor, tool, resolution=0.1, alpha=0.3, global_field='rolled_error', symmetry_types=None):
        """
        初始化高级表面分区器
        Args:
            mesh: 网格处理器
            tool: 刀具对象
            resolution: 聚类分辨率参数，控制分区数量
            alpha: 全局引导强度参数，范围[0,1]
            global_field: 全局场类型，可选值：'rolled_error', 'curvature', 'cutting_width'
            symmetry_types: 对称性类型列表，可选值：'rotation', 'translation', 'reflection', 'helical', 'combined'
        """
        self.mesh = mesh
        self.tool = tool
        self.num_vertices = len(mesh.vertices)
        self.resolution = resolution
        self.alpha = alpha
        self.global_field = global_field
        self.symmetry_types = symmetry_types
        
        # 初始化指标计算器
        self.indicator_calculator = IndicatorCalculator(mesh, tool)
        
        # 预计算几何特征
        self._precompute_features()
        
        # 预计算全局场值
        self._precompute_global_field()

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
    
    def _precompute_global_field(self):
        """
        预计算全局场值
        """
        print("预计算全局场值...")
        
        # 根据选择的全局场类型计算场值
        if self.global_field == 'rolled_error':
            # 使用直纹面逼近误差作为全局场
            if hasattr(self.mesh, 'rolled_error'):
                self.global_field_values = self.mesh.rolled_error
            else:
                # 如果没有rolled_error属性，计算它
                self.mesh.calculate_rolled_error()
                self.global_field_values = self.mesh.rolled_error
        elif self.global_field == 'curvature':
            # 使用曲率作为全局场
            if hasattr(self.mesh, 'curvatures'):
                self.global_field_values = self.mesh.curvatures
            else:
                # 如果没有curvatures属性，计算它
                self.mesh._estimate_curvatures()
                self.global_field_values = self.mesh.curvatures
        elif self.global_field == 'cutting_width':
            # 使用切削宽度作为全局场
            if hasattr(self.mesh, 'max_cutting_widths'):
                self.global_field_values = self.mesh.max_cutting_widths
            else:
                # 如果没有max_cutting_widths属性，计算它
                self.mesh.calculate_max_cutting_width(self.tool)
                self.global_field_values = self.mesh.max_cutting_widths
        else:
            # 默认使用直纹面逼近误差
            if hasattr(self.mesh, 'rolled_error'):
                self.global_field_values = self.mesh.rolled_error
            else:
                self.mesh.calculate_rolled_error()
                self.global_field_values = self.mesh.rolled_error
        
        # 计算全局场的标准差，用于带宽参数
        self.sigma_f = np.std(self.global_field_values) if np.std(self.global_field_values) > 0 else 1.0
        
        print("全局场值预计算完成")
    
    def calculate_normal_variation(self):
        """
        计算相邻顶点间的法向量变化率
        Returns:
            法向量变化率数组
        """
        print("计算法向量变化率...")
        
        # 检查法线是否存在
        if not hasattr(self.mesh, 'vertex_normals') or len(self.mesh.vertex_normals) != self.num_vertices:
            print("警告: 法向量不存在，无法计算法向量变化率")
            return np.zeros(self.num_vertices)
        
        # 计算每个顶点的法向量变化率
        normal_variation = np.zeros(self.num_vertices)
        
        for i in range(self.num_vertices):
            neighbors = self.mesh.adjacency[i]
            if not neighbors:
                continue
            
            # 获取当前顶点的法向量
            current_normal = self.mesh.vertex_normals[i]
            
            # 计算与邻居顶点的法向量夹角
            total_angle = 0.0
            for neighbor in neighbors:
                neighbor_normal = self.mesh.vertex_normals[neighbor]
                # 计算夹角
                dot_product = np.dot(current_normal, neighbor_normal)
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = np.arccos(dot_product)
                total_angle += angle
            
            # 计算平均变化率
            normal_variation[i] = total_angle / len(neighbors)
        
        print("法向量变化率计算完成")
        return normal_variation
    
    def detect_c1_discontinuities(self, threshold=0.1):
        """
        检测C1不连续边界
        Args:
            threshold: 法向量变化率阈值
        Returns:
            不连续边界边列表
        """
        print("检测C1不连续边界...")
        
        # 计算法向量变化率
        normal_variation = self.calculate_normal_variation()
        
        # 检测不连续边界
        discontinuity_edges = []
        
        for i in range(self.num_vertices):
            neighbors = self.mesh.adjacency[i]
            for neighbor in neighbors:
                if i < neighbor:  # 避免重复
                    # 检查两个顶点的法向量变化率
                    if normal_variation[i] > threshold or normal_variation[neighbor] > threshold:
                        # 计算两个顶点法向量的夹角
                        dot_product = np.dot(self.mesh.vertex_normals[i], self.mesh.vertex_normals[neighbor])
                        dot_product = np.clip(dot_product, -1.0, 1.0)
                        angle = np.arccos(dot_product)
                        
                        if angle > threshold:
                            discontinuity_edges.append((i, neighbor))
        
        print(f"检测到 {len(discontinuity_edges)} 条C1不连续边界")
        return discontinuity_edges
    
    def partition_by_c1_continuity(self, threshold=0.1):
        """
        基于C1连续性进行分区
        Args:
            threshold: 法向量变化率阈值
        Returns:
            分区标签数组
        """
        print("基于C1连续性进行分区...")
        
        # 检测C1不连续边界
        discontinuity_edges = self.detect_c1_discontinuities(threshold)
        
        # 构建图，不连续边界作为割边
        G = nx.Graph()
        G.add_nodes_from(range(self.num_vertices))
        
        # 添加所有边，除了不连续边界
        for i in range(self.num_vertices):
            neighbors = self.mesh.adjacency[i]
            for neighbor in neighbors:
                if (i, neighbor) not in discontinuity_edges and (neighbor, i) not in discontinuity_edges:
                    G.add_edge(i, neighbor)
        
        # 提取连通分量作为分区
        components = list(nx.connected_components(G))
        
        # 分配分区标签
        labels = np.zeros(self.num_vertices, dtype=int)
        for idx, component in enumerate(components):
            for vertex in component:
                labels[vertex] = idx
        
        print(f"C1连续性分区完成: {len(components)} 个分区")
        return labels

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
        # 首先尝试使用Dijkstra算法计算测地距离
        try:
            # 构建边权重图
            G = nx.Graph()
            for v in range(self.num_vertices):
                for neighbor in self.mesh.adjacency[v]:
                    weight = np.linalg.norm(self.mesh.vertices[v] - self.mesh.vertices[neighbor])
                    G.add_edge(v, neighbor, weight=weight)
            
            # 计算最短路径距离
            distance = nx.shortest_path_length(G, source=i, target=j, weight='weight')
            return distance
        except:
            # 如果失败，使用欧氏距离作为近似
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
                if distance > 0.5:  # 距离阈值，可根据实际情况调整
                    continue
                
                # 计算综合相似性（局部相似性）
                local_weight = self.indicator_calculator.calculate_combined_similarity(i, j)
                
                # 计算全局引导相似性
                fi = self.global_field_values[i]
                fj = self.global_field_values[j]
                global_similarity = np.exp(-(fi - fj)**2 / (2 * self.sigma_f**2))
                
                # 组合权重
                weight = (1 - self.alpha) * local_weight + self.alpha * global_similarity
                
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
            
            # 自适应分辨率参数
            # 根据图的大小和边权重分布自动调整resolution
            edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
            avg_weight = np.mean(edge_weights) if edge_weights else 0.5
            
            # 基于图大小和平均权重计算自适应分辨率
            adaptive_resolution = self.resolution * (1 + 0.5 * (1 - avg_weight))
            
            # 多分辨率聚类
            # 先使用较低分辨率获取初始分区
            initial_partition = leidenalg.find_partition(
                g, 
                leidenalg.CPMVertexPartition, 
                resolution_parameter=adaptive_resolution * 0.8,
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
    
    def _spectral_clustering(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        使用谱聚类算法进行分区
        Args:
            adjacency_matrix: 加权邻接矩阵
        Returns:
            分区标签数组
        """
        print("执行谱聚类...")
        
        # 1. 构建相似性矩阵 S
        # 假设adjacency_matrix已经是对称非负的相似性矩阵
        S = adjacency_matrix
        
        # 2. 构建度矩阵 D
        D = np.diag(np.sum(S, axis=1))
        
        # 3. 计算归一化拉普拉斯矩阵 L
        # 使用对称归一化形式：L=I-D^{-1/2}SD^{-1/2}
        D_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        D_sqrt[np.isinf(D_sqrt)] = 0  # 处理零度顶点
        L = np.eye(self.num_vertices) - D_sqrt @ S @ D_sqrt
        
        # 4. 特征分解
        # 计算前k个最小特征值对应的特征向量
        # 自动估计k值
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        # 选择k值：观察特征值间隙
        k = self._estimate_k(eigenvalues)
        print(f"估计的分区数 k: {k}")
        
        # 5. 聚类特征向量
        # 取前k个最小特征值对应的特征向量（跳过第一个特征值为0的平凡解）
        if k > 1:
            X = eigenvectors[:, :k]
        else:
            X = eigenvectors[:, [1]]  # 至少取一个特征向量
        
        # 6. 使用k-means对特征向量进行聚类
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        
        print(f"谱聚类完成: {len(np.unique(labels))} 个分区")
        return labels
    
    def _estimate_k(self, eigenvalues: np.ndarray) -> int:
        """
        根据特征值间隙估计k值
        Args:
            eigenvalues: 拉普拉斯矩阵的特征值
        Returns:
            估计的分区数k
        """
        # 排序特征值
        sorted_eigenvalues = np.sort(eigenvalues)
        
        # 计算特征值间隙
        gaps = []
        for i in range(1, len(sorted_eigenvalues)):
            gap = sorted_eigenvalues[i] - sorted_eigenvalues[i-1]
            gaps.append(gap)
        
        # 找到最大的间隙
        if gaps:
            max_gap_index = np.argmax(gaps)
            k = max_gap_index + 1
            # 限制k的范围在2-20之间
            k = max(2, min(20, k))
        else:
            k = 5  # 默认值
        
        return k



    def partition_surface(self, clustering_method='leiden') -> Tuple[np.ndarray, np.ndarray]:
        """
        执行表面分区
        Args:
            clustering_method: 聚类方法，可选值：'leiden', 'spectral', 'alternative'
        Returns:
            分区标签数组和中点边缘点数组
        """
        print("开始表面分区...")
        
        # 检查是否指定了对称性
        if self.symmetry_types is not None and len(self.symmetry_types) > 0:
            # 使用基于对称性的分区方法
            print("使用基于对称性的分区方法")
            
            # 检测对称性
            symmetries = self._detect_all_symmetries()
            
            if symmetries:
                # 基于对称性生成分区
                labels = self._partition_by_symmetry(symmetries)
            else:
                # 未检测到对称性，回退到原有方法
                print("未检测到对称性，回退到原有分区方法")
                # 1. 构建加权邻接矩阵
                adjacency_matrix = self._build_weighted_adjacency_matrix()
                
                # 2. 执行聚类
                if clustering_method == 'spectral':
                    # 执行谱聚类
                    labels = self._spectral_clustering(adjacency_matrix)
                elif clustering_method == 'alternative':
                    # 执行备用聚类
                    G = nx.Graph()
                    G.add_nodes_from(range(self.num_vertices))
                    for i in range(self.num_vertices):
                        for j in range(i+1, self.num_vertices):
                            if adjacency_matrix[i, j] > 0:
                                G.add_edge(i, j, weight=adjacency_matrix[i, j])
                    labels = self._alternative_clustering(G)
                else:
                    # 执行Leiden聚类
                    labels = self._leiden_clustering(adjacency_matrix)
                
                # 3. 应用对称性约束
                labels = self._apply_symmetry_constraints(labels)
                
                # 4. 确保分区连通性
                labels = self._ensure_connectivity(labels)
                
                # 5. 重新编号标签为连续的整数
                unique_labels = np.unique(labels)
                label_map = {old: new for new, old in enumerate(unique_labels)}
                labels = np.array([label_map[l] for l in labels])
        else:
            # 使用原有分区方法
            print("使用原有分区方法")
            # 1. 构建加权邻接矩阵
            adjacency_matrix = self._build_weighted_adjacency_matrix()
            
            # 2. 执行聚类
            if clustering_method == 'spectral':
                # 执行谱聚类
                labels = self._spectral_clustering(adjacency_matrix)
            elif clustering_method == 'alternative':
                # 执行备用聚类
                G = nx.Graph()
                G.add_nodes_from(range(self.num_vertices))
                for i in range(self.num_vertices):
                    for j in range(i+1, self.num_vertices):
                        if adjacency_matrix[i, j] > 0:
                            G.add_edge(i, j, weight=adjacency_matrix[i, j])
                labels = self._alternative_clustering(G)
            else:
                # 执行Leiden聚类
                labels = self._leiden_clustering(adjacency_matrix)
            
            # 3. 应用对称性约束
            labels = self._apply_symmetry_constraints(labels)
            
            # 4. 确保分区连通性
            labels = self._ensure_connectivity(labels)
            
            # 5. 重新编号标签为连续的整数
            unique_labels = np.unique(labels)
            label_map = {old: new for new, old in enumerate(unique_labels)}
            labels = np.array([label_map[l] for l in labels])
        
        # 提取中点边缘
        edge_midpoints = self._extract_edge_midpoints(labels)
        
        print(f"分区完成: {len(np.unique(labels))} 个分区")
        
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
    
    def _detect_rotational_symmetry(self) -> Dict[str, Any]:
        """
        检测旋转对称性
        Returns:
            旋转对称信息字典
        """
        print("检测旋转对称性...")
        start_time = time.time()
        
        vertices = self.mesh.vertices
        n = len(vertices)
        
        print(f"  顶点数量: {n}")
        
        # 构建KD树加速最近邻搜索
        print("  构建KD树...")
        kdtree = cKDTree(vertices)
        
        # 计算平均边长
        avg_edge_length = self._calculate_avg_edge_length()
        epsilon = 1.5 * avg_edge_length
        print(f"  平均边长: {avg_edge_length:.4f}, epsilon: {epsilon:.4f}")
        
        best_axis = None
        best_center = None
        best_n = 0
        best_inliers = []
        best_score = 0
        
        # 生成候选轴
        print("  生成候选轴...")
        candidate_axes = []
        
        # 方法1: 使用PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca.fit(vertices)
        candidate_axes.append(pca.components_[-1])  # 最小特征值对应的轴
        print(f"  PCA轴: {pca.components_[-1]}")
        
        # 方法2: 随机采样点对 (减少采样数量)
        Naxis = min(50, n)  # 减少候选轴数量
        for _ in range(Naxis):
            if n >= 2:
                i, j = np.random.choice(n, 2, replace=False)
                axis = vertices[j] - vertices[i]
                if np.linalg.norm(axis) > 0:
                    axis = axis / np.linalg.norm(axis)
                    candidate_axes.append(axis)
        
        print(f"  候选轴数量: {len(candidate_axes)}")
        
        # RANSAC主循环 - 减少迭代次数
        max_iterations = 50  # 减少迭代次数
        total_iterations = len(candidate_axes) * max_iterations
        current_iteration = 0
        
        print(f"  开始RANSAC迭代 (总迭代次数: {total_iterations})...")
        
        for axis_idx, axis in enumerate(candidate_axes):
            if axis_idx % 10 == 0:
                print(f"  处理候选轴 {axis_idx + 1}/{len(candidate_axes)}...")
            
            for iter_idx in range(max_iterations):
                current_iteration += 1
                if current_iteration % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"  进度: {current_iteration}/{total_iterations} ({elapsed:.1f}秒)")
                
                if n >= 2:
                    # 随机选择两个点
                    i, j = np.random.choice(n, 2, replace=False)
                    p, q = vertices[i], vertices[j]
                    
                    # 计算旋转角度
                    # 投影到垂直于轴的平面
                    def project(v):
                        return v - np.dot(v, axis) * axis
                    
                    p_proj = project(p)
                    q_proj = project(q)
                    
                    if np.linalg.norm(p_proj) > 1e-8 and np.linalg.norm(q_proj) > 1e-8:
                        # 计算极角
                        angle_p = np.arctan2(p_proj[1], p_proj[0])
                        angle_q = np.arctan2(q_proj[1], q_proj[0])
                        phi = abs(angle_q - angle_p)
                        if phi > np.pi:
                            phi = 2 * np.pi - phi
                        
                        # 估计阶数n
                        if phi > 1e-8:
                            estimated_n = round(2 * np.pi / phi)
                            estimated_n = max(2, min(12, estimated_n))  # 限制阶数范围
                            phi = 2 * np.pi / estimated_n
                        else:
                            continue
                        
                        # 批量旋转所有点并使用KD树查找最近邻
                        inliers = []
                        rotated_points = np.array([self._rotate_point(v, axis, phi) for v in vertices])
                        
                        # 使用KD树批量查询最近邻
                        distances, indices = kdtree.query(rotated_points, k=1)
                        inliers = np.where(distances < epsilon)[0].tolist()
                        
                        # 计算分数
                        score = len(inliers) / n
                        if score > best_score:
                            best_score = score
                            best_axis = axis
                            best_n = estimated_n
                            best_inliers = inliers
        
        elapsed = time.time() - start_time
        print(f"  RANSAC完成，耗时: {elapsed:.2f}秒")
        
        if best_score > 0.6:  # 内点比例阈值
            print(f"检测到旋转对称性: 轴={best_axis}, 阶数={best_n}, 内点比例={best_score:.3f}")
            return {
                'type': 'rotation',
                'axis': best_axis,
                'center': np.mean(vertices[best_inliers], axis=0),
                'n': best_n,
                'inliers': best_inliers,
                'score': best_score
            }
        else:
            print("未检测到旋转对称性")
            return None
    
    def _detect_translational_symmetry(self) -> Dict[str, Any]:
        """
        检测平移对称性
        Returns:
            平移对称信息字典
        """
        print("检测平移对称性...")
        start_time = time.time()
        
        vertices = self.mesh.vertices
        n = len(vertices)
        
        print(f"  顶点数量: {n}")
        
        # 构建KD树加速最近邻搜索
        print("  构建KD树...")
        kdtree = cKDTree(vertices)
        
        # 计算平均边长
        avg_edge_length = self._calculate_avg_edge_length()
        epsilon = 1.5 * avg_edge_length
        print(f"  平均边长: {avg_edge_length:.4f}, epsilon: {epsilon:.4f}")
        
        best_translation = None
        best_inliers = []
        best_score = 0
        
        # 生成候选平移向量 (减少采样数量)
        print("  生成候选平移向量...")
        candidate_translations = []
        
        # 采样点对
        Nsamples = min(100, n)  # 减少采样数量
        for _ in range(Nsamples):
            if n >= 2:
                i, j = np.random.choice(n, 2, replace=False)
                t = vertices[j] - vertices[i]
                if np.linalg.norm(t) > avg_edge_length:  # 过滤过小的向量
                    candidate_translations.append(t)
        
        print(f"  候选平移向量数量: {len(candidate_translations)}")
        
        # RANSAC主循环
        print(f"  开始RANSAC迭代...")
        for idx, t in enumerate(candidate_translations):
            if idx % 20 == 0:
                print(f"  处理候选向量 {idx + 1}/{len(candidate_translations)}...")
            
            # 批量平移所有点并使用KD树查找最近邻
            translated_points = vertices + t
            distances, indices = kdtree.query(translated_points, k=1)
            inliers = np.where(distances < epsilon)[0].tolist()
            
            score = len(inliers) / n
            if score > best_score:
                best_score = score
                best_translation = t
                best_inliers = inliers
        
        elapsed = time.time() - start_time
        print(f"  RANSAC完成，耗时: {elapsed:.2f}秒")
        
        if best_score > 0.6:  # 内点比例阈值
            print(f"检测到平移对称性: 向量={best_translation}, 内点比例={best_score:.3f}")
            return {
                'type': 'translation',
                'vector': best_translation,
                'inliers': best_inliers,
                'score': best_score
            }
        else:
            print("未检测到平移对称性")
            return None
    
    def _detect_reflection_symmetry(self) -> Dict[str, Any]:
        """
        检测反射对称性
        Returns:
            反射对称信息字典
        """
        print("检测反射对称性...")
        start_time = time.time()
        
        vertices = self.mesh.vertices
        n = len(vertices)
        
        print(f"  顶点数量: {n}")
        
        # 构建KD树加速最近邻搜索
        print("  构建KD树...")
        kdtree = cKDTree(vertices)
        
        # 计算平均边长
        avg_edge_length = self._calculate_avg_edge_length()
        epsilon = 1.5 * avg_edge_length
        print(f"  平均边长: {avg_edge_length:.4f}, epsilon: {epsilon:.4f}")
        
        best_normal = None
        best_d = 0
        best_inliers = []
        best_score = 0
        
        # 生成候选平面
        print("  生成候选平面法向...")
        candidate_normals = []
        
        # 方法1: 使用PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca.fit(vertices)
        candidate_normals.append(pca.components_[-1])  # 最小特征值对应的法向
        print(f"  PCA法向: {pca.components_[-1]}")
        
        # 方法2: 随机采样三个点 (减少采样数量)
        Nsamples = min(100, n)  # 减少采样数量
        for _ in range(Nsamples):
            if n >= 3:
                indices = np.random.choice(n, 3, replace=False)
                p1, p2, p3 = vertices[indices]
                # 计算平面法向
                v1 = p2 - p1
                v2 = p3 - p1
                normal = np.cross(v1, v2)
                if np.linalg.norm(normal) > 0:
                    normal = normal / np.linalg.norm(normal)
                    candidate_normals.append(normal)
        
        print(f"  候选法向数量: {len(candidate_normals)}")
        
        # RANSAC主循环
        print(f"  开始RANSAC迭代...")
        for idx, normal in enumerate(candidate_normals):
            if idx % 20 == 0:
                print(f"  处理候选法向 {idx + 1}/{len(candidate_normals)}...")
            
            # 计算平面偏移d（使用点云中位数）
            projections = np.dot(vertices, normal)
            d = np.median(projections)
            
            # 批量计算反射点并使用KD树查找最近邻
            reflected_points = vertices - 2 * (projections - d)[:, np.newaxis] * normal
            distances, indices = kdtree.query(reflected_points, k=1)
            inliers = np.where(distances < epsilon)[0].tolist()
            
            score = len(inliers) / n
            if score > best_score:
                best_score = score
                best_normal = normal
                best_d = d
                best_inliers = inliers
        
        elapsed = time.time() - start_time
        print(f"  RANSAC完成，耗时: {elapsed:.2f}秒")
        
        if best_score > 0.6:  # 内点比例阈值
            print(f"检测到反射对称性: 法向={best_normal}, 偏移={best_d}, 内点比例={best_score:.3f}")
            return {
                'type': 'reflection',
                'normal': best_normal,
                'd': best_d,
                'inliers': best_inliers,
                'score': best_score
            }
        else:
            print("未检测到反射对称性")
            return None
    
    def _detect_helical_symmetry(self) -> Dict[str, Any]:
        """
        检测螺旋对称性
        Returns:
            螺旋对称信息字典
        """
        print("检测螺旋对称性...")
        start_time = time.time()
        
        # 首先检测旋转轴
        rotation_info = self._detect_rotational_symmetry()
        if not rotation_info:
            print("未检测到螺旋对称性（无旋转轴）")
            return None
        
        axis = rotation_info['axis']
        center = rotation_info['center']
        vertices = self.mesh.vertices
        n = len(vertices)
        
        print(f"  顶点数量: {n}")
        
        # 构建KD树加速最近邻搜索
        print("  构建KD树...")
        kdtree = cKDTree(vertices)
        
        # 沿轴分析
        print("  计算投影和角度...")
        # 投影到轴上
        projections = np.dot(vertices - center, axis)
        
        # 计算绕轴的角度
        angles = []
        for v in vertices:
            v_rel = v - center
            # 投影到垂直于轴的平面
            v_proj = v_rel - np.dot(v_rel, axis) * axis
            if np.linalg.norm(v_proj) > 1e-8:
                angle = np.arctan2(v_proj[1], v_proj[0])
                angles.append(angle)
            else:
                angles.append(0)
        
        # 寻找线性关系 angle = k * t + angle0
        # 使用最小二乘法
        from sklearn.linear_model import LinearRegression
        X = projections.reshape(-1, 1)
        y = np.array(angles)
        
        model = LinearRegression()
        model.fit(X, y)
        k = model.coef_[0]
        angle0 = model.intercept_
        
        print(f"  线性回归结果: k={k:.6f}, angle0={angle0:.3f}")
        
        # 计算平移量h和旋转角theta
        if abs(k) > 1e-8:
            h = 2 * np.pi / abs(k)
            theta = 2 * np.pi / rotation_info['n']
        else:
            print("未检测到螺旋对称性（无平移分量）")
            return None
        
        print(f"  计算参数: h={h:.3f}, theta={theta:.3f}")
        
        # 验证 - 使用KD树批量查询
        avg_edge_length = self._calculate_avg_edge_length()
        epsilon = 1.5 * avg_edge_length
        
        print("  验证螺旋对称性...")
        # 批量计算螺旋变换后的点
        helical_points = []
        for v in vertices:
            v_rotated = self._rotate_point(v - center, axis, theta) + center
            v_helical = v_rotated + h * axis
            helical_points.append(v_helical)
        helical_points = np.array(helical_points)
        
        # 使用KD树批量查询最近邻
        distances, indices = kdtree.query(helical_points, k=1)
        inliers = np.where(distances < epsilon)[0].tolist()
        
        score = len(inliers) / n
        
        elapsed = time.time() - start_time
        print(f"  螺旋对称性检测完成，耗时: {elapsed:.2f}秒")
        
        if score > 0.6:  # 内点比例阈值
            print(f"检测到螺旋对称性: 轴={axis}, 旋转角={theta:.3f}, 平移量={h:.3f}, 内点比例={score:.3f}")
            return {
                'type': 'helical',
                'axis': axis,
                'center': center,
                'theta': theta,
                'h': h,
                'inliers': inliers,
                'score': score
            }
        else:
            print("未检测到螺旋对称性")
            return None
    
    def _calculate_avg_edge_length(self) -> float:
        """
        计算网格的平均边长
        Returns:
            平均边长
        """
        total_length = 0
        edge_count = 0
        vertices = self.mesh.vertices
        
        for v in range(self.num_vertices):
            neighbors = self.mesh.adjacency[v]
            for neighbor in neighbors:
                if v < neighbor:  # 避免重复计算
                    length = np.linalg.norm(vertices[v] - vertices[neighbor])
                    total_length += length
                    edge_count += 1
        
        if edge_count > 0:
            return total_length / edge_count
        else:
            return 0.1  # 默认值
    
    def _rotate_point(self, point: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        绕轴旋转点
        Args:
            point: 点坐标
            axis: 旋转轴（单位向量）
            angle: 旋转角度（弧度）
        Returns:
            旋转后的点
        """
        # 使用罗德里格斯旋转公式
        axis = axis / np.linalg.norm(axis)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        # 计算旋转矩阵
        R = np.array([
            [cos_theta + axis[0]**2 * (1 - cos_theta),
             axis[0] * axis[1] * (1 - cos_theta) - axis[2] * sin_theta,
             axis[0] * axis[2] * (1 - cos_theta) + axis[1] * sin_theta],
            [axis[1] * axis[0] * (1 - cos_theta) + axis[2] * sin_theta,
             cos_theta + axis[1]**2 * (1 - cos_theta),
             axis[1] * axis[2] * (1 - cos_theta) - axis[0] * sin_theta],
            [axis[2] * axis[0] * (1 - cos_theta) - axis[1] * sin_theta,
             axis[2] * axis[1] * (1 - cos_theta) + axis[0] * sin_theta,
             cos_theta + axis[2]**2 * (1 - cos_theta)]
        ])
        
        return R @ point

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
    
    def _detect_all_symmetries(self) -> List[Dict[str, Any]]:
        """
        检测所有指定类型的对称性
        Returns:
            对称性信息列表
        """
        print("\n" + "="*60)
        print("检测所有指定类型的对称性...")
        print("="*60)
        start_time = time.time()
        
        symmetries = []
        
        if self.symmetry_types is None:
            print("未指定对称性类型，跳过检测")
            return symmetries
        
        print(f"指定的对称性类型: {self.symmetry_types}")
        
        # 检测旋转对称性
        if 'rotation' in self.symmetry_types:
            print("\n--- 检测旋转对称性 ---")
            rotation_info = self._detect_rotational_symmetry()
            if rotation_info:
                symmetries.append(rotation_info)
                print(f"  ✓ 检测到旋转对称性")
            else:
                print(f"  ✗ 未检测到旋转对称性")
        
        # 检测平移对称性
        if 'translation' in self.symmetry_types:
            print("\n--- 检测平移对称性 ---")
            translation_info = self._detect_translational_symmetry()
            if translation_info:
                symmetries.append(translation_info)
                print(f"  ✓ 检测到平移对称性")
            else:
                print(f"  ✗ 未检测到平移对称性")
        
        # 检测反射对称性
        if 'reflection' in self.symmetry_types:
            print("\n--- 检测反射对称性 ---")
            reflection_info = self._detect_reflection_symmetry()
            if reflection_info:
                symmetries.append(reflection_info)
                print(f"  ✓ 检测到反射对称性")
            else:
                print(f"  ✗ 未检测到反射对称性")
        
        # 检测螺旋对称性
        if 'helical' in self.symmetry_types:
            print("\n--- 检测螺旋对称性 ---")
            helical_info = self._detect_helical_symmetry()
            if helical_info:
                symmetries.append(helical_info)
                print(f"  ✓ 检测到螺旋对称性")
            else:
                print(f"  ✗ 未检测到螺旋对称性")
        
        # 检测组合对称性
        if 'combined' in self.symmetry_types:
            print("\n--- 检测组合对称性（检测所有类型）---")
            # 检测所有类型
            rotation_info = self._detect_rotational_symmetry()
            if rotation_info:
                symmetries.append(rotation_info)
                print(f"  ✓ 检测到旋转对称性")
            
            translation_info = self._detect_translational_symmetry()
            if translation_info:
                symmetries.append(translation_info)
                print(f"  ✓ 检测到平移对称性")
            
            reflection_info = self._detect_reflection_symmetry()
            if reflection_info:
                symmetries.append(reflection_info)
                print(f"  ✓ 检测到反射对称性")
            
            helical_info = self._detect_helical_symmetry()
            if helical_info:
                symmetries.append(helical_info)
                print(f"  ✓ 检测到螺旋对称性")
        
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print(f"对称性检测完成，找到 {len(symmetries)} 种对称性，总耗时: {elapsed:.2f}秒")
        print("="*60)
        return symmetries
    
    def _partition_by_symmetry(self, symmetries: List[Dict[str, Any]]) -> np.ndarray:
        """
        基于对称性生成分区
        Args:
            symmetries: 对称性信息列表
        Returns:
            分区标签数组
        """
        print("\n" + "="*60)
        print("基于对称性生成分区...")
        print("="*60)
        start_time = time.time()
        
        vertices = self.mesh.vertices
        n = len(vertices)
        
        print(f"顶点数量: {n}")
        print(f"对称性数量: {len(symmetries)}")
        
        # 构建KD树加速最近邻搜索
        print("构建KD树...")
        kdtree = cKDTree(vertices)
        
        # 构建对称对应图
        print("构建对称对应图...")
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        # 计算平均边长
        avg_edge_length = self._calculate_avg_edge_length()
        epsilon_match = 1.5 * avg_edge_length
        print(f"平均边长: {avg_edge_length:.4f}, epsilon_match: {epsilon_match:.4f}")
        
        # 为每个变换添加边
        for sym_idx, symmetry in enumerate(symmetries):
            symmetry_type = symmetry['type']
            print(f"\n处理对称性 {sym_idx + 1}/{len(symmetries)}: {symmetry_type}")
            sym_start_time = time.time()
            
            if symmetry_type == 'rotation':
                axis = symmetry['axis']
                center = symmetry['center']
                theta = 2 * np.pi / symmetry['n']
                
                print(f"  旋转轴: {axis}, 阶数: {symmetry['n']}, 角度: {theta:.3f}")
                
                # 批量旋转所有点
                rotated_points = np.array([self._rotate_point(v - center, axis, theta) + center for v in vertices])
                
                # 使用KD树批量查询最近邻
                distances, indices = kdtree.query(rotated_points, k=1)
                valid_pairs = np.where(distances < epsilon_match)[0]
                
                for i in valid_pairs:
                    G.add_edge(i, indices[i])
                
                print(f"  添加了 {len(valid_pairs)} 条边")
            
            elif symmetry_type == 'translation':
                t = symmetry['vector']
                
                print(f"  平移向量: {t}")
                
                # 批量平移所有点
                translated_points = vertices + t
                
                # 使用KD树批量查询最近邻
                distances, indices = kdtree.query(translated_points, k=1)
                valid_pairs = np.where(distances < epsilon_match)[0]
                
                for i in valid_pairs:
                    G.add_edge(i, indices[i])
                
                print(f"  添加了 {len(valid_pairs)} 条边")
            
            elif symmetry_type == 'reflection':
                normal = symmetry['normal']
                d = symmetry['d']
                
                print(f"  反射平面法向: {normal}, 偏移: {d:.3f}")
                
                # 批量计算反射点
                projections = np.dot(vertices, normal)
                reflected_points = vertices - 2 * (projections - d)[:, np.newaxis] * normal
                
                # 使用KD树批量查询最近邻
                distances, indices = kdtree.query(reflected_points, k=1)
                valid_pairs = np.where(distances < epsilon_match)[0]
                
                for i in valid_pairs:
                    G.add_edge(i, indices[i])
                
                print(f"  添加了 {len(valid_pairs)} 条边")
            
            elif symmetry_type == 'helical':
                axis = symmetry['axis']
                center = symmetry['center']
                theta = symmetry['theta']
                h = symmetry['h']
                
                print(f"  螺旋轴: {axis}, 旋转角: {theta:.3f}, 平移量: {h:.3f}")
                
                # 批量计算螺旋变换后的点
                helical_points = np.array([
                    self._rotate_point(v - center, axis, theta) + center + h * axis 
                    for v in vertices
                ])
                
                # 使用KD树批量查询最近邻
                distances, indices = kdtree.query(helical_points, k=1)
                valid_pairs = np.where(distances < epsilon_match)[0]
                
                for i in valid_pairs:
                    G.add_edge(i, indices[i])
                
                print(f"  添加了 {len(valid_pairs)} 条边")
            
            sym_elapsed = time.time() - sym_start_time
            print(f"  处理耗时: {sym_elapsed:.2f}秒")
        
        # 提取连通分量
        print("\n提取连通分量...")
        components = list(nx.connected_components(G))
        print(f"找到 {len(components)} 个连通分量")
        
        # 为每个分量分配标签
        labels = np.zeros(n, dtype=int)
        for idx, component in enumerate(components):
            for v in component:
                labels[v] = idx
        
        # 合并小分量
        print("合并小分量...")
        min_component_size = max(1, int(n * 0.005))  # 最小分量大小
        unique_labels = np.unique(labels)
        
        # 计算每个分量的几何特征
        component_features = {}
        for label in unique_labels:
            component_vertices = np.where(labels == label)[0]
            if len(component_vertices) > 0:
                # 计算高斯曲率平均值
                if hasattr(self.mesh, 'gaussian_curvatures'):
                    avg_curvature = np.mean(self.mesh.gaussian_curvatures[component_vertices])
                else:
                    avg_curvature = 0
                
                # 计算平均位置
                avg_position = np.mean(vertices[component_vertices], axis=0)
                
                component_features[label] = {
                    'size': len(component_vertices),
                    'avg_curvature': avg_curvature,
                    'avg_position': avg_position
                }
        
        # 合并小分量
        merged_count = 0
        for label in unique_labels:
            if component_features[label]['size'] < min_component_size:
                # 找到几何特征最相似的相邻分量
                min_distance = float('inf')
                nearest_label = -1
                
                # 找到该分量的所有相邻顶点
                component_vertices = np.where(labels == label)[0]
                neighbor_labels = set()
                
                for v in component_vertices:
                    for neighbor in self.mesh.adjacency[v]:
                        if labels[neighbor] != label:
                            neighbor_labels.add(labels[neighbor])
                
                # 计算与相邻分量的几何相似性
                for neighbor_label in neighbor_labels:
                    if neighbor_label in component_features:
                        # 计算曲率差异
                        curvature_diff = abs(component_features[label]['avg_curvature'] - 
                                           component_features[neighbor_label]['avg_curvature'])
                        # 计算位置距离
                        position_diff = np.linalg.norm(component_features[label]['avg_position'] - 
                                                     component_features[neighbor_label]['avg_position'])
                        # 综合距离
                        distance = curvature_diff + position_diff
                        
                        if distance < min_distance:
                            min_distance = distance
                            nearest_label = neighbor_label
                
                # 合并到最近的分量
                if nearest_label != -1:
                    for v in component_vertices:
                        labels[v] = nearest_label
                    merged_count += 1
        
        print(f"合并了 {merged_count} 个小分量")
        
        # 确保分区连通性
        print("确保分区连通性...")
        labels = self._ensure_connectivity(labels)
        
        # 重新编号标签
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
        
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print(f"基于对称性的分区完成: {len(unique_labels)} 个分区，总耗时: {elapsed:.2f}秒")
        print("="*60)
        return labels

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
