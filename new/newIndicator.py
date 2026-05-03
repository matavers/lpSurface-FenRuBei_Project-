from core.meshProcessor import MeshProcessor
import numpy as np
import heapq
from typing import Dict, Set, List, Tuple

class NewIndicatorCalculator:
    def __init__(self,mesh:MeshProcessor):
        """
        初始化新指标类
        Args:
            mesh (MeshProcessor): 网格处理器对象
        """
        self.mesh=mesh
        # 预计算平均边长，用于设置默认的R_max
        self.avg_edge_length = self._calculate_avg_edge_length()
        # 预计算全局归一化因子
        self.sigma_K, self.sigma_n, self.sigma_R = self._precompute_normalization_factors()
        # 缓存度量张量
        self.metric_tensor_cache = {}
        # 缓存属性差异值
        self.attribute_diff_cache = {}
        # 缓存有效长度
        self.effective_length_cache = {}
        # 缓存区域生长结果
        self.grow_region_cache = {}
        # 缓存相似性判断结果
        self.is_similar_cache = {}
    
    def _calculate_avg_edge_length(self) -> float:
        """
        计算网格的平均边长
        Returns:
            float: 平均边长
        """
        total_length = 0
        edge_count = 0
        
        # 遍历所有边，计算总长度
        for i, neighbors in enumerate(self.mesh.adjacency):
            for j in neighbors:
                if i < j:  # 避免重复计算
                    edge_vector = self.mesh.vertices[i] - self.mesh.vertices[j]
                    total_length += np.linalg.norm(edge_vector)
                    edge_count += 1
        
        return total_length / edge_count if edge_count > 0 else 1.0
    
    def _precompute_normalization_factors(self) -> Tuple[float, float, float]:
        """
        预计算全局归一化因子
        Returns:
            tuple: (sigma_K, sigma_n, sigma_R)
        """
        # 计算sigma_K
        if hasattr(self.mesh, 'gaussian_curvatures'):
            sigma_K = np.std(self.mesh.gaussian_curvatures)
        else:
            avg_curvatures = [(k1 + k2) / 2 for k1, k2 in self.mesh.principal_curvatures]
            sigma_K = np.std(avg_curvatures)
        
        # 设置合理范围，防止sigma_K过小或过大
        # 过小会导致属性差异被过度放大
        # 过大会导致归一化失效
        sigma_K = max(0.01, min(sigma_K, 10.0))
        
        # 计算sigma_n - 法向量夹角的标准差
        normal_angles = []
        for i, neighbors in enumerate(self.mesh.adjacency):
            for j in neighbors:
                if i < j:  # 避免重复计算
                    n_i = self.mesh.vertex_normals[i]
                    n_j = self.mesh.vertex_normals[j]
                    dot_product = np.clip(np.dot(n_i, n_j), -1.0, 1.0)
                    angle = np.arccos(dot_product)
                    normal_angles.append(angle)
        
        if normal_angles:
            sigma_n = np.std(normal_angles)
            sigma_n = max(0.01, sigma_n)  # 防止过小
        else:
            sigma_n = 1.0
        
        # 计算sigma_R
        if hasattr(self.mesh, 'rolled_error'):
            sigma_R = np.std(self.mesh.rolled_error)
        else:
            sigma_R = 1.0
        sigma_R = max(0.01, sigma_R)  # 防止过小
        
        return sigma_K, sigma_n, sigma_R
    
    def _calculate_metric_tensor(self, vertex_idx: int, alpha: float) -> np.ndarray:
        """
        计算顶点的局部度量张量
        Args:
            vertex_idx: 顶点索引
            alpha: 曲率拉伸强度
        Returns:
            np.ndarray: 3x3度量张量
        """
        # 检查缓存
        cache_key = (vertex_idx, alpha)
        if cache_key in self.metric_tensor_cache:
            return self.metric_tensor_cache[cache_key]
        
        # 单位矩阵
        I = np.eye(3)
        
        # 获取主曲率和主方向
        k1, k2 = self.mesh.principal_curvatures[vertex_idx]
        d1 = self.mesh.principal_directions1[vertex_idx]
        d2 = self.mesh.principal_directions2[vertex_idx]
        
        # 限制曲率值，防止过度膨胀
        # 将曲率限制在 [-50, 50] 范围内
        k1_clamped = max(-50.0, min(50.0, k1))
        k2_clamped = max(-50.0, min(50.0, k2))
        
        # 计算度量张量
        term = alpha * (abs(k1_clamped) * np.outer(d1, d1) + abs(k2_clamped) * np.outer(d2, d2))
        M = I + term
        
        # 缓存结果
        self.metric_tensor_cache[cache_key] = M
        
        return M
    
    def _calculate_effective_length(self, u: int, v: int, alpha: float) -> float:
        """
        计算边的有效长度
        Args:
            u: 起点顶点索引
            v: 终点顶点索引
            alpha: 曲率拉伸强度
        Returns:
            float: 有效长度
        """
        # 确保u < v，避免重复计算
        if u > v:
            u, v = v, u
        
        # 检查缓存
        cache_key = (u, v, alpha)
        if cache_key in self.effective_length_cache:
            return self.effective_length_cache[cache_key]
        
        # 计算边的欧氏长度
        edge_vector = self.mesh.vertices[v] - self.mesh.vertices[u]
        edge_length = np.linalg.norm(edge_vector)
        
        if edge_length < 1e-8:
            result = 0.0
        else:
            # 计算单位方向向量
            e_hat = edge_vector / edge_length
            
            # 计算两端点的度量张量
            M_u = self._calculate_metric_tensor(u, alpha)
            M_v = self._calculate_metric_tensor(v, alpha)
            
            # 计算平均度量张量
            M_avg = (M_u + M_v) / 2
            
            # 计算有效长度
            result = edge_length * np.sqrt(np.dot(e_hat, np.dot(M_avg, e_hat)))
        
        # 缓存结果
        self.effective_length_cache[cache_key] = result
        
        return result
    
    def _calculate_attribute_difference(self, v: int, b: int) -> float:
        """
        计算顶点v与基准点b的属性差异
        Args:
            v: 候选顶点索引
            b: 基准点索引
        Returns:
            float: 属性差异（使用法向夹角，单位：度）
        """
        # 检查缓存
        cache_key = (v, b)
        if cache_key in self.attribute_diff_cache:
            return self.attribute_diff_cache[cache_key]
        
        # 计算法向差异（主要依赖这个，因为曲率估计不稳定）
        n_v = self.mesh.vertex_normals[v]
        n_b = self.mesh.vertex_normals[b]
        dot_product = np.dot(n_v, n_b)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        normal_angle = np.arccos(dot_product)
        
        # 将弧度转换为度作为属性差异值
        result = np.degrees(normal_angle)
        
        # 缓存结果
        self.attribute_diff_cache[cache_key] = result
        
        return result
    
    def grow_region(self, benchmark: int, alpha: float = 2.0, R_max: float = None, theta_attr: float = 1.5, debug: bool = False) -> Set[int]:
        """
        从基准点生长区域
        Args:
            benchmark: 基准点索引
            alpha: 曲率拉伸强度
            R_max: 最大有效半径
            theta_attr: 属性差异阈值
            debug: 是否打印调试信息
        Returns:
            Set[int]: 区域内的顶点集合
        """
        # 如果未指定R_max，使用默认值（10倍平均边长）
        if R_max is None:
            R_max = 10 * self.avg_edge_length
        
        # 调试输出
        if debug:
            print(f"\n=== grow_region Debug for benchmark {benchmark} ===")
            print(f"alpha={alpha}, R_max={R_max:.4f}, theta_attr={theta_attr}")
            print(f"sigma_K={self.sigma_K:.4f}, sigma_n={self.sigma_n:.4f}, sigma_R={self.sigma_R:.4f}")
            print(f"avg_edge_length={self.avg_edge_length:.4f}")
        
        # 检查缓存
        cache_key = (benchmark, alpha, R_max, theta_attr)
        if cache_key in self.grow_region_cache:
            if debug:
                print(f"Returning cached result for benchmark {benchmark}")
            return self.grow_region_cache[cache_key]
        
        # 初始化距离字典
        dist = {v: float('inf') for v in range(len(self.mesh.vertices))}
        dist[benchmark] = 0
        
        # 初始化优先队列
        queue = [(0, benchmark)]
        
        # 初始化区域集合
        region = set()
        
        # 统计信息
        rejected_by_attr = 0
        rejected_by_dist = 0
        first_few_neighbors_checked = 0
        
        while queue:
            # 弹出距离最小的顶点
            d, u = heapq.heappop(queue)
            
            # 如果距离超过最大半径，跳过
            if d > R_max:
                rejected_by_dist += 1
                continue
            
            # 检查属性差异
            attr_diff = self._calculate_attribute_difference(u, benchmark)
            if attr_diff > theta_attr:
                rejected_by_attr += 1
                # 调试输出前几个被拒绝的顶点
                if debug and first_few_neighbors_checked < 5 and u != benchmark:
                    first_few_neighbors_checked += 1
                    print(f"  Vertex {u}: attr_diff={attr_diff:.3f} > {theta_attr}, REJECTED")
                continue
            
            # 将顶点加入区域
            region.add(u)
            
            # 遍历邻居顶点
            for v in self.mesh.adjacency[u]:
                # 计算有效长度
                edge_len = self._calculate_effective_length(u, v, alpha)
                new_d = d + edge_len
                
                # 调试输出前几个邻居的信息
                if debug and u == benchmark and first_few_neighbors_checked < 3:
                    print(f"  First neighbor {v}: edge_len={edge_len:.4f}, new_d={new_d:.4f}, R_max={R_max:.4f}")
                
                # 如果新距离更小，更新距离并加入队列
                if new_d < dist[v]:
                    dist[v] = new_d
                    heapq.heappush(queue, (new_d, v))
        
        # 调试输出统计信息
        if debug:
            print(f"Region size: {len(region)}")
            print(f"Rejected by distance: {rejected_by_dist}")
            print(f"Rejected by attribute: {rejected_by_attr}")
        
        # 缓存结果
        self.grow_region_cache[cache_key] = region
        
        return region
    
    def is_similar(self, v: int, b: int, alpha: float = 2.0, R_max: float = None, theta_attr: float = 1.5) -> bool:
        """
        判断顶点v与基准点b是否相似
        Args:
            v: 候选顶点索引
            b: 基准点索引
            alpha: 曲率拉伸强度
            R_max: 最大有效半径
            theta_attr: 属性差异阈值
        Returns:
            bool: 是否相似
        """
        # 如果未指定R_max，使用默认值（5倍平均边长）
        if R_max is None:
            R_max = 5 * self.avg_edge_length
        
        # 检查缓存
        cache_key = (v, b, alpha, R_max, theta_attr)
        if cache_key in self.is_similar_cache:
            return self.is_similar_cache[cache_key]
        
        # 初始化距离字典
        dist = {vertex: float('inf') for vertex in range(len(self.mesh.vertices))}
        dist[b] = 0
        queue = [(0, b)]
        
        result = False
        while queue:
            d, u = heapq.heappop(queue)
            
            # 如果到达目标顶点，返回结果
            if u == v:
                result = d <= R_max
                break
            
            # 如果距离超过最大半径，跳过
            if d > R_max:
                continue
            
            # 检查属性差异
            if self._calculate_attribute_difference(u, b) > theta_attr:
                continue
            
            # 遍历邻居顶点
            for neighbor in self.mesh.adjacency[u]:
                # 计算有效长度
                edge_len = self._calculate_effective_length(u, neighbor, alpha)
                new_d = d + edge_len
                
                # 如果新距离更小，更新距离并加入队列
                if new_d < dist[neighbor]:
                    dist[neighbor] = new_d
                    heapq.heappush(queue, (new_d, neighbor))
        
        # 缓存结果
        self.is_similar_cache[cache_key] = result
        
        return result
    
    def clear_cache(self):
        """
        清理所有缓存
        """
        self.metric_tensor_cache.clear()
        self.attribute_diff_cache.clear()
        self.effective_length_cache.clear()
        self.grow_region_cache.clear()
        self.is_similar_cache.clear()
    
    def clear_cache_by_alpha(self, alpha: float):
        """
        清理指定alpha值的缓存
        Args:
            alpha: 曲率拉伸强度
        """
        # 清理度量张量缓存
        keys_to_remove = [key for key in self.metric_tensor_cache if key[1] == alpha]
        for key in keys_to_remove:
            del self.metric_tensor_cache[key]
        
        # 清理有效长度缓存
        keys_to_remove = [key for key in self.effective_length_cache if key[2] == alpha]
        for key in keys_to_remove:
            del self.effective_length_cache[key]
        
        # 清理区域生长缓存
        keys_to_remove = [key for key in self.grow_region_cache if key[1] == alpha]
        for key in keys_to_remove:
            del self.grow_region_cache[key]
        
        # 清理相似性判断缓存
        keys_to_remove = [key for key in self.is_similar_cache if key[2] == alpha]
        for key in keys_to_remove:
            del self.is_similar_cache[key]