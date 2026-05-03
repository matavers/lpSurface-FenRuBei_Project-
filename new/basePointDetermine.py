
"""
新分区算法的基准点迭代部分
实现基准点的初始化类
实现基准点的迭代确定类
"""

from core.meshProcessor import MeshProcessor
from .newIndicator import NewIndicatorCalculator
import numpy as np
from typing import List, Set, Tuple, Dict
import heapq


class BasePointInitializer:
    def __init__(self, mesh: MeshProcessor, basePointNum: int):
        """
        初始化基准点初始化器,支持随机采样、均匀采样等方式初始化基准点
        Args:
            mesh (MeshProcessor): 网格处理器对象
            basePointNum (int): 基准点数量

        之后会加入基于网格微分几何特性的基准点确定方法<**unfinished**>
        """
        self.mesh = mesh
        self.basePointNum = basePointNum

    def random_sampling(self):
        """
        随机采样指定数量的点
        Returns:
            list: 选取的点在网格数组中的索引
        """
        num_vertices = len(self.mesh.vertices)
        if self.basePointNum >= num_vertices:
            return list(range(num_vertices))

        sampled_indices = np.random.choice(num_vertices, self.basePointNum, replace=False)
        return sampled_indices.tolist()

    def uniform_sampling(self):
        """
        均匀采样指定数量的点
        Returns:
            list: 选取的点在网格数组中的索引
        """
        vertices = self.mesh.vertices
        num_vertices = len(vertices)

        if self.basePointNum >= num_vertices:
            return list(range(num_vertices))

        sampled_indices = []
        first_idx = np.random.randint(0, num_vertices)
        sampled_indices.append(first_idx)

        distances = np.full(num_vertices, np.inf)

        for i in range(1, self.basePointNum):
            last_added_idx = sampled_indices[-1]
            last_added_vertex = vertices[last_added_idx]

            current_distances = np.linalg.norm(vertices - last_added_vertex, axis=1)
            distances = np.minimum(distances, current_distances)

            next_idx = np.argmax(distances)
            sampled_indices.append(next_idx)
            distances[next_idx] = 0

        return [int(idx) for idx in sampled_indices]

    def poisson_disk_sampling(self):
        """
        泊松碟采样 + 几何过滤
        Returns:
            list: 选取的点在网格数组中的索引
        """
        vertices = self.mesh.vertices
        num_vertices = len(vertices)

        if self.basePointNum >= num_vertices:
            return list(range(num_vertices))

        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        space_size = max_coords - min_coords
        volume = np.prod(space_size)
        density = num_vertices / volume
        radius = (1 / (density * np.sqrt(2) * self.basePointNum)) ** (1 / 3)
        cell_size = radius / np.sqrt(3)
        grid_shape = tuple(int(np.ceil(s / cell_size)) for s in space_size)
        grid = np.full(grid_shape, -1, dtype=int)

        sampled_indices = []
        first_idx = np.random.randint(0, num_vertices)
        sampled_indices.append(first_idx)
        first_pos = vertices[first_idx] - min_coords
        grid_pos = tuple(int(p / cell_size) for p in first_pos)
        grid[grid_pos] = first_idx
        active_list = [first_idx]

        while active_list and len(sampled_indices) < self.basePointNum:
            idx = np.random.randint(0, len(active_list))
            current_idx = active_list[idx]
            current_pos = vertices[current_idx]

            found = False
            for _ in range(30):
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                distance = radius * (np.random.rand() + 1)
                new_pos = current_pos + direction * distance

                if np.all(new_pos >= min_coords) and np.all(new_pos <= max_coords):
                    grid_pos = tuple(int((p - min_coords[i]) / cell_size) for i, p in enumerate(new_pos))

                    if all(0 <= p < grid_shape[i] for i, p in enumerate(grid_pos)):
                        valid = True
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                for dz in [-1, 0, 1]:
                                    neighbor_pos = tuple(p + d for p, d in zip(grid_pos, (dx, dy, dz)))
                                    if all(0 <= p < grid_shape[i] for i, p in enumerate(neighbor_pos)):
                                        neighbor_idx = grid[neighbor_pos]
                                        if neighbor_idx != -1:
                                            neighbor_pos = vertices[neighbor_idx]
                                            if np.linalg.norm(new_pos - neighbor_pos) < radius:
                                                valid = False
                                                break
                                if not valid:
                                    break
                            if not valid:
                                break

                        if valid:
                            distances = np.linalg.norm(vertices - new_pos, axis=1)
                            nearest_idx = np.argmin(distances)
                            sampled_indices.append(nearest_idx)
                            active_list.append(nearest_idx)
                            grid[grid_pos] = nearest_idx
                            found = True
                            break

            if not found:
                active_list.pop(idx)

        if len(sampled_indices) < self.basePointNum:
            distances = np.full(num_vertices, np.inf)
            for idx in sampled_indices:
                dists = np.linalg.norm(vertices - vertices[idx], axis=1)
                distances = np.minimum(distances, dists)

            while len(sampled_indices) < self.basePointNum:
                next_idx = np.argmax(distances)
                sampled_indices.append(next_idx)
                distances[next_idx] = 0
                new_dists = np.linalg.norm(vertices - vertices[next_idx], axis=1)
                distances = np.minimum(distances, new_dists)

        return [int(idx) for idx in sampled_indices]

    def spectral_clustering_initialization(self):
        """
        基于谱聚类的中心初始化
        Returns:
            list: 选取的点在网格数组中的索引
        """
        from sklearn.cluster import KMeans
        from sklearn.neighbors import kneighbors_graph
        from scipy.sparse.linalg import eigsh

        vertices = self.mesh.vertices
        num_vertices = len(vertices)

        if self.basePointNum >= num_vertices:
            return list(range(num_vertices))

        k = min(10, num_vertices - 1)
        adjacency = kneighbors_graph(vertices, n_neighbors=k, mode='connectivity', include_self=False)
        degree = np.array(adjacency.sum(axis=1)).flatten()
        laplacian = np.diag(degree) - adjacency.toarray()

        try:
            _, eigenvectors = eigsh(laplacian, k=self.basePointNum, which='SM')
        except:
            kmeans = KMeans(n_clusters=self.basePointNum, random_state=42)
            kmeans.fit(vertices)
            sampled_indices = []
            for center in kmeans.cluster_centers_:
                distances = np.linalg.norm(vertices - center, axis=1)
                nearest_idx = np.argmin(distances)
                sampled_indices.append(nearest_idx)
            return [int(idx) for idx in sampled_indices]

        kmeans = KMeans(n_clusters=self.basePointNum, random_state=42)
        kmeans.fit(eigenvectors)
        sampled_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(eigenvectors - center, axis=1)
            nearest_idx = np.argmin(distances)
            sampled_indices.append(nearest_idx)

        return [int(idx) for idx in sampled_indices]

    def sample(self, method='random'):
        """
        通用采样方法
        Args:
            method: 采样方法，可选值：'random'（随机采样）或 'uniform'（均匀采样）或 'poisson'（泊松碟采样）或 'spectral'（谱聚类中心初始化）
        Returns:
            list: 选取的点在网格数组中的索引
        """
        if method == 'random':
            return self.random_sampling()
        elif method == 'uniform':
            return self.uniform_sampling()
        elif method == 'poisson':
            return self.poisson_disk_sampling()
        elif method == 'spectral':
            return self.spectral_clustering_initialization()
        else:
            raise ValueError("不支持的采样方法，请使用 'random'、'uniform'、'poisson' 或 'spectral'")


class BasePointIteration:
    def __init__(
        self,
        mesh: MeshProcessor,
        indicator_calculator: NewIndicatorCalculator = None
    ):
        """
        初始化基准点迭代优化器
        Args:
            mesh (MeshProcessor): 网格处理器对象
            indicator_calculator (NewIndicatorCalculator): 指标计算器对象，可选
        """
        self.mesh = mesh
        self.num_vertices = len(mesh.vertices)

        if indicator_calculator is None:
            self.indicator_calculator = NewIndicatorCalculator(mesh)
        else:
            self.indicator_calculator = indicator_calculator

        self.alpha = 2.0
        self.R_max = None
        self.theta_attr = 1.5
        self.max_iterations = 100
        self.convergence_tolerance = 1e-3
        self.convergence_streak = 5

        self.region_cache: Dict[int, Set[int]] = {}
        self.iteration_data: List[Dict] = []

    def compute_regions(self, benchmarks: List[int]) -> Dict[int, Set[int]]:
        """
        为基准点列表计算区域
        Args:
            benchmarks: 基准点索引列表
        Returns:
            基准点到区域的映射字典
        """
        regions = {}
        for b in benchmarks:
            if b in self.region_cache:
                regions[b] = self.region_cache[b]
            else:
                region = self.indicator_calculator.grow_region(
                    b, self.alpha, self.R_max, self.theta_attr
                )
                regions[b] = region
                self.region_cache[b] = region
        return regions

    def compute_coverage(
        self,
        regions: Dict[int, Set[int]]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        计算顶点覆盖次数和未覆盖顶点
        Args:
            regions: 基准点到区域的映射字典
        Returns:
            (覆盖次数数组, 未覆盖顶点索引列表)
        """
        coverage = np.zeros(self.num_vertices, dtype=int)

        for region in regions.values():
            for v in region:
                coverage[v] += 1

        uncovered = np.where(coverage == 0)[0].tolist()
        return coverage, uncovered

    def compute_total_overlap(self, coverage: np.ndarray) -> float:
        """
        计算总重叠量
        Args:
            coverage: 覆盖次数数组
        Returns:
            总重叠量
        """
        return float(np.sum(coverage - 1))

    def compute_shortest_effective_distance(
        self,
        source: int,
        target: int
    ) -> float:
        """
        计算从 source 到 target 的最短有效路径距离，检查路径上所有顶点的属性差异
        Args:
            source: 源顶点索引
            target: 目标顶点索引
        Returns:
            最短有效距离（如果无法到达，返回无穷大）
        """
        dist = {v: float('inf') for v in range(self.num_vertices)}
        dist[source] = 0
        visited = set()

        heap = [(0.0, source)]

        while heap:
            current_dist, u = heapq.heappop(heap)

            if u == target:
                return current_dist

            if u in visited:
                continue
            visited.add(u)

            for v in self.mesh.adjacency[u]:
                attr_diff = self.indicator_calculator._calculate_attribute_difference(v, source)
                if attr_diff > self.theta_attr:
                    continue

                edge_len = self.indicator_calculator._calculate_effective_length(u, v, self.alpha)
                new_dist = current_dist + edge_len

                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(heap, (new_dist, v))

        return float('inf')

    def find_nearest_benchmark(
        self,
        vertex: int,
        benchmarks: List[int]
    ) -> Tuple[int, float]:
        """
        找到距离顶点最近的基准点
        Args:
            vertex: 目标顶点索引
            benchmarks: 基准点列表
        Returns:
            (最近基准点索引, 距离)
        """
        min_dist = float('inf')
        nearest_b = None

        for b in benchmarks:
            d = self.compute_shortest_effective_distance(b, vertex)
            if d < min_dist:
                min_dist = d
                nearest_b = b

        return nearest_b, min_dist

    def find_best_move_direction(
        self,
        current_b: int,
        target_vertex: int
    ) -> int:
        """
        找到基准点最佳移动方向，通过模拟区域生长判断
        Args:
            current_b: 当前基准点索引
            target_vertex: 目标顶点索引
        Returns:
            最佳邻域顶点索引（或原基准点）
        """
        neighbors = self.mesh.adjacency[current_b]
        best_vertex = current_b
        best_score = 0

        current_region = self.indicator_calculator.grow_region(
            current_b, self.alpha, self.R_max, self.theta_attr
        )
        if target_vertex in current_region:
            return current_b

        for neighbor in neighbors:
            candidate_region = self.indicator_calculator.grow_region(
                neighbor, self.alpha, self.R_max, self.theta_attr
            )

            score = 0
            if target_vertex in candidate_region:
                score += 10
            dist = self.compute_shortest_effective_distance(neighbor, target_vertex)
            if dist != float('inf'):
                score += (1.0 / (1.0 + dist)) * 5

            if score > best_score:
                best_score = score
                best_vertex = neighbor

        return best_vertex

    def fix_coverage(
        self,
        benchmarks: List[int],
        regions: Dict[int, Set[int]],
        uncovered: List[int]
    ) -> List[int]:
        """
        修复覆盖问题
        Args:
            benchmarks: 基准点列表
            regions: 区域字典
            uncovered: 未覆盖顶点列表
        Returns:
            更新后的基准点列表
        """
        print(f"修复覆盖，未覆盖顶点数: {len(uncovered)}")

        new_benchmarks = benchmarks.copy()

        for u in uncovered:
            nearest_b, min_dist = self.find_nearest_benchmark(u, new_benchmarks)

            if nearest_b is not None and min_dist != float('inf'):
                new_b = self.find_best_move_direction(nearest_b, u)

                if new_b != nearest_b:
                    idx = new_benchmarks.index(nearest_b)
                    new_benchmarks[idx] = new_b

                    if nearest_b in self.region_cache:
                        del self.region_cache[nearest_b]
            else:
                new_benchmarks.append(u)

        return new_benchmarks

    def compute_unique_vertices(
        self,
        benchmark: int,
        regions: Dict[int, Set[int]],
        coverage: np.ndarray
    ) -> Set[int]:
        """
        计算基准点的独有顶点
        Args:
            benchmark: 基准点索引
            regions: 区域字典
            coverage: 覆盖次数数组
        Returns:
            独有顶点集合
        """
        unique = set()
        region = regions[benchmark]

        for v in region:
            if coverage[v] == 1:
                unique.add(v)

        return unique

    def compute_avg_overlap(
        self,
        benchmark: int,
        regions: Dict[int, Set[int]],
        coverage: np.ndarray
    ) -> float:
        """
        计算区域的平均重叠度
        Args:
            benchmark: 基准点索引
            regions: 区域字典
            coverage: 覆盖次数数组
        Returns:
            平均重叠度
        """
        region = regions[benchmark]

        if not region:
            return 0.0

        total_overlap = 0.0
        for v in region:
            total_overlap += (coverage[v] - 1)

        return total_overlap / len(region)

    def reduce_overlap(
        self,
        benchmarks: List[int],
        regions: Dict[int, Set[int]],
        coverage: np.ndarray
    ) -> Tuple[List[int], bool]:
        """
        减少重叠
        Args:
            benchmarks: 基准点列表
            regions: 区域字典
            coverage: 覆盖次数数组
        Returns:
            (更新后的基准点列表, 是否有改进)
        """
        new_benchmarks = benchmarks.copy()
        has_improvement = False

        for b in benchmarks:
            unique = self.compute_unique_vertices(b, regions, coverage)
            if len(unique) == 0:
                new_benchmarks.remove(b)
                if b in self.region_cache:
                    del self.region_cache[b]
                has_improvement = True
                print(f"删除冗余基准点 {b}")
                return new_benchmarks, has_improvement

        max_avg_overlap = -1.0
        best_b = None

        for b in benchmarks:
            avg_overlap = self.compute_avg_overlap(b, regions, coverage)
            if avg_overlap > max_avg_overlap:
                max_avg_overlap = avg_overlap
                best_b = b

        if best_b is not None:
            neighbors = self.mesh.adjacency[best_b]
            best_new_b = None
            best_overlap_reduction = 0.0

            for neighbor in neighbors:
                temp_benchmarks = [b for b in benchmarks if b != best_b]
                temp_benchmarks.append(neighbor)

                temp_regions = {}
                for tb in temp_benchmarks:
                    if tb in self.region_cache and tb != neighbor:
                        temp_regions[tb] = self.region_cache[tb]
                    else:
                        tr = self.indicator_calculator.grow_region(
                            tb, self.alpha, self.R_max, self.theta_attr
                        )
                        temp_regions[tb] = tr

                temp_coverage, _ = self.compute_coverage(temp_regions)

                if np.all(temp_coverage > 0):
                    temp_total_overlap = self.compute_total_overlap(temp_coverage)
                    current_total_overlap = self.compute_total_overlap(coverage)

                    overlap_reduction = current_total_overlap - temp_total_overlap
                    if overlap_reduction > best_overlap_reduction:
                        best_overlap_reduction = overlap_reduction
                        best_new_b = neighbor

            if best_new_b is not None and best_overlap_reduction > self.convergence_tolerance:
                idx = new_benchmarks.index(best_b)
                new_benchmarks[idx] = best_new_b

                if best_b in self.region_cache:
                    del self.region_cache[best_b]

                has_improvement = True
                print(f"移动基准点 {best_b} -> {best_new_b}，重叠减少 {best_overlap_reduction:.2f}")

        return new_benchmarks, has_improvement

    def optimize(
        self,
        initial_benchmarks: List[int],
        alpha: float = 2.0,
        R_max: float = None,
        theta_attr: float = 1.5,
        max_iterations: int = 100
    ) -> Tuple[List[int], Dict[int, Set[int]], np.ndarray]:
        """
        执行基准点迭代优化
        Args:
            initial_benchmarks: 初始基准点列表
            alpha: 曲率拉伸强度
            R_max: 最大有效半径
            theta_attr: 属性差异阈值
            max_iterations: 最大迭代次数
        Returns:
            (优化后的基准点列表, 区域字典, 最终覆盖次数数组)
        """
        self.alpha = alpha
        self.R_max = R_max
        self.theta_attr = theta_attr
        self.max_iterations = max_iterations

        benchmarks = initial_benchmarks.copy()
        self.region_cache = {}
        self.iteration_data = []

        prev_total_overlap = float('inf')
        no_improvement_count = 0

        print(f"开始基准点优化，初始基准点数: {len(benchmarks)}")

        for iteration in range(self.max_iterations):
            print(f"\n=== 迭代 {iteration + 1}/{self.max_iterations} ===")

            regions = self.compute_regions(benchmarks)
            coverage, uncovered = self.compute_coverage(regions)
            total_overlap = self.compute_total_overlap(coverage)

            avg_coverage = np.mean(coverage)
            print(f"当前基准点数: {len(benchmarks)}, 未覆盖: {len(uncovered)}, 总重叠: {total_overlap:.2f}, 平均覆盖: {avg_coverage:.2f}")

            self.iteration_data.append({
                "iteration": iteration + 1,
                "num_benchmarks": len(benchmarks),
                "uncovered": len(uncovered),
                "total_overlap": total_overlap,
                "avg_coverage": avg_coverage
            })

            if len(uncovered) > 0:
                benchmarks = self.fix_coverage(benchmarks, regions, uncovered)
                no_improvement_count = 0

                regions = self.compute_regions(benchmarks)
                coverage, uncovered = self.compute_coverage(regions)
                total_overlap = self.compute_total_overlap(coverage)
                prev_total_overlap = total_overlap
            else:
                benchmarks, has_improvement = self.reduce_overlap(benchmarks, regions, coverage)

                if has_improvement:
                    no_improvement_count = 0
                    prev_total_overlap = total_overlap
                else:
                    overlap_change = abs(prev_total_overlap - total_overlap)
                    if overlap_change < self.convergence_tolerance:
                        no_improvement_count += 1
                    else:
                        no_improvement_count = 0

                    prev_total_overlap = total_overlap

                    if no_improvement_count >= self.convergence_streak:
                        print(f"收敛，连续 {self.convergence_streak} 次无改进")
                        break

        final_regions = self.compute_regions(benchmarks)
        final_coverage, _ = self.compute_coverage(final_regions)

        print(f"\n优化完成！")
        print(f"最终基准点数: {len(benchmarks)}")
        print(f"最终平均覆盖: {np.mean(final_coverage):.2f}")
        print(f"最终总重叠: {self.compute_total_overlap(final_coverage):.2f}")

        return benchmarks, final_regions, final_coverage
