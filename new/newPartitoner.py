
from core.meshProcessor import MeshProcessor
import numpy as np
from typing import Tuple, List, Set, Dict
from .newIndicator import NewIndicatorCalculator
from .basePointDetermine import BasePointInitializer, BasePointIteration


class NewPartitioner:
    def __init__(self, mesh: MeshProcessor):
        """
        初始化新分区类
        Args:
            mesh (MeshProcessor): 网格处理器对象
        """
        self.mesh = mesh
        self.indicator_calculator = NewIndicatorCalculator(mesh)
        self.num_vertices = len(mesh.vertices)

    def partition_surface(
        self,
        benchmarks: List[int],
        alpha: float = 0.0,
        R_max: float = None,
        theta_attr: float = 30.0
    ) -> Tuple[List[Set[int]], Dict[int, List[int]], np.ndarray]:
        """
        执行表面分区，基于基准点列表生长区域，保留分区重叠信息
        Args:
            benchmarks: 基准点索引列表
            alpha: 曲率拉伸强度
            R_max: 最大有效半径
            theta_attr: 属性差异阈值
        Returns:
            (分区列表, 顶点到分区索引的映射, 边缘中点数组)
        """
        print("开始基于相似性指标的表面分区...")
        print(f"使用 {len(benchmarks)} 个基准点")

        partitions = []
        regions_dict = {}

        for idx, b in enumerate(benchmarks):
            region = self.indicator_calculator.grow_region(b, alpha, R_max, theta_attr)
            partitions.append(region)
            regions_dict[b] = region
            print(f"基准点 {idx} (索引 {b}) 生成区域大小: {len(region)}")

        vertex_to_partitions = self._build_vertex_to_partitions_mapping(partitions)
        edge_midpoints = self._extract_edge_midpoints(vertex_to_partitions, partitions)

        print(f"分区完成: {len(partitions)} 个分区")

        return partitions, vertex_to_partitions, edge_midpoints

    def partition_with_optimization(
        self,
        initial_num_benchmarks: int = None,
        initial_benchmarks: List[int] = None,
        sampling_method: str = 'uniform',
        alpha: float = 0.0,
        R_max: float = None,
        theta_attr: float = 30.0,
        max_iterations: int = 100
    ) -> Tuple[List[int], Dict[int, Set[int]], np.ndarray, Dict[int, List[int]], np.ndarray, List[Dict]]:
        """
        执行带基准点优化的完整分区，保留分区重叠信息
        Args:
            initial_num_benchmarks: 初始基准点数量（如果没有提供初始基准点）
            initial_benchmarks: 初始基准点列表（可选）
            sampling_method: 采样方法（如果没有提供初始基准点）
            alpha: 曲率拉伸强度
            R_max: 最大有效半径
            theta_attr: 属性差异阈值
            max_iterations: 最大迭代次数
        Returns:
            (优化后的基准点列表, 区域字典, 最终覆盖次数数组, 顶点到分区的映射, 边缘中点数组, 迭代数据)
        """
        print("=== 开始完整分区流程 ===\n")

        if initial_benchmarks is None:
            if initial_num_benchmarks is None:
                initial_num_benchmarks = max(5, self.num_vertices // 100)

            initializer = BasePointInitializer(self.mesh, initial_num_benchmarks)
            initial_benchmarks = initializer.sample(method=sampling_method)
            print(f"初始化 {len(initial_benchmarks)} 个基准点，方法: {sampling_method}\n")
        else:
            print(f"使用提供的 {len(initial_benchmarks)} 个基准点\n")

        optimizer = BasePointIteration(self.mesh, self.indicator_calculator)
        optimized_benchmarks, regions_dict, final_coverage = optimizer.optimize(
            initial_benchmarks,
            alpha=alpha,
            R_max=R_max,
            theta_attr=theta_attr,
            max_iterations=max_iterations
        )

        iteration_data = optimizer.iteration_data if hasattr(optimizer, 'iteration_data') else []

        print("\n=== 生成分区结果 ===")
        partitions_list = [regions_dict[b] for b in optimized_benchmarks]
        vertex_to_partitions = self._build_vertex_to_partitions_mapping(partitions_list)
        edge_midpoints = self._extract_edge_midpoints(vertex_to_partitions, partitions_list)

        print(f"\n完整流程完成！")
        return optimized_benchmarks, regions_dict, final_coverage, vertex_to_partitions, edge_midpoints, iteration_data

    def _build_vertex_to_partitions_mapping(
        self,
        partitions: List[Set[int]]
    ) -> Dict[int, List[int]]:
        """
        构建顶点到分区索引的映射，保留重叠信息
        Args:
            partitions: 分区列表
        Returns:
            顶点索引到分区索引列表的映射
        """
        vertex_to_partitions = {v: [] for v in range(self.num_vertices)}

        for partition_idx, region in enumerate(partitions):
            for vertex_idx in region:
                vertex_to_partitions[vertex_idx].append(partition_idx)

        return vertex_to_partitions

    def _extract_edge_midpoints(
        self,
        vertex_to_partitions: Dict[int, List[int]],
        partitions: List[Set[int]]
    ) -> np.ndarray:
        """
        提取分区边缘的中点
        Args:
            vertex_to_partitions: 顶点到分区的映射
            partitions: 分区列表
        Returns:
            边缘中点数组
        """
        print("提取分区边缘中点...")

        edge_midpoints = []

        for i in range(self.num_vertices):
            neighbors = self.mesh.adjacency[i]
            for j in neighbors:
                if i < j:
                    partitions_i = vertex_to_partitions[i]
                    partitions_j = vertex_to_partitions[j]

                    if not partitions_i or not partitions_j:
                        continue

                    if set(partitions_i) != set(partitions_j):
                        midpoint = (self.mesh.vertices[i] + self.mesh.vertices[j]) / 2
                        edge_midpoints.append(midpoint)

        edge_midpoints = np.array(edge_midpoints)
        print(f"提取到 {len(edge_midpoints)} 个边缘中点")

        return edge_midpoints
