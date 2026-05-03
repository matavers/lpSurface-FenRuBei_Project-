
import numpy as np
from typing import List, Set, Dict, Tuple
from scipy.linalg import svd


def calculate_coverage_metrics(
    coverage: np.ndarray
) -> Dict[str, float]:
    """
    计算覆盖相关指标
    
    Args:
        coverage: 覆盖次数数组
        
    Returns:
        指标字典
    """
    num_vertices = len(coverage)
    uncovered_count = np.sum(coverage == 0)
    uncovered_ratio = uncovered_count / num_vertices
    avg_coverage = np.mean(coverage)
    total_overlap = np.sum(coverage - 1)
    
    return {
        "uncovered_ratio": uncovered_ratio,
        "avg_coverage": avg_coverage,
        "total_overlap": total_overlap,
        "max_coverage": np.max(coverage)
    }


def calculate_partition_anisotropy(
    partition_vertices: np.ndarray
) -> float:
    """
    计算单个分区的各向异性比（主成分分析）
    
    Args:
        partition_vertices: 分区顶点坐标，shape=(n, 3)
        
    Returns:
        各向异性比 λ_max / λ_min
    """
    if len(partition_vertices) < 3:
        return 1.0
    
    centered = partition_vertices - np.mean(partition_vertices, axis=0)
    
    try:
        U, s, Vt = svd(centered, full_matrices=False)
        if len(s) < 2:
            return 1.0
        
        lambda_max = s[0]
        lambda_min = s[-1]
        
        if lambda_min < 1e-8:
            return float('inf')
        
        return lambda_max / lambda_min
    except Exception:
        return 1.0


def calculate_all_partitions_anisotropy(
    partitions: List[Set[int]],
    vertices: np.ndarray
) -> List[float]:
    """
    计算所有分区的各向异性比
    
    Args:
        partitions: 分区列表
        vertices: 顶点坐标
        
    Returns:
        各分区的各向异性比列表
    """
    anisotropy_ratios = []
    
    for region in partitions:
        if len(region) < 3:
            continue
        
        region_vertices = vertices[list(region)]
        ratio = calculate_partition_anisotropy(region_vertices)
        anisotropy_ratios.append(ratio)
    
    return anisotropy_ratios


def calculate_boundary_straightness(
    edge_midpoints: np.ndarray
) -> float:
    """
    计算边界的平直度（边界点到最佳拟合直线的平均偏差）
    
    Args:
        edge_midpoints: 边界中点数组，shape=(n, 3)
        
    Returns:
        平均偏差（越小越平直）
    """
    if len(edge_midpoints) < 3:
        return 0.0
    
    try:
        centered = edge_midpoints - np.mean(edge_midpoints, axis=0)
        U, s, Vt = svd(centered, full_matrices=False)
        
        direction = Vt[0]
        point = edge_midpoints[0]
        
        distances = []
        for p in edge_midpoints:
            v = p - point
            proj = np.dot(v, direction) * direction
            perp = v - proj
            distances.append(np.linalg.norm(perp))
        
        return np.mean(distances)
    except Exception:
        return 0.0


def calculate_redundant_benchmarks_ratio(
    benchmarks: List[int],
    regions_dict: Dict[int, Set[int]],
    coverage: np.ndarray
) -> float:
    """
    计算冗余基准点比例
    
    Args:
        benchmarks: 基准点列表
        regions_dict: 基准点到区域的映射
        coverage: 覆盖次数数组
        
    Returns:
        冗余基准点比例
    """
    if len(benchmarks) == 0:
        return 0.0
    
    redundant_count = 0
    
    for b in benchmarks:
        region = regions_dict[b]
        has_unique = False
        
        for v in region:
            if coverage[v] == 1:
                has_unique = True
                break
        
        if not has_unique:
            redundant_count += 1
    
    return redundant_count / len(benchmarks)


def evaluate_full_partition(
    benchmarks: List[int],
    regions_dict: Dict[int, Set[int]],
    coverage: np.ndarray,
    vertices: np.ndarray,
    edge_midpoints: np.ndarray
) -> Dict:
    """
    完整评价分区结果
    
    Args:
        benchmarks: 基准点列表
        regions_dict: 区域字典
        coverage: 覆盖次数数组
        vertices: 顶点坐标
        edge_midpoints: 边界中点
        
    Returns:
        完整指标字典
    """
    metrics = {}
    
    coverage_metrics = calculate_coverage_metrics(coverage)
    metrics.update(coverage_metrics)
    
    partitions_list = list(regions_dict.values())
    anisotropy_ratios = calculate_all_partitions_anisotropy(partitions_list, vertices)
    
    if anisotropy_ratios:
        metrics["mean_anisotropy"] = np.mean(anisotropy_ratios)
        metrics["max_anisotropy"] = np.max(anisotropy_ratios)
        metrics["min_anisotropy"] = np.min(anisotropy_ratios)
        metrics["median_anisotropy"] = np.median(anisotropy_ratios)
    
    metrics["boundary_straightness"] = calculate_boundary_straightness(edge_midpoints)
    
    metrics["redundant_ratio"] = calculate_redundant_benchmarks_ratio(
        benchmarks, regions_dict, coverage
    )
    
    metrics["num_benchmarks"] = len(benchmarks)
    metrics["num_partitions"] = len(regions_dict)
    
    return metrics


def print_metrics_summary(metrics: Dict, name: str = "Partition"):
    """
    打印指标摘要
    
    Args:
        metrics: 指标字典
        name: 测试名称
    """
    print(f"\n===== {name} Metrics =====")
    
    coverage_keys = ["uncovered_ratio", "avg_coverage", "total_overlap", "max_coverage"]
    if all(k in metrics for k in coverage_keys):
        print(f"Coverage:")
        print(f"  Uncovered ratio: {metrics['uncovered_ratio']:.2%}")
        print(f"  Average coverage: {metrics['avg_coverage']:.2f}")
        print(f"  Total overlap: {metrics['total_overlap']:.0f}")
        print(f"  Max coverage: {metrics['max_coverage']:.0f}")
    
    anisotropy_keys = ["mean_anisotropy", "median_anisotropy"]
    if all(k in metrics for k in anisotropy_keys):
        print(f"Anisotropy:")
        print(f"  Mean: {metrics['mean_anisotropy']:.2f}")
        print(f"  Median: {metrics['median_anisotropy']:.2f}")
        if "max_anisotropy" in metrics:
            print(f"  Max: {metrics['max_anisotropy']:.2f}")
    
    if "boundary_straightness" in metrics:
        print(f"Boundary straightness: {metrics['boundary_straightness']:.4f}")
    
    if "redundant_ratio" in metrics:
        print(f"Redundant benchmarks: {metrics['redundant_ratio']:.2%}")
    
    if "num_benchmarks" in metrics:
        print(f"Number of benchmarks: {metrics['num_benchmarks']}")
