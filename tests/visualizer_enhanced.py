"""
增强的可视化功能
"""
import numpy as np
import open3d as o3d
from typing import List, Set, Dict, Tuple
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# 忽略 matplotlib 中文字体警告
warnings.filterwarnings('ignore', category=UserWarning, message="Glyph.*missing from font")

# 设置 matplotlib 使用支持中文的字体（如果可用）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


def generate_colors(num_colors: int) -> np.ndarray:
    """生成明显不同的颜色"""
    colors = plt.cm.hsv(np.linspace(0, 1, num_colors, endpoint=False))
    return colors[:, :3]


def save_colored_mesh_open3d(
    mesh: o3d.geometry.TriangleMesh,
    vertex_to_partitions: Dict[int, List[int]],
    benchmarks: List[int],
    output_path: str
):
    """使用 Open3D 保存彩色网格"""
    num_vertices = len(np.asarray(mesh.vertices))
    vertex_colors = np.ones((num_vertices, 3)) * 0.8
    
    # 统计分区数
    num_partitions = 0
    if vertex_to_partitions:
        all_partitions = set()
        for p_list in vertex_to_partitions.values():
            all_partitions.update(p_list)
        num_partitions = len(all_partitions)
    
    # 为每个顶点设置颜色（使用它属于的第一个分区）
    if num_partitions > 0:
        colors = generate_colors(num_partitions)
        for v in range(num_vertices):
            p_list = vertex_to_partitions.get(v, [])
            if p_list:
                p_idx = p_list[0]
                vertex_colors[v] = colors[p_idx % len(colors)]
    
    # 高亮基准点
    if benchmarks:
        benchmark_set = set(benchmarks)
        for b in benchmark_set:
            if b < len(vertex_colors):
                vertex_colors[b] = [1.0, 0.0, 0.0]  # 红色
    
    # 设置颜色并保存
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Saved colored mesh to {output_path}")
    
    return mesh


def save_edge_midpoints_open3d(
    edge_midpoints: np.ndarray,
    output_path: str,
    color: List[float] = [0.0, 1.0, 0.0]
):
    """使用 Open3D 保存边界中点"""
    if len(edge_midpoints) == 0:
        return
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(edge_midpoints)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(edge_midpoints), 1)))
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved edge midpoints to {output_path}")


def visualize_interactive(
    mesh: o3d.geometry.TriangleMesh,
    vertex_to_partitions: Dict[int, List[int]],
    edge_midpoints: np.ndarray,
    benchmarks: List[int],
    window_name: str = "分区可视化"
):
    """交互式可视化分区结果"""
    # 创建彩色网格
    num_vertices = len(np.asarray(mesh.vertices))
    vertex_colors = np.ones((num_vertices, 3)) * 0.8
    
    # 统计分区数
    num_partitions = 0
    if vertex_to_partitions:
        all_partitions = set()
        for p_list in vertex_to_partitions.values():
            all_partitions.update(p_list)
        num_partitions = len(all_partitions)
    
    # 为每个顶点设置颜色
    if num_partitions > 0:
        colors = generate_colors(num_partitions)
        for v in range(num_vertices):
            p_list = vertex_to_partitions.get(v, [])
            if p_list:
                p_idx = p_list[0]
                vertex_colors[v] = colors[p_idx % len(colors)]
    
    # 高亮基准点
    if benchmarks:
        benchmark_set = set(benchmarks)
        for b in benchmark_set:
            if b < len(vertex_colors):
                vertex_colors[b] = [1.0, 0.0, 0.0]
    
    colored_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
    )
    colored_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    # 创建边界点云
    objects = [colored_mesh]
    
    if len(edge_midpoints) > 0:
        edge_pcd = o3d.geometry.PointCloud()
        edge_pcd.points = o3d.utility.Vector3dVector(edge_midpoints)
        edge_pcd.colors = o3d.utility.Vector3dVector(
            np.tile([0.0, 1.0, 0.0], (len(edge_midpoints), 1))
        )
        objects.append(edge_pcd)
    
    # 创建基准点云
    if benchmarks:
        benchmark_points = np.asarray(mesh.vertices)[benchmarks]
        benchmark_pcd = o3d.geometry.PointCloud()
        benchmark_pcd.points = o3d.utility.Vector3dVector(benchmark_points)
        benchmark_pcd.colors = o3d.utility.Vector3dVector(
            np.tile([1.0, 0.0, 0.0], (len(benchmark_points), 1))
        )
        objects.append(benchmark_pcd)
    
    # 显示
    o3d.visualization.draw_geometries(
        objects,
        window_name=window_name,
        width=1024,
        height=768
    )


def plot_partition_sizes(partitions: List[Set[int]], output_path: str = "partition_sizes.png"):
    """绘制分区大小分布图"""
    sizes = [len(p) for p in partitions]
    
    plt.figure(figsize=(10, 6))
    plt.hist(sizes, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Region Size (vertices)')
    plt.ylabel('Frequency')
    plt.title('Partition Size Distribution')
    plt.grid(True, alpha=0.3)
    plt.axvline(np.mean(sizes), color='red', linestyle='--', label=f'Mean = {np.mean(sizes):.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved partition size distribution to {output_path}")


def plot_coverage_stats(coverage: np.ndarray, output_path: str = "coverage_stats.png"):
    """绘制覆盖统计"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 覆盖次数分布
    ax1.hist(coverage, bins=range(int(np.max(coverage)) + 2), edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Coverage Count')
    ax1.set_ylabel('Number of Vertices')
    ax1.set_title('Coverage Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 统计信息
    stats_text = f"""
    Statistics:
    Uncovered: {np.sum(coverage == 0)} ({np.mean(coverage == 0):.2%})
    Average: {np.mean(coverage):.2f}
    Max: {np.max(coverage):.0f}
    Total Overlap: {np.sum(coverage - 1):.0f}
    """
    ax2.axis('off')
    ax2.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved coverage statistics to {output_path}")


def visualize_partitions_2d(
    vertices: np.ndarray,
    vertex_to_partitions: Dict[int, List[int]],
    output_path: str = "partitions_2d.png",
    view_type: str = "xy"
):
    """二维投影可视化"""
    # 根据视图类型选择坐标轴
    if view_type == "xy":
        x_idx, y_idx = 0, 1
        x_label, y_label = "X", "Y"
    elif view_type == "xz":
        x_idx, y_idx = 0, 2
        x_label, y_label = "X", "Z"
    elif view_type == "yz":
        x_idx, y_idx = 1, 2
        x_label, y_label = "Y", "Z"
    else:
        x_idx, y_idx = 0, 1
        x_label, y_label = "X", "Y"
    
    # 为每个顶点分配颜色
    num_partitions = max([len(p_list) for p_list in vertex_to_partitions.values()]) + 1 if vertex_to_partitions else 1
    colors = generate_colors(num_partitions)
    vertex_colors = []
    x_coords = []
    y_coords = []
    
    for v in range(len(vertices)):
        p_list = vertex_to_partitions.get(v, [])
        if p_list:
            color = colors[p_list[0] % len(colors)]
        else:
            color = [0.8, 0.8, 0.8]
        vertex_colors.append(color)
        x_coords.append(vertices[v, x_idx])
        y_coords.append(vertices[v, y_idx])
    
    # 绘制
    plt.figure(figsize=(10, 8))
    plt.scatter(x_coords, y_coords, c=vertex_colors, s=10, alpha=0.7)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'2D Projection ({view_type} view)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 2D projection to {output_path}")


def save_complete_visualization(
    mesh: o3d.geometry.TriangleMesh,
    partitions: List[Set[int]],
    vertex_to_partitions: Dict[int, List[int]],
    edge_midpoints: np.ndarray,
    benchmarks: List[int],
    coverage: np.ndarray,
    output_prefix: str,
    interactive: bool = False
):
    """保存完整的可视化结果"""
    import os
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True) if os.path.dirname(output_prefix) else None
    
    # 1. 彩色网格
    mesh_path = f"{output_prefix}_colored_mesh.ply"
    save_colored_mesh_open3d(mesh, vertex_to_partitions, benchmarks, mesh_path)
    
    # 2. 边界中点
    edge_path = f"{output_prefix}_edge_midpoints.ply"
    save_edge_midpoints_open3d(edge_midpoints, edge_path)
    
    # 3. 分区大小分布
    size_path = f"{output_prefix}_partition_sizes.png"
    plot_partition_sizes(partitions, size_path)
    
    # 4. 覆盖统计
    coverage_path = f"{output_prefix}_coverage_stats.png"
    plot_coverage_stats(coverage, coverage_path)
    
    # 5. 二维投影（三个视图）
    vertices = np.asarray(mesh.vertices)
    for view in ["xy", "xz", "yz"]:
        proj_path = f"{output_prefix}_2d_{view}.png"
        visualize_partitions_2d(vertices, vertex_to_partitions, proj_path, view)
    
    # 6. 交互式可视化（可选）
    if interactive:
        visualize_interactive(mesh, vertex_to_partitions, edge_midpoints, benchmarks)
    
    print(f"\nAll visualizations saved to: {os.path.dirname(output_prefix)}")


def create_summary_report(
    name: str,
    metrics: Dict,
    output_path: str = "summary_report.txt"
):
    """创建测试摘要报告"""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
{'=' * 80}
Partition Test Report: {name}
{'=' * 80}

Generated: {timestamp}

{'=' * 40}
1. Coverage Metrics
{'=' * 40}
Uncovered ratio: {metrics.get('uncovered_ratio', np.nan):.2%}
Average coverage: {metrics.get('avg_coverage', np.nan):.2f}
Total overlap: {metrics.get('total_overlap', np.nan):.0f}
Max coverage: {metrics.get('max_coverage', np.nan):.0f}

{'=' * 40}
2. Shape Metrics
{'=' * 40}
Mean anisotropy: {metrics.get('mean_anisotropy', np.nan):.2f}
Median anisotropy: {metrics.get('median_anisotropy', np.nan):.2f}
Max anisotropy: {metrics.get('max_anisotropy', np.nan):.2f}
Boundary straightness: {metrics.get('boundary_straightness', np.nan):.4f}

{'=' * 40}
3. Benchmark Metrics
{'=' * 40}
Number of benchmarks: {metrics.get('num_benchmarks', np.nan):.0f}
Redundant benchmarks: {metrics.get('redundant_ratio', np.nan):.2%}
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Summary report saved to: {output_path}")
