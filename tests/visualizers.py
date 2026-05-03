
import numpy as np
import trimesh
from typing import List, Set, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_distinct_colors(num_colors: int) -> np.ndarray:
    """
    生成一组明显不同的颜色
    
    Args:
        num_colors: 颜色数量
        
    Returns:
        RGB颜色数组
    """
    colors = plt.cm.hsv(np.linspace(0, 1, num_colors, endpoint=False))
    return colors[:, :3]


def save_colored_partition_mesh(
    mesh,
    vertex_to_partitions: Dict[int, List[int]],
    benchmarks: List[int],
    output_path: str
):
    """
    保存带分区颜色的网格
    
    Args:
        mesh: 原始网格（trimesh.Trimesh 或 open3d.geometry.TriangleMesh）
        vertex_to_partitions: 顶点到分区的映射
        benchmarks: 基准点列表
        output_path: 输出文件路径
    """
    import open3d as o3d
    
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        num_vertices = len(vertices)
        
        colored_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(faces)
        )
    else:
        vertices = mesh.vertices
        num_vertices = len(vertices)
        colored_mesh = mesh.copy()
    
    vertex_colors = np.ones((num_vertices, 3)) * 0.8
    
    num_partitions = len(set([p for plist in vertex_to_partitions.values() for p in plist])) if vertex_to_partitions else 0
    if num_partitions > 0:
        colors = generate_distinct_colors(num_partitions)
        
        for v in range(num_vertices):
            partitions = vertex_to_partitions.get(v, [])
            if partitions:
                main_partition = partitions[0]
                vertex_colors[v] = colors[main_partition % len(colors)]
    
    if benchmarks:
        benchmark_set = set(benchmarks)
        for b in benchmark_set:
            if b < len(vertex_colors):
                vertex_colors[b] = [1.0, 0.0, 0.0]
    
    if isinstance(colored_mesh, o3d.geometry.TriangleMesh):
        colored_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        o3d.io.write_triangle_mesh(output_path, colored_mesh)
    else:
        colored_mesh.visual.vertex_colors = (vertex_colors * 255).astype(np.uint8)
        colored_mesh.export(output_path)
    
    print(f"Saved colored mesh to {output_path}")


def save_edge_midpoints(
    edge_midpoints: np.ndarray,
    output_path: str,
    color: List[float] = [0.0, 1.0, 0.0]
):
    """
    保存边界中点为点云
    
    Args:
        edge_midpoints: 边界中点数组
        output_path: 输出文件路径
        color: 点云颜色
    """
    if len(edge_midpoints) == 0:
        return
    
    pcd = trimesh.PointCloud(edge_midpoints)
    pcd.colors = np.tile(color, (len(edge_midpoints), 1))
    pcd.export(output_path)
    print(f"Saved edge midpoints to {output_path}")


def save_visualization(
    mesh: trimesh.Trimesh,
    vertex_to_partitions: Dict[int, List[int]],
    edge_midpoints: np.ndarray,
    benchmarks: List[int],
    output_prefix: str
):
    """
    保存完整的可视化结果
    
    Args:
        mesh: 原始网格
        vertex_to_partitions: 顶点到分区的映射
        edge_midpoints: 边界中点
        benchmarks: 基准点列表
        output_prefix: 输出文件前缀
    """
    mesh_path = f"{output_prefix}_colored_mesh.ply"
    edge_path = f"{output_prefix}_edge_midpoints.ply"
    
    save_colored_partition_mesh(mesh, vertex_to_partitions, benchmarks, mesh_path)
    save_edge_midpoints(edge_midpoints, edge_path)


def plot_convergence_curve(
    iteration_data: List[Dict],
    output_path: str = "convergence_curve.png"
):
    """
    绘制收敛曲线
    
    Args:
        iteration_data: 迭代数据列表
        output_path: 输出文件路径
    """
    iterations = [d["iteration"] for d in iteration_data]
    total_overlaps = [d["total_overlap"] for d in iteration_data]
    num_benchmarks_list = [d["num_benchmarks"] for d in iteration_data]
    uncovered_list = [d["uncovered"] for d in iteration_data]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(iterations, total_overlaps, 'b-', linewidth=2, label='Total Overlap')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Overlap', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax1_right = ax1.twinx()
    ax1_right.plot(iterations, num_benchmarks_list, 'r-', linewidth=2, label='Num Benchmarks')
    ax1_right.set_ylabel('Number of Benchmarks', color='r')
    ax1_right.tick_params(axis='y', labelcolor='r')
    
    ax2.plot(iterations, uncovered_list, 'g-', linewidth=2, label='Uncovered Vertices')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Uncovered Vertices', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence curve to {output_path}")


def plot_parameter_sensitivity(
    parameter_name: str,
    parameter_values: List,
    metric_name: str,
    metric_values: List,
    output_path: str = "parameter_sensitivity.png"
):
    """
    绘制参数敏感性曲线
    
    Args:
        parameter_name: 参数名称
        parameter_values: 参数值列表
        metric_name: 指标名称
        metric_values: 指标值列表
        output_path: 输出文件路径
    """
    plt.figure(figsize=(8, 6))
    plt.plot(parameter_values, metric_values, 'o-', linewidth=2, markersize=8)
    plt.xlabel(parameter_name, fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'{metric_name} vs {parameter_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved parameter sensitivity plot to {output_path}")
