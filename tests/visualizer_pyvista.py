"""
基于 PyVista 的增强可视化模块 - 优化版
- 覆盖次数热力图底色
- 彩色分区边界线
- 性能优化：边界提取和渲染

修复：
1. 修复 mesh.lines 解析错误（正确使用 reshape(-1,3)[:,1:]）
2. 优化边界提取（一次全局扫描）
3. 优化边界渲染（合并为 PolyData）
4. 热力图支持 0 值表示未覆盖
"""
import numpy as np
from typing import List, Set, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not available. Install with: pip install pyvista")


def extract_boundaries_efficient(mesh, vertex_to_partitions, num_partitions):
    """
    高效的边界提取算法 - 一次全局扫描

    Args:
        mesh: PyVista 网格对象
        vertex_to_partitions: 顶点 -> 分区列表映射 {v: [pid1, pid2, ...]}
        num_partitions: 分区总数

    Returns:
        boundary_edges: {pid: np.array(shape(n, 2, 3))} 每个分区的边界线段
    """
    # 步骤1: 构建每个顶点所属分区的集合（便于快速判断）
    vertex_partition_sets = [set() for _ in range(mesh.n_points)]
    for v, pids in vertex_to_partitions.items():
        vertex_partition_sets[v] = set(pids)

    # 步骤2: 正确获取和解析所有边
    edge_mesh = mesh.extract_all_edges()
    lines_arr = edge_mesh.lines

    # PyVista lines 格式: [n0, idx0, idx1, n1, idx2, idx3, ...]
    # 对于线段，每个都是 [2, start, end]
    if len(lines_arr) % 3 != 0:
        raise ValueError("Invalid lines array format for edges")

    edge_indices = lines_arr.reshape(-1, 3)[:, 1:]  # shape (num_edges, 2)

    # 步骤3: 一次全局扫描，找出所有边界边
    boundary_edges = {pid: [] for pid in range(num_partitions)}

    for v1, v2 in edge_indices:
        set1 = vertex_partition_sets[v1]
        set2 = vertex_partition_sets[v2]

        # 找出只在一个端点的分区
        only_in_v1 = set1 - set2
        only_in_v2 = set2 - set1

        for pid in only_in_v1:
            boundary_edges[pid].append((v1, v2))
        for pid in only_in_v2:
            boundary_edges[pid].append((v1, v2))

    # 步骤4: 转换回世界坐标
    vertices = mesh.points
    result = {}

    for pid, edges in boundary_edges.items():
        if len(edges) == 0:
            result[pid] = np.array([])
            continue

        segments = []
        for v1, v2 in edges:
            segments.append([vertices[v1], vertices[v2]])
        result[pid] = np.array(segments)

    return result


def add_partition_boundaries(plotter, boundary_edges, colors):
    """
    高效添加分区边界 - 每个分区合并为单个 PolyData 对象

    Args:
        plotter: PyVista Plotter 对象
        boundary_edges: {pid: np.array(shape(n,2,3))}
        colors: 颜色列表 [(r,g,b,a), ...]
    """
    for pid, edges in boundary_edges.items():
        if len(edges) == 0:
            continue

        # 将所有边合并为单个 PolyData
        pts = []
        lines = []
        offset = 0

        for edge in edges:
            p1, p2 = edge[0], edge[1]
            pts.extend([p1, p2])
            lines.append([2, offset, offset + 1])
            offset += 2

        pts = np.array(pts)
        line_data = np.hstack(lines)

        line_mesh = pv.PolyData(pts, line_data)
        color = colors[pid % len(colors)][:3]
        plotter.add_mesh(line_mesh, color=color, line_width=2.5, opacity=0.9)


def generate_colors(num_colors):
    """
    生成分区边界颜色

    Args:
        num_colors: 分区数量

    Returns:
        颜色列表 [(r,g,b,a), ...]
    """
    if num_colors <= 10:
        colors = [
            (1.0, 0.0, 0.0, 1.0),  # 红
            (0.0, 0.0, 1.0, 1.0),  # 蓝
            (0.0, 0.8, 0.0, 1.0),  # 绿
            (1.0, 0.8, 0.0, 1.0),  # 黄
            (0.8, 0.0, 1.0, 1.0),  # 紫
            (0.0, 1.0, 1.0, 1.0),  # 青
            (1.0, 0.0, 0.5, 1.0),  # 粉
            (0.5, 0.5, 0.5, 1.0),  # 灰
            (1.0, 0.5, 0.0, 1.0),  # 橙
            (0.0, 0.5, 1.0, 1.0),  # 浅蓝
        ]
        return colors[:num_colors]
    else:
        # 使用 HSV 色彩空间生成更多颜色
        colors = []
        for i in range(num_colors):
            hue = (i / num_colors) % 1.0
            # HSV to RGB (saturation=0.8, value=1.0)
            c = 1.0
            x = c * (1 - abs((hue * 6) % 2 - 1))
            if 0 <= hue < 1/6:
                r, g, b = c, x, 0
            elif 1/6 <= hue < 2/6:
                r, g, b = x, c, 0
            elif 2/6 <= hue < 3/6:
                r, g, b = 0, c, x
            elif 3/6 <= hue < 4/6:
                r, g, b = 0, x, c
            elif 4/6 <= hue < 5/6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            colors.append((r, g, b, 0.9))
        return colors


def visualize_boundaries_only(
    mesh,
    partitions: List[Set[int]],
    vertex_to_partitions: Dict[int, List[int]],
    benchmarks: List[int],
    window_name: str = "Partition Boundaries Only"
):
    """
    只显示分区边界的可视化窗口（无热力图）

    Args:
        mesh: PyVista PolyData 网格对象
        partitions: 分区列表
        vertex_to_partitions: 顶点 -> 分区映射
        benchmarks: 基准点索引
        window_name: 窗口标题
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available. Skipping boundary visualization.")
        return

    # 创建 plotter
    plotter = pv.Plotter(window_size=[1280, 720])
    plotter.set_background("white")

    # 添加网格（透明显示，只作为背景参考）
    plotter.add_mesh(
        mesh,
        color="lightgray",
        opacity=0.3,
        show_edges=False,
        lighting=True
    )

    # 提取边界
    num_partitions = len(partitions)
    boundary_edges = extract_boundaries_efficient(mesh, vertex_to_partitions, num_partitions)
    colors = generate_colors(num_partitions)

    # 添加分区边界（每个分区用不同颜色）
    for pid, edges in boundary_edges.items():
        if len(edges) == 0:
            continue

        pts = []
        lines = []
        offset = 0

        for edge in edges:
            p1, p2 = edge[0], edge[1]
            pts.extend([p1, p2])
            lines.append([2, offset, offset + 1])
            offset += 2

        pts = np.array(pts)
        line_data = np.hstack(lines)
        line_mesh = pv.PolyData(pts, line_data)

        color = colors[pid % len(colors)][:3]
        plotter.add_mesh(
            line_mesh,
            color=color,
            line_width=3.0,
            opacity=0.95
        )

    # 添加基准点
    if benchmarks and len(benchmarks) > 0:
        benchmark_points = mesh.points[benchmarks]
        benchmark_cloud = pv.PolyData(benchmark_points)
        plotter.add_mesh(
            benchmark_cloud,
            color=(1.0, 0.0, 0.0),
            render_points_as_spheres=True,
            point_size=10.0
        )

    # 添加坐标轴
    plotter.add_axes()

    # 设置视角
    plotter.camera_position = 'iso'

    # 显示窗口（阻塞模式，关闭后返回）
    plotter.show(window_name)


def visualize_with_pyvista(
    mesh,
    partitions: List[Set[int]],
    vertex_to_partitions: Dict[int, List[int]],
    edge_midpoints: np.ndarray,
    benchmarks: List[int],
    coverage: np.ndarray,
    window_name: str = "Partition Visualization",
    screenshot_path: Optional[str] = None,
    html_path: Optional[str] = None,
    interactive: bool = True
):
    """
    使用 PyVista 可视化分区结果 - 优化版

    Args:
        mesh: PyVista PolyData 网格对象
        partitions: 分区列表
        vertex_to_partitions: 顶点 -> 分区映射
        edge_midpoints: 边界边中点
        benchmarks: 基准点索引
        coverage: 每个顶点的覆盖次数数组
        window_name: 窗口标题
        screenshot_path: 截图保存路径
        html_path: HTML 导出路径
        interactive: 是否启用交互
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available. Skipping visualization.")
        return

    # 创建 plotter
    plotter = pv.Plotter(window_size=[1280, 720], off_screen=not interactive)
    plotter.set_background("white")

    # 步骤1: 添加热力图底色（覆盖次数，保留 0 值表示未覆盖）
    mesh.point_data["coverage"] = coverage

    # 使用 plasma 颜色映射 + 自定义范围
    plotter.add_mesh(
        mesh,
        scalars="coverage",
        cmap="plasma",
        lighting=True,
        show_edges=False,
        opacity=1.0,
        clim=[0, np.max(coverage)] if np.max(coverage) > 0 else None
    )

    # 步骤2: 高效提取和添加边界
    num_partitions = len(partitions)
    print(f"Extracting partition boundaries (optimized)...")
    boundary_edges = extract_boundaries_efficient(mesh, vertex_to_partitions, num_partitions)
    colors = generate_colors(num_partitions)

    print(f"Adding partition boundaries (optimized)...")
    add_partition_boundaries(plotter, boundary_edges, colors)
    print(f"Added {len([e for e in boundary_edges.values() if len(e) > 0])} partition boundaries")

    # 步骤3: 添加基准点
    if benchmarks and len(benchmarks) > 0:
        benchmark_points = mesh.points[benchmarks]
        benchmark_cloud = pv.PolyData(benchmark_points)
        plotter.add_mesh(
            benchmark_cloud,
            color=(1.0, 0.0, 0.0),
            render_points_as_spheres=True,
            point_size=8.0
        )

    # 步骤4: 添加坐标轴
    plotter.add_axes()

    # 步骤5: 设置视角
    plotter.camera_position = 'iso'

    # 保存截图
    if screenshot_path:
        plotter.show(screenshot=screenshot_path, auto_close=False)
        print(f"Screenshot saved to {screenshot_path}")

    # 导出 HTML（可选，需要额外依赖）
    if html_path:
        try:
            plotter.export_html(html_path)
            print(f"HTML export saved to {html_path}")
        except ImportError:
            print("Skipping HTML export: trame_vtk not available. Install with: pip install 'pyvista[jupyter]'")

    # 交互显示
    if interactive:
        plotter.show(window_name, auto_close=False)

    # 清理资源
    plotter.close()


def create_pyvista_mesh(vertices: np.ndarray, faces: np.ndarray):
    """
    从顶点和面创建 PyVista 网格

    Args:
        vertices: 顶点坐标数组 (n,3)
        faces: 面索引数组 (m,3)

    Returns:
        PyVista PolyData 对象
    """
    if not PYVISTA_AVAILABLE:
        return None

    # PyVista 格式: [3, v0, v1, v2, 3, v0, v1, v2, ...]
    n_faces = len(faces)
    pv_faces = np.empty((n_faces, 4), dtype=np.int32)
    pv_faces[:, 0] = 3
    pv_faces[:, 1:] = faces
    pv_faces = pv_faces.flatten()

    mesh = pv.PolyData(vertices, pv_faces)
    return mesh


def visualize_partition_advanced(
    vertices: np.ndarray,
    faces: np.ndarray,
    partitions: List[Set[int]],
    vertex_to_partitions: Dict[int, List[int]],
    edge_midpoints: np.ndarray,
    benchmarks: List[int],
    coverage: np.ndarray,
    window_name: str = "Partition Visualization",
    screenshot_path: Optional[str] = None,
    interactive: bool = True,
    show_boundaries_only: bool = True
):
    """
    高级分区可视化（热力图 + 彩色边界）- 优化版

    Args:
        vertices: 顶点坐标
        faces: 面索引
        partitions: 分区列表
        vertex_to_partitions: 顶点到分区映射
        edge_midpoints: 边界边中点
        benchmarks: 基准点索引
        coverage: 覆盖次数数组
        window_name: 窗口标题
        screenshot_path: 截图路径
        interactive: 是否交互
        show_boundaries_only: 是否显示第二个边界专用窗口
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available. Please install: pip install pyvista")
        return

    mesh = create_pyvista_mesh(vertices, faces)

    if mesh is None:
        return

    html_path = None
    if screenshot_path:
        html_path = screenshot_path.replace("_heatmap.png", "_interactive.html")

    # 第一个窗口：热力图 + 边界
    print(f"\n=== 显示窗口 1: {window_name} (热力图 + 边界) ===")
    print("按 Q 关闭窗口后显示边界专用窗口")
    visualize_with_pyvista(
        mesh=mesh,
        partitions=partitions,
        vertex_to_partitions=vertex_to_partitions,
        edge_midpoints=edge_midpoints,
        benchmarks=benchmarks,
        coverage=coverage,
        window_name=window_name,
        screenshot_path=screenshot_path,
        html_path=html_path,
        interactive=interactive
    )

    # 第二个窗口：只显示边界（不同分区用不同颜色）
    if show_boundaries_only and interactive:
        # 创建新的 mesh 副本，避免与第一个窗口共享数据
        mesh_boundary = create_pyvista_mesh(vertices, faces)
        
        print(f"\n=== 显示窗口 2: 分区边界 (彩色) ===")
        print("每个分区的边界用不同颜色显示")
        visualize_boundaries_only(
            mesh=mesh_boundary,
            partitions=partitions,
            vertex_to_partitions=vertex_to_partitions,
            benchmarks=benchmarks,
            window_name="Partition Boundaries (Colored)"
        )


def save_pyvista_visualization(
    vertices: np.ndarray,
    faces: np.ndarray,
    partitions: List[Set[int]],
    vertex_to_partitions: Dict[int, List[int]],
    edge_midpoints: np.ndarray,
    benchmarks: List[int],
    coverage: np.ndarray,
    output_prefix: str,
    interactive: bool = False,
    show_boundaries_only: bool = True
):
    """
    保存 PyVista 可视化结果 - 优化版

    Args:
        vertices: 顶点坐标
        faces: 面索引
        partitions: 分区列表
        vertex_to_partitions: 顶点到分区映射
        edge_midpoints: 边界边中点
        benchmarks: 基准点索引
        coverage: 覆盖次数
        output_prefix: 输出文件前缀
        interactive: 是否显示交互窗口
        show_boundaries_only: 是否显示边界专用窗口
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available. Skipping advanced visualization.")
        return

    screenshot_path = f"{output_prefix}_heatmap.png"
    html_path = f"{output_prefix}_interactive.html"

    visualize_partition_advanced(
        vertices=vertices,
        faces=faces,
        partitions=partitions,
        vertex_to_partitions=vertex_to_partitions,
        edge_midpoints=edge_midpoints,
        benchmarks=benchmarks,
        coverage=coverage,
        window_name=f"Partition: {output_prefix}",
        screenshot_path=screenshot_path,
        interactive=interactive,
        show_boundaries_only=show_boundaries_only
    )

    print(f"Advanced visualization saved to:")
    print(f"  - {screenshot_path}")
    print(f"  - {html_path}")


if __name__ == "__main__":
    print("PyVista Visualization Module (Optimized)")
    print("=" * 60)
    print(f"PyVista available: {PYVISTA_AVAILABLE}")
    if PYVISTA_AVAILABLE:
        print(f"PyVista version: {pv.__version__}")
    print("\nOptimizations:")
    print("  - Edge extraction: one global scan (O(n) instead of O(kn))")
    print("  - Edge rendering: single PolyData per partition (fast!)")
    print("  - Heatmap: supports 0 coverage for uncovered vertices")
    print("  - Lines parsing: fixed reshape logic")
    print("\nUsage:")
    print("  from tests.visualizer_pyvista import save_pyvista_visualization")
    print("  save_pyvista_visualization(vertices, faces, ...)")
