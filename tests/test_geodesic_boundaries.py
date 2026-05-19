"""
直纹面拟合测试脚本
验证：分区 ``List[Set[int]]`` -> 每分区 PCA 分箱构造准线 -> 曲面域样条拟合 -> 直纹面近似
"""

import sys
import os
import warnings
import numpy as np
import argparse
from typing import Dict, List, Tuple, Set
from datetime import datetime

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _configure_runtime_warnings() -> None:
    """将 NURBS 求值中已处理的数值问题从警告升级为可忽略的过滤项。"""
    warnings.filterwarnings(
        'ignore',
        message='divide by zero encountered',
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        'ignore',
        message='invalid value encountered',
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        'ignore',
        category=DeprecationWarning,
        module=r'numpy\.core',
    )


def _configure_stdio_utf8() -> None:
    """在 Windows 终端中尽量使用 UTF-8，减轻中文乱码（仍需终端字体支持 Unicode）。"""
    if sys.platform != 'win32':
        return
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("警告: PyVista不可用，将不进行可视化")

from core.meshProcessor import MeshProcessor
from core.nurbsProcessor import NURBSProcessor
from new import NewPartitioner
from new import LinearSurfaceFitter
from geometry_generators import generate_cylinder, generate_cone, generate_wavy_plane


def create_test_surface(surface_type: str = 'cylinder') -> Tuple[NURBSProcessor, MeshProcessor]:
    """
    创建测试用的NURBS曲面和对应的网格
    
    Args:
        surface_type: 曲面类型 ('cylinder', 'cone', 'wavy')
    
    Returns:
        (nurbs_processor, mesh_processor)
    """
    if surface_type == 'cylinder':
        nurbs = NURBSProcessor.create_cylinder(radius=1.0, height=2.0)
        trimesh_mesh = generate_cylinder()
    elif surface_type == 'cone':
        nurbs = NURBSProcessor.create_cone(radius=1.0, height=2.0)
        trimesh_mesh = generate_cone()
    elif surface_type == 'wavy':
        nurbs = create_freeform_nurbs()
        trimesh_mesh = generate_wavy_plane()
    else:
        raise ValueError(f"未知的曲面类型: {surface_type}")
    
    # 将trimesh转换为open3d格式并包装为MeshProcessor
    import open3d as o3d
    
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    o3d_mesh.compute_vertex_normals()
    
    mesh_processor = MeshProcessor(o3d_mesh)
    
    return nurbs, mesh_processor


def create_freeform_nurbs() -> NURBSProcessor:
    """
    创建自由形式的NURBS曲面用于测试
    
    Returns:
        NURBSProcessor实例
    """
    # 创建一个波浪形的曲面
    degree_u = 3
    degree_v = 3
    
    # 创建控制点网格
    control_points = []
    for i in range(7):
        row = []
        for j in range(7):
            u = (i - 3) / 2.0
            v = (j - 3) / 2.0
            # 构造波浪形曲面
            z = 0.2 * np.sin(u * np.pi) * np.cos(v * np.pi)
            row.append([u, v, z])
        control_points.append(row)
    
    control_points = np.array(control_points)
    
    # 创建节点向量
    knots_u = np.concatenate([
        np.zeros(degree_u),
        np.linspace(0, 1, control_points.shape[0] - degree_u + 1),
        np.ones(degree_u)
    ])
    knots_v = np.concatenate([
        np.zeros(degree_v),
        np.linspace(0, 1, control_points.shape[1] - degree_v + 1),
        np.ones(degree_v)
    ])
    
    return NURBSProcessor(
        control_points=control_points,
        knots_u=knots_u,
        knots_v=knots_v,
        degree_u=degree_u,
        degree_v=degree_v
    )


def visualize_partition_advanced(
    vertices: np.ndarray,
    faces: np.ndarray,
    partitions: List[Set[int]],
    vertex_to_partitions: Dict[int, List[int]],
    benchmarks: List[int],
    coverage: np.ndarray,
    screenshot_path: str = None,
    interactive: bool = False,
    multi_view_screenshots: bool = False,
):
    """
    可视化分区结果（高级版本）
    
    Args:
        vertices: 顶点数组
        faces: 面索引数组
        partitions: 分区列表
        vertex_to_partitions: 顶点到分区映射
        benchmarks: 基准点列表
        coverage: 覆盖次数数组
        screenshot_path: 截图保存路径
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista不可用，跳过分区可视化")
        return
    
    plotter = pv.Plotter(off_screen=not interactive)
    
    # 创建网格
    mesh = pv.PolyData(vertices, np.hstack([np.full((len(faces), 1), 3), faces]))
    
    # 添加覆盖热度图
    mesh['coverage'] = coverage
    plotter.add_mesh(
        mesh,
        scalars='coverage',
        cmap='plasma',
        opacity=0.7,
        show_edges=False
    )
    
    # 添加分区边界（仅用顶点索引，避免对每条边做全表 .index）
    boundary_edge_pairs = set()
    for i in range(len(faces)):
        for j in range(3):
            v0 = int(faces[i, j])
            v1 = int(faces[i, (j + 1) % 3])
            p0 = vertex_to_partitions.get(v0, [])
            p1 = vertex_to_partitions.get(v1, [])
            if set(p0) != set(p1):
                a, b = (v0, v1) if v0 < v1 else (v1, v0)
                boundary_edge_pairs.add((a, b))

    if boundary_edge_pairs:
        line_cells = []
        for a, b in boundary_edge_pairs:
            line_cells.extend([2, a, b])
        edge_mesh = pv.PolyData(vertices, lines=np.array(line_cells, dtype=np.int64))
        plotter.add_mesh(edge_mesh, color='red', line_width=3)

    # 添加基准点
    if benchmarks:
        benchmark_pts = vertices[benchmarks]
        plotter.add_mesh(
            pv.PolyData(benchmark_pts),
            color='yellow',
            point_size=10,
            render_points_as_spheres=True
        )

    plotter.add_axes()
    plotter.add_title("分区结果 (覆盖热力图 + 边界线)", font_size=12)

    if multi_view_screenshots and screenshot_path:
        import os as _os

        root, ext = _os.path.splitext(screenshot_path)
        if not ext:
            ext = ".png"
        plotter.reset_camera()
        for label, meth in (
            ("iso", "view_isometric"),
            ("xy", "view_xy"),
            ("xz", "view_xz"),
            ("yz", "view_yz"),
        ):
            try:
                fn = getattr(plotter, meth, None)
                if callable(fn):
                    fn()
                    out = f"{root}_{label}{ext}"
                    plotter.screenshot(out)
                    print(f"分区多视角截图: {out}", flush=True)
            except Exception as _e:
                print(f"  视角 {label} 截图失败: {_e}", flush=True)
    elif screenshot_path:
        plotter.screenshot(screenshot_path)
        print(f"分区截图已保存: {screenshot_path}", flush=True)

    if interactive:
        print("交互式分区视图：关闭窗口后继续。", flush=True)
        plotter.show()
    plotter.close()


def visualize_ruled_patches(
    mesh_processor: MeshProcessor,
    fitter: LinearSurfaceFitter,
    patches: List[Dict],
    degree: int,
    screenshot_path: str = None,
    interactive: bool = False,
    multi_view_screenshots: bool = False,
):
    """
    可视化每个分区拟合的直纹面：半透明 StructuredGrid 曲面片 + 稀疏母线 + 准线折线。
    interactive=True 时弹出可旋转的 3D 窗口；multi_view_screenshots=True 时在无交互模式下保存多视角 PNG。
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista不可用，跳过直纹面可视化")
        return

    plotter = pv.Plotter(off_screen=not interactive)

    vertices = np.array(mesh_processor.vertices)
    faces = np.array(mesh_processor.faces)
    mesh = pv.PolyData(vertices, np.hstack([np.full((len(faces), 1), 3), faces]))
    plotter.add_mesh(mesh, opacity=0.45, color='lightgray', style='surface')

    cmap = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]

    for i, patch in enumerate(patches):
        d0 = patch['directrix0']
        d1 = patch['directrix1']
        if len(d0.get('control_points', [])) < 2 or len(d1.get('control_points', [])) < 2:
            continue
        color = cmap[i % len(cmap)]

        try:
            nu, nv = 28, 14
            grid_xyz = fitter.sample_ruled_surface_grid(d0, d1, degree, nu=nu, nv=nv)
            surf = pv.StructuredGrid(
                grid_xyz[:, :, 0],
                grid_xyz[:, :, 1],
                grid_xyz[:, :, 2],
            )
            plotter.add_mesh(
                surf,
                color=color,
                opacity=0.38,
                smooth_shading=True,
                show_edges=False,
            )

            ns = 40
            pts0 = fitter.sample_geodesic(d0['control_points'], d0['knots'], degree, num_samples=ns)
            pts1 = fitter.sample_geodesic(d1['control_points'], d1['knots'], degree, num_samples=ns)
            for pts, c in ((pts0, color), (pts1, 'orange')):
                if len(pts) >= 2:
                    plotter.add_mesh(pv.lines_from_points(pts, close=False), color=c, line_width=2)

            for k in range(0, ns, 6):
                u = k / (ns - 1) if ns > 1 else 0.5
                p_a = fitter.evaluate_ruled_surface(d0, d1, u, 0.0)
                p_b = fitter.evaluate_ruled_surface(d0, d1, u, 1.0)
                plotter.add_mesh(pv.Line(p_a, p_b), color='cyan', line_width=1)
        except Exception as e:
            print(f"绘制分区 {i} 直纹面时出错: {e}", flush=True)

    plotter.add_axes()
    plotter.add_title("Ruled surface patches (StructuredGrid)", font_size=12)

    if multi_view_screenshots and screenshot_path:
        import os as _os

        root, ext = _os.path.splitext(screenshot_path)
        if not ext:
            ext = ".png"
        plotter.reset_camera()
        _views = [
            ("iso", "view_isometric"),
            ("xy", "view_xy"),
            ("xz", "view_xz"),
            ("yz", "view_yz"),
        ]
        for label, meth in _views:
            try:
                fn = getattr(plotter, meth, None)
                if callable(fn):
                    fn()
                    out = f"{root}_{label}{ext}"
                    plotter.screenshot(out)
                    print(f"直纹面多视角截图: {out}", flush=True)
            except Exception as _e:
                print(f"  视角 {label} 截图失败: {_e}", flush=True)
    elif screenshot_path:
        plotter.screenshot(screenshot_path)
        print(f"直纹面截图已保存: {screenshot_path}", flush=True)

    if interactive:
        print("交互式直纹面视图：关闭窗口后继续。", flush=True)
        plotter.show()
    plotter.close()


def print_statistics(result: Dict, surface_type: str):
    """
    打印直纹面拟合质量统计信息

    Args:
        result: ``fit_ruled_surfaces`` 的返回结果
        surface_type: 曲面类型
    """
    print(f"\n{'='*60}")
    print(f"测试曲面: {surface_type}")
    print(f"{'='*60}")
    print(f"分区片数量: {result['num_patches']}")
    print(f"整体准线平均拟合误差: {result['overall_mean_directrix_fit_error']:.8f}")
    print(f"整体直纹面近似 RMS（顶点到采样片）: {result['overall_ruled_approximation_rms']:.8f}")
    print(f"{'='*60}")

    patches = result.get('patches', [])
    show = min(10, len(patches))
    print(f"\n单分区质量（前{show}片）:")
    for i, p in enumerate(patches[:show]):
        print(f"  分区 {p.get('partition_id', i)}:")
        print(f"    顶点数: {p.get('num_vertices', 0)}")
        print(f"    准线平均拟合误差: {p.get('mean_directrix_fit_error', 0):.6f}")
        print(f"    直纹面近似 RMS: {p.get('ruled_approximation_rms', 0):.6f}")


def test_geodesic_boundaries(
    surface_type: str = 'cylinder',
    output_dir: str = 'test_output',
    interactive_viz: bool = False,
    multi_view_screenshots: bool = False,
):
    """
    执行完整的分区直纹面拟合测试
    
    Args:
        surface_type: 曲面类型 ('cylinder', 'cone', 'wavy')
        output_dir: 输出目录
    """
    _configure_stdio_utf8()
    _configure_runtime_warnings()
    print(f"\n{'='*80}")
    print(f"开始测试: {surface_type} 曲面（每分区直纹面）")
    print(f"{'='*80}\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 创建NURBS曲面与网格
    print("步骤 1/6: 创建NURBS曲面与网格...")
    nurbs, mesh_processor = create_test_surface(surface_type)
    print(f"  网格顶点数: {len(mesh_processor.vertices)}")
    print(f"  网格面数: {len(mesh_processor.faces)}")
    
    # 2. 分区流程
    print("\n步骤 2/6: 执行分区流程...")
    partitioner = NewPartitioner(mesh_processor)
    
    # 计算平均边长；分区半径与属性阈值与 newIndicator.grow_region 一致
    edges = np.array(mesh_processor.edge_vertices)
    avg_edge_length = np.mean(np.linalg.norm(
        mesh_processor.vertices[edges[:, 0]] - mesh_processor.vertices[edges[:, 1]],
        axis=1
    ))
    # 更大 R_max、更宽松法向阈值 -> 单区沿曲面可延伸更远，减少“够不着”的未覆盖点；
    # 初始基准点略少，避免与过大区域叠加后重叠爆炸；优化仍限制在 20 轮。
    R_max = 48.0 * avg_edge_length
    theta_attr = 58.0
    partition_alpha = 0.8

    (optimized_benchmarks, regions_dict, final_coverage,
     vertex_to_partitions, edge_midpoints, _) = partitioner.partition_with_optimization(
        initial_num_benchmarks=10,
        alpha=partition_alpha,
        R_max=R_max,
        theta_attr=theta_attr,
        max_iterations=20,
    )
    
    print(f"  优化后基准点数: {len(optimized_benchmarks)}")
    print(f"  分区数: {len(regions_dict)}")
    
    # 3. 直纹面拟合（按分区 ``List[Set[int]]``，与 newPartitoner 输出一致）
    print("\n步骤 3/6: 拟合各分区直纹面...")
    fitter = LinearSurfaceFitter(mesh_processor, nurbs)

    partitions_ordered = [regions_dict[b] for b in optimized_benchmarks]
    degree = 3
    result = fitter.fit_ruled_surfaces(
        partitions_ordered,
        num_control_points=None,
        degree=degree,
        mu=0.35,
        max_iterations=30,
        num_bins=None,
        verbose=True,
    )

    # 4. 可视化分区结果
    print("\n步骤 4/6: 可视化分区结果...")
    vertices = np.array(mesh_processor.vertices)
    faces = np.array(mesh_processor.faces)
    partitions = list(regions_dict.values())

    partition_screenshot = os.path.join(output_dir, f'ruled_partition_result_{surface_type}.png')
    visualize_partition_advanced(
        vertices, faces, partitions, vertex_to_partitions,
        optimized_benchmarks, final_coverage,
        screenshot_path=partition_screenshot,
        interactive=interactive_viz,
        multi_view_screenshots=multi_view_screenshots,
    )
    
    # 5. 可视化直纹面
    print("\n步骤 5/6: 可视化直纹面...")
    curve_screenshot = os.path.join(output_dir, f'ruled_fitted_patches_{surface_type}.png')
    visualize_ruled_patches(
        mesh_processor, fitter, result['patches'], degree,
        screenshot_path=curve_screenshot,
        interactive=interactive_viz,
        multi_view_screenshots=multi_view_screenshots,
    )

    # 6. 输出统计信息
    print("\n步骤 6/6: 输出统计信息...")
    print_statistics(result, surface_type)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='直纹面拟合测试脚本（按分区点云）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python tests/test_geodesic_boundaries.py --surface cylinder
  python tests/test_geodesic_boundaries.py --surface all --output_dir results
  python tests/test_geodesic_boundaries.py --surface cylinder --interactive
  python tests/test_geodesic_boundaries.py --surface cylinder --multi-view
"""
    )
    parser.add_argument(
        '--surface', type=str, default='cylinder',
        choices=['cylinder', 'cone', 'wavy', 'all'],
        help='要测试的曲面类型 (默认: cylinder)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='test_output',
        help='输出目录 (默认: test_output)'
    )
    parser.add_argument(
        '--interactive', action='store_true',
        help='弹出 PyVista 交互窗口（可旋转缩放），关闭后继续'
    )
    parser.add_argument(
        '--multi-view', action='store_true', dest='multi_view',
        help='除主截图外保存 iso/xy/xz/yz 多视角 PNG（仍使用离屏渲染）'
    )
    
    args = parser.parse_args()
    
    surfaces_to_test = []
    if args.surface == 'all':
        surfaces_to_test = ['cylinder', 'cone', 'wavy']
    else:
        surfaces_to_test = [args.surface]
    
    results = {}
    for surface in surfaces_to_test:
        try:
            results[surface] = test_geodesic_boundaries(
                surface, args.output_dir,
                interactive_viz=args.interactive,
                multi_view_screenshots=args.multi_view,
            )
        except Exception as e:
            print(f"\n测试 {surface} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("所有测试完成！")
        print(f"{'='*80}")
        for surface, result in results.items():
            print(f"\n{surface} 结果:")
            print(f"  准线误差: {result['overall_mean_directrix_fit_error']:.8f}")
            print(f"  直纹面 RMS: {result['overall_ruled_approximation_rms']:.8f}")


if __name__ == '__main__':
    _configure_stdio_utf8()
    _configure_runtime_warnings()
    main()