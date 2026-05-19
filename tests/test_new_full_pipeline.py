"""
简化的分区测试脚本 - PyVista 增强可视化
- 覆盖次数热力图底色
- 彩色分区边界线
"""
import sys
import os
import time
import numpy as np
import argparse
from typing import Dict, List
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meshProcessor import MeshProcessor
from new import NewPartitioner
from new import BasePointInitializer

from geometry_generators import generate_cylinder, generate_cone, generate_wavy_plane
from tests.metrics import evaluate_full_partition, print_metrics_summary
from tests.visualizers import plot_convergence_curve, plot_parameter_sensitivity
from tests.visualizer_enhanced import save_complete_visualization, create_summary_report, visualize_boundaries_open3d
from tests.visualizer_pyvista import save_pyvista_visualization, PYVISTA_AVAILABLE


def wrap_mesh_with_processor(trimesh_mesh) -> MeshProcessor:
    """
    将 trimesh 网格包装为 MeshProcessor 对象
    """
    import open3d as o3d
    
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.compute_triangle_normals()
    
    mesh_processor = MeshProcessor(o3d_mesh)
    return mesh_processor


def test_single_shape(
    name: str,
    mesh_processor: MeshProcessor,
    alpha: float = 0.0,
    R_max: float = None,
    theta_attr: float = 30.0,
    initial_num_benchmarks: int = 20,
    sampling_method: str = 'uniform',
    max_iterations: int = 50,
    output_dir: str = 'test_output',
    auto_view: bool = True,
    use_pyvista: bool = True
) -> Dict:
    """
    测试单个形状的完整流程
    
    Args:
        name: 形状名称
        mesh_processor: MeshProcessor 对象
        alpha: 曲率拉伸强度
        R_max: 最大有效半径
        theta_attr: 属性差异阈值
        initial_num_benchmarks: 初始基准点数量
        sampling_method: 采样方法
        max_iterations: 最大迭代次数
        output_dir: 输出目录
        auto_view: 是否在完成后自动打开可视化
        use_pyvista: 是否使用 PyVista 增强可视化
        
    Returns:
        结果字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'=' * 80}")
    print(f"Testing shape: {name}")
    print(f"{'=' * 80}")
    
    results = {"shape": name}
    start_time = time.time()
    
    partitioner = NewPartitioner(mesh_processor)
    
    print(f"\n--- Running optimization pipeline ---")
    print(f"Initial benchmarks: {initial_num_benchmarks} ({sampling_method} sampling)")
    print(f"Parameters: alpha={alpha}, theta_attr={theta_attr}")
    print(f"Max iterations: {max_iterations}")
    
    # 执行优化流程
    opt_benchmarks, regions_dict_opt, final_coverage, vertex_to_partitions_opt, edge_midpoints_opt, iteration_data = \
        partitioner.partition_with_optimization(
            initial_num_benchmarks=initial_num_benchmarks,
            sampling_method=sampling_method,
            alpha=alpha, R_max=R_max, theta_attr=theta_attr,
            max_iterations=max_iterations
        )
    
    print(f"\n--- Evaluation ---")
    
    # 绘制收敛曲线
    if iteration_data:
        plot_convergence_curve(
            iteration_data,
            os.path.join(output_dir, f"{name}_convergence.png")
        )
    
    # 评估指标
    metrics_opt = evaluate_full_partition(
        opt_benchmarks, regions_dict_opt, final_coverage,
        mesh_processor.vertices, edge_midpoints_opt
    )
    print_metrics_summary(metrics_opt, "Optimized Results")
    results["optimized"] = metrics_opt
    
    # 保存 Open3D 可视化结果（PLY 文件）
    print(f"\n--- Saving Open3D visualizations ---")
    save_complete_visualization(
        mesh_processor.mesh,
        list(regions_dict_opt.values()),
        vertex_to_partitions_opt,
        edge_midpoints_opt,
        opt_benchmarks,
        final_coverage,
        os.path.join(output_dir, f"{name}"),
        interactive=False
    )
    
    # 创建摘要报告
    create_summary_report(
        f"{name}",
        metrics_opt,
        os.path.join(output_dir, f"{name}_report.txt")
    )
    
    # PyVista 增强可视化（热力图 + 边界 + 边界专用窗口）
    if use_pyvista and PYVISTA_AVAILABLE:
        print(f"\n--- Saving PyVista heatmap visualization ---")
        
        # 获取顶点和面数据
        vertices = np.asarray(mesh_processor.mesh.vertices)
        faces = np.asarray(mesh_processor.mesh.triangles)
        
        # 保存 PyVista 可视化（显示热力图窗口 + 第二个边界专用窗口）
        save_pyvista_visualization(
            vertices=vertices,
            faces=faces,
            partitions=list(regions_dict_opt.values()),
            vertex_to_partitions=vertex_to_partitions_opt,
            edge_midpoints=edge_midpoints_opt,
            benchmarks=opt_benchmarks,
            coverage=final_coverage,
            output_prefix=os.path.join(output_dir, f"{name}"),
            interactive=auto_view,
            show_boundaries_only=True  # 使用 PyVista 显示第二个边界窗口
        )
    elif use_pyvista and not PYVISTA_AVAILABLE:
        print(f"\n--- PyVista not available, skipping enhanced visualization ---")
    
    # 仅在 PyVista 不可用时才使用 Open3D 边界窗口
    if auto_view and (not use_pyvista or not PYVISTA_AVAILABLE):
        print(f"\n--- Opening Open3D boundary visualization ---")
        print(f"窗口 2: 分区边界（彩色）")
        print("按 Q 或点击窗口关闭按钮退出")
        visualize_boundaries_open3d(
            mesh=mesh_processor.mesh,
            partitions=list(regions_dict_opt.values()),
            vertex_to_partitions=vertex_to_partitions_opt,
            benchmarks=opt_benchmarks,
            window_name=f"分区边界 - {name}"
        )
    
    total_time = time.time() - start_time
    results["total_time"] = total_time
    print(f"\nTotal time: {total_time:.2f} seconds")
    
    return results


def test_parameter_sensitivity(
    name: str,
    mesh_processor: MeshProcessor,
    output_dir: str = 'test_output'
) -> Dict:
    """
    参数敏感性测试
    """
    print(f"\n{'=' * 80}")
    print(f"Parameter Sensitivity: {name}")
    print(f"{'=' * 80}")
    
    results = {}
    partitioner = NewPartitioner(mesh_processor)
    initializer = BasePointInitializer(mesh_processor, 20)
    benchmarks = initializer.sample('uniform')
    
    print(f"\n--- Testing alpha sensitivity ---")
    alpha_values = [0.0, 0.5, 1.0, 2.0]
    avg_coverage_values = []
    
    for alpha in alpha_values:
        partitions, vertex_to_partitions, edge_midpoints = \
            partitioner.partition_surface(benchmarks, alpha, None, 30.0)
        
        coverage = np.zeros(partitioner.num_vertices)
        for v, p_list in vertex_to_partitions.items():
            coverage[v] = len(p_list)
        
        avg_coverage = np.mean(coverage)
        avg_coverage_values.append(avg_coverage)
        print(f"  alpha={alpha}: avg_coverage={avg_coverage:.2f}")
    
    results["alpha"] = {
        "values": alpha_values,
        "avg_coverage": avg_coverage_values
    }
    
    plot_parameter_sensitivity(
        "alpha", alpha_values, "Average Coverage", avg_coverage_values,
        os.path.join(output_dir, f"{name}_alpha_sensitivity.png")
    )
    
    print(f"\n--- Testing theta_attr sensitivity ---")
    theta_values = [10.0, 20.0, 30.0, 45.0]
    avg_coverage_values = []
    
    for theta in theta_values:
        partitions, vertex_to_partitions, edge_midpoints = \
            partitioner.partition_surface(benchmarks, 0.0, None, theta)
        
        coverage = np.zeros(partitioner.num_vertices)
        for v, p_list in vertex_to_partitions.items():
            coverage[v] = len(p_list)
        
        avg_coverage = np.mean(coverage)
        avg_coverage_values.append(avg_coverage)
        print(f"  theta_attr={theta}: avg_coverage={avg_coverage:.2f}")
    
    results["theta_attr"] = {
        "values": theta_values,
        "avg_coverage": avg_coverage_values
    }
    
    plot_parameter_sensitivity(
        "theta_attr", theta_values, "Average Coverage", avg_coverage_values,
        os.path.join(output_dir, f"{name}_theta_sensitivity.png")
    )
    
    return results


def get_timestamped_output_dir(base_dir="new_output"):
    """
    创建带时间戳的输出目录
    
    Args:
        base_dir: 基础目录 (默认: new_output)
    
    Returns:
        带时间戳的输出目录路径
    """
    # 生成时间戳：YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 创建完整路径
    output_dir = os.path.join(base_dir, timestamp)
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="分区算法测试 - PyVista 增强可视化（热力图 + 彩色边界）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Features:
  - Coverage heatmap: plasma colormap (blue -> yellow -> red)
  - Partition boundaries: colored lines showing partition edges
  - Interactive 3D: rotate, zoom, pan with mouse

Examples:
  # Basic test (auto visualization)
  python .\\tests\\test_new_full_pipeline.py --shapes cylinder

  # Multiple shapes
  python .\\tests\\test_new_full_pipeline.py --shapes cylinder cone wavy_plane

  # Disable PyVista (use only Open3D)
  python .\\tests\\test_new_full_pipeline.py --shapes cylinder --no-pyvista

  # Disable auto visualization (files only)
  python .\\tests\\test_new_full_pipeline.py --shapes cylinder --no-view

  # Full test with sensitivity analysis
  python .\\tests\\test_new_full_pipeline.py --shapes cylinder --full
        """
    )

    parser.add_argument(
        "--base-dir", type=str, default="new_output",
        help="基础输出目录 (默认: new_output)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="快速测试模式 (更少迭代次数)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="完整测试模式 (包含参数敏感性分析)"
    )
    parser.add_argument(
        "--no-view", action="store_true",
        help="禁用自动可视化 (只生成文件)"
    )
    parser.add_argument(
        "--no-pyvista", action="store_true",
        help="禁用 PyVista 增强可视化"
    )
    parser.add_argument(
        "--shapes", type=str, nargs="+",
        default=["cylinder", "cone", "wavy_plane"],
        help="要测试的形状 (cylinder, cone, wavy_plane)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0,
        help="曲率拉伸强度 (默认: 0.0)"
    )
    parser.add_argument(
        "--theta", type=float, default=30.0,
        help="属性差异阈值，度 (默认: 30.0)"
    )
    parser.add_argument(
        "--benchmarks", type=int, default=None,
        help="初始基准点数量 (默认: 自动)"
    )

    args = parser.parse_args()

    # 获取带时间戳的输出目录
    output_dir = get_timestamped_output_dir(args.base_dir)
    
    # 参数设置
    max_iterations = 20 if args.quick else 50
    initial_num_benchmarks = args.benchmarks if args.benchmarks else (15 if args.quick else 20)
    auto_view = not args.no_view
    use_pyvista = not args.no_pyvista
    
    shape_generators = {
        "cylinder": generate_cylinder,
        "cone": generate_cone,
        "wavy_plane": generate_wavy_plane
    }
    
    all_results = {}
    
    # PyVista 可视化说明
    if use_pyvista and PYVISTA_AVAILABLE:
        print(f"\n{'=' * 80}")
        print(f"PyVista Enhanced Visualization")
        print(f"{'=' * 80}")
        print(f"- Coverage heatmap: plasma colormap (blue -> yellow -> red)")
        print(f"- Partition boundaries: colored lines showing partition edges")
        print(f"- Interactive 3D: rotate, zoom, pan with mouse")
        print(f"{'=' * 80}\n")
    elif use_pyvista and not PYVISTA_AVAILABLE:
        print(f"\nWarning: PyVista not installed.")
        print(f"Install with: pip install pyvista")
        print(f"Running without PyVista visualization...\n")
    
    print(f"\n{'=' * 80}")
    print(f"Partition Algorithm Test")
    print(f"{'=' * 80}")
    print(f"Output directory: {output_dir}")
    print(f"Max iterations: {max_iterations}")
    print(f"Initial benchmarks: {initial_num_benchmarks}")
    print(f"Parameters: alpha={args.alpha}, theta_attr={args.theta}")
    print(f"{'=' * 80}\n")
    
    for shape_name in args.shapes:
        if shape_name not in shape_generators:
            print(f"Unknown shape: {shape_name}, skipping")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Generating mesh: {shape_name}")
        print(f"{'=' * 80}")
        
        generator = shape_generators[shape_name]
        trimesh_mesh = generator()
        print(f"Generated mesh: {len(trimesh_mesh.vertices)} vertices, {len(trimesh_mesh.faces)} faces")
        
        mesh_processor = wrap_mesh_with_processor(trimesh_mesh)
        
        results = test_single_shape(
            shape_name, mesh_processor,
            alpha=args.alpha,
            theta_attr=args.theta,
            initial_num_benchmarks=initial_num_benchmarks,
            max_iterations=max_iterations,
            output_dir=output_dir,
            auto_view=auto_view,
            use_pyvista=use_pyvista
        )
        
        all_results[shape_name] = results
        
        # 参数敏感性测试（仅完整模式）
        if args.full:
            sensitivity_results = test_parameter_sensitivity(
                shape_name, mesh_processor, output_dir
            )
            all_results[shape_name]["sensitivity"] = sensitivity_results
    
    # 打印最终摘要
    print(f"\n{'=' * 80}")
    print("Testing Complete! Summary:")
    print(f"{'=' * 80}")
    
    for shape_name, results in all_results.items():
        print(f"\n{shape_name}:")
        if "optimized" in results:
            opt = results["optimized"]
            print(f"  Uncovered: {opt['uncovered_ratio']:.2%}")
            print(f"  Avg coverage: {opt['avg_coverage']:.2f}")
            print(f"  Num benchmarks: {opt['num_benchmarks']}")
            if "median_anisotropy" in opt:
                print(f"  Median anisotropy: {opt['median_anisotropy']:.2f}")
    
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 80}\n")
    
    # 提示用户如何查看结果
    print("To view saved results:")
    print(f"  - Open3D PLY files: MeshLab or Open3D")
    if PYVISTA_AVAILABLE:
        print(f"  - PyVista HTML (interactive): Open in browser")
    print(f"  - Screenshots: PNG files in {output_dir}")


if __name__ == "__main__":
    main()
