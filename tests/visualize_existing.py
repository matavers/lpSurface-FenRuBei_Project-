"""
可视化已有的 test_new_full_pipeline 测试结果
可以直接加载并可视化之前生成的结果
"""
import sys
import os
import numpy as np
import pickle
from typing import Dict, List, Set

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.visualizer_enhanced import (
    visualize_interactive,
    save_complete_visualization,
    plot_partition_sizes,
    plot_coverage_stats
)
import open3d as o3d


def load_partition_results(output_dir: str, prefix: str) -> Dict:
    """
    加载分区结果（如果之前有保存的话）
    
    Args:
        output_dir: 输出目录
        prefix: 结果前缀（如 "cylinder_fixed", "cylinder_optimized"）
        
    Returns:
        包含所有结果的字典
    """
    results = {}
    
    # 尝试加载 mesh
    mesh_path = os.path.join(output_dir, f"{prefix}_colored_mesh.ply")
    if os.path.exists(mesh_path):
        results['mesh'] = o3d.io.read_triangle_mesh(mesh_path)
    
    return results


def visualize_existing_results(output_dir: str, prefix: str):
    """
    可视化已有的测试结果
    
    Args:
        output_dir: 输出目录
        prefix: 结果前缀（如 "cylinder_fixed", "cylinder_optimized"）
    """
    print(f"{'=' * 80}")
    print(f"可视化已有结果: {prefix}")
    print(f"{'=' * 80}")
    
    # 加载彩色网格
    mesh_path = os.path.join(output_dir, f"{prefix}_colored_mesh.ply")
    if not os.path.exists(mesh_path):
        print(f"错误: 找不到文件 {mesh_path}")
        print(f"请先运行 test_new_full_pipeline.py 生成结果")
        return
    
    print(f"加载网格: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    print(f"网格顶点数: {len(np.asarray(mesh.vertices))}")
    
    # 直接显示交互式可视化
    print(f"\n打开交互式可视化窗口...")
    print(f"按 Q 退出，或使用鼠标旋转/缩放视图")
    
    o3d.visualization.draw_geometries(
        [mesh],
        window_name=f"分区可视化: {prefix}",
        width=1024,
        height=768
    )


def show_all_results_in_directory(output_dir: str):
    """
    显示目录下所有可用的结果文件
    
    Args:
        output_dir: 输出目录
    """
    print(f"{'=' * 80}")
    print(f"扫描目录: {output_dir}")
    print(f"{'=' * 80}")
    
    if not os.path.exists(output_dir):
        print(f"错误: 目录不存在 {output_dir}")
        print(f"\n请先运行 test_new_full_pipeline.py 生成结果：")
        print(f"  python .\\tests\\test_new_full_pipeline.py --quick --shapes cylinder")
        return
    
    # 查找所有 colored_mesh 文件
    mesh_files = [f for f in os.listdir(output_dir) if f.endswith('_colored_mesh.ply')]
    
    if not mesh_files:
        print(f"未找到任何 colored_mesh.ply 文件")
        print(f"\n请先运行 test_new_full_pipeline.py 生成结果：")
        print(f"  python .\\tests\\test_new_full_pipeline.py --quick --shapes cylinder")
        return
    
    print(f"\n找到 {len(mesh_files)} 个结果文件：\n")
    
    for i, filename in enumerate(sorted(mesh_files), 1):
        prefix = filename.replace('_colored_mesh.ply', '')
        print(f"  {i}. {prefix}")
    
    print(f"\n使用方法：")
    print(f"  python .\\tests\\visualize_existing.py --prefix <result_prefix>")
    print(f"\n示例：")
    print(f"  python .\\tests\\visualize_existing.py --prefix cylinder_fixed")
    print(f"  python .\\tests\\visualize_existing.py --prefix cylinder_optimized")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="可视化 test_new_full_pipeline 生成的测试结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看目录下所有可用结果
  python .\\tests\\visualize_existing.py --scan test_output
  
  # 可视化指定结果
  python .\\tests\\visualize_existing.py --prefix cylinder_fixed
  
  # 指定结果目录和前缀
  python .\\tests\\visualize_existing.py --dir test_output --prefix cylinder_optimized
        """
    )
    
    parser.add_argument(
        "--dir", "--output-dir",
        type=str,
        default="test_output",
        help="结果所在目录 (默认: test_output)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="要可视化的结果前缀 (如 cylinder_fixed, cylinder_optimized)"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="扫描目录并显示所有可用结果"
    )
    
    args = parser.parse_args()
    
    if args.scan:
        show_all_results_in_directory(args.dir)
    elif args.prefix:
        visualize_existing_results(args.dir, args.prefix)
    else:
        # 默认：扫描目录
        show_all_results_in_directory(args.dir)


if __name__ == "__main__":
    main()
