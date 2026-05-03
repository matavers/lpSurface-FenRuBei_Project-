"""
简单的分区可视化演示脚本
"""
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meshProcessor import MeshProcessor
from new import NewPartitioner, BasePointInitializer
from geometry_generators import generate_cylinder, generate_cone, generate_wavy_plane
from tests.visualizer_enhanced import visualize_interactive, save_complete_visualization


def wrap_mesh_with_processor(trimesh_mesh):
    """将 trimesh 网格包装为 MeshProcessor 对象"""
    import open3d as o3d
    
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.compute_triangle_normals()
    
    return MeshProcessor(o3d_mesh)


def visualize_single_shape(shape_name: str, output_dir: str = "demo_output"):
    """可视化单个形状的分区"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{'=' * 80}")
    print(f"可视化: {shape_name}")
    print(f"{'=' * 80}")
    
    # 生成网格
    generators = {
        "cylinder": generate_cylinder,
        "cone": generate_cone,
        "wavy_plane": generate_wavy_plane
    }
    
    if shape_name not in generators:
        print(f"未知形状: {shape_name}")
        return
    
    trimesh_mesh = generators[shape_name]()
    print(f"生成网格: {len(trimesh_mesh.vertices)} 顶点, {len(trimesh_mesh.faces)} 面")
    
    # 创建分区器
    mesh_processor = wrap_mesh_with_processor(trimesh_mesh)
    partitioner = NewPartitioner(mesh_processor)
    
    # 初始化基准点
    initializer = BasePointInitializer(mesh_processor, 15)
    benchmarks = initializer.sample('uniform')
    
    # 分区
    partitions, vertex_to_partitions, edge_midpoints = partitioner.partition_surface(
        benchmarks, alpha=0.0, theta_attr=30.0
    )
    
    # 计算覆盖
    coverage = np.zeros(partitioner.num_vertices)
    for v, p_list in vertex_to_partitions.items():
        coverage[v] = len(p_list)
    
    print(f"\n分区完成:")
    print(f"  分区数量: {len(partitions)}")
    print(f"  平均覆盖: {np.mean(coverage):.2f}")
    print(f"  未覆盖: {np.mean(coverage == 0):.2%}")
    print(f"  边界点数量: {len(edge_midpoints)}")
    
    # 保存可视化
    print(f"\n保存可视化结果到: {output_dir}")
    save_complete_visualization(
        mesh_processor.mesh,
        partitions,
        vertex_to_partitions,
        edge_midpoints,
        benchmarks,
        coverage,
        os.path.join(output_dir, f"{shape_name}"),
        interactive=False
    )
    
    print(f"\n{'=' * 80}")
    print(f"打开交互式可视化...")
    print(f"{'=' * 80}")
    
    # 显示交互式可视化
    visualize_interactive(
        mesh_processor.mesh,
        vertex_to_partitions,
        edge_midpoints,
        benchmarks,
        window_name=f"分区可视化: {shape_name}"
    )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="分区可视化演示")
    parser.add_argument("--shape", type=str, default="cylinder",
                       choices=["cylinder", "cone", "wavy_plane"],
                       help="要可视化的形状")
    parser.add_argument("--output-dir", type=str, default="demo_output",
                       help="输出目录")
    args = parser.parse_args()
    
    visualize_single_shape(args.shape, args.output_dir)


if __name__ == "__main__":
    main()
