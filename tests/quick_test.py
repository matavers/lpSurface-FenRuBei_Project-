"""
快速测试脚本：使用宽松参数
"""
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meshProcessor import MeshProcessor
from new import NewPartitioner, BasePointInitializer
from tests.geometry_generators import generate_cylinder, generate_cone, generate_wavy_plane
import open3d as o3d


def wrap_mesh_with_processor(trimesh_mesh):
    """将 trimesh 网格包装为 MeshProcessor 对象"""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.compute_triangle_normals()
    return MeshProcessor(o3d_mesh)


def quick_test():
    """快速测试"""
    print("=" * 80)
    print("快速测试：使用宽松参数")
    print("=" * 80)
    
    trimesh_mesh = generate_cylinder(num_u=20, num_v=20)
    print(f"生成网格: {len(trimesh_mesh.vertices)} 顶点")
    
    mesh_processor = wrap_mesh_with_processor(trimesh_mesh)
    partitioner = NewPartitioner(mesh_processor)
    
    # 使用宽松参数测试
    initializer = BasePointInitializer(mesh_processor, 10)
    benchmarks = initializer.sample('uniform')
    
    print(f"\n测试参数: alpha=0 (各向同性), theta_attr=30度")
    partitions, vertex_to_partitions, edge_midpoints = partitioner.partition_surface(
        benchmarks, alpha=0.0, R_max=None, theta_attr=30.0
    )
    
    # 计算覆盖
    coverage = np.zeros(partitioner.num_vertices)
    for v, p_list in vertex_to_partitions.items():
        coverage[v] = len(p_list)
    
    uncovered_ratio = np.mean(coverage == 0)
    avg_coverage = np.mean(coverage)
    
    print(f"\n结果:")
    print(f"  未覆盖比例: {uncovered_ratio:.2%}")
    print(f"  平均覆盖: {avg_coverage:.2f}")
    print(f"  区域大小: min={min(len(p) for p in partitions)}, max={max(len(p) for p in partitions)}, avg={np.mean([len(p) for p in partitions]):.1f}")


if __name__ == "__main__":
    quick_test()
