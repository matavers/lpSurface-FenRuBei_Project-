"""
分析属性差异计算
"""
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meshProcessor import MeshProcessor
from tests.geometry_generators import generate_cylinder
import open3d as o3d


def wrap_mesh_with_processor(trimesh_mesh):
    """将 trimesh 网格包装为 MeshProcessor 对象"""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.compute_triangle_normals()
    return MeshProcessor(o3d_mesh)


def analyze_attribute_diff():
    """分析属性差异计算"""
    print("=" * 80)
    print("分析属性差异计算")
    print("=" * 80)
    
    # 生成网格
    trimesh_mesh = generate_cylinder(num_u=20, num_v=20)
    mesh_processor = wrap_mesh_with_processor(trimesh_mesh)
    
    # 选择一个基准点
    test_benchmark = 248
    print(f"\n基准点 {test_benchmark}:")
    print(f"  位置: {mesh_processor.vertices[test_benchmark]}")
    print(f"  法向量: {mesh_processor.vertex_normals[test_benchmark]}")
    print(f"  曲率: {mesh_processor.principal_curvatures[test_benchmark]}")
    
    # 计算 sigma_n
    normal_angles = []
    for i, neighbors in enumerate(mesh_processor.adjacency):
        for j in neighbors:
            if i < j:
                n_i = mesh_processor.vertex_normals[i]
                n_j = mesh_processor.vertex_normals[j]
                dot_product = np.clip(np.dot(n_i, n_j), -1.0, 1.0)
                angle = np.arccos(dot_product)
                normal_angles.append(angle)
    
    sigma_n = np.std(normal_angles)
    print(f"\nsigma_n = {sigma_n:.6f}")
    print(f"法向角分布: min={min(normal_angles):.4f}, max={max(normal_angles):.4f}")
    print(f"法向角标准差: {sigma_n:.4f} rad = {np.degrees(sigma_n):.2f} deg")
    
    # 分析邻居
    neighbors = list(mesh_processor.adjacency[test_benchmark])[:5]
    print(f"\n邻居属性差异分析:")
    
    for neighbor in neighbors:
        n_v = mesh_processor.vertex_normals[neighbor]
        n_b = mesh_processor.vertex_normals[test_benchmark]
        
        # 法向角
        dot_product = np.clip(np.dot(n_v, n_b), -1.0, 1.0)
        normal_angle = np.arccos(dot_product)
        
        # 曲率
        k1_v, k2_v = mesh_processor.principal_curvatures[neighbor]
        k1_b, k2_b = mesh_processor.principal_curvatures[test_benchmark]
        avg_v = (k1_v + k2_v) / 2
        avg_b = (k1_b + k2_b) / 2
        avg_diff = abs(avg_v - avg_b)
        gaussian_diff = min(avg_diff / 10.0, 5.0)
        
        # delta 向量
        delta_n = normal_angle / sigma_n
        delta_K = gaussian_diff
        total_diff = np.sqrt(delta_n**2 + delta_K**2)
        
        print(f"\n  邻居 {neighbor}:")
        print(f"    法向角: {np.degrees(normal_angle):.2f} deg")
        print(f"    delta_n = {normal_angle:.4f} / {sigma_n:.4f} = {delta_n:.4f}")
        print(f"    曲率差异: {avg_diff:.4f}")
        print(f"    delta_K = min({avg_diff:.4f} / 10, 5) = {delta_K:.4f}")
        print(f"    总差异 = sqrt({delta_n:.4f}^2 + {delta_K:.4f}^2) = {total_diff:.4f}")


if __name__ == "__main__":
    analyze_attribute_diff()
