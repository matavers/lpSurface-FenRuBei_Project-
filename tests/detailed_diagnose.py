"""
详细诊断脚本：分析属性差异的各个分量
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


def detailed_attribute_analysis():
    """详细分析属性差异的各个分量"""
    print("=" * 80)
    print("详细诊断属性差异分量")
    print("=" * 80)
    
    # 生成网格
    trimesh_mesh = generate_cylinder(num_u=20, num_v=20)
    print(f"生成网格: {len(trimesh_mesh.vertices)} 顶点, {len(trimesh_mesh.faces)} 面")
    
    mesh_processor = wrap_mesh_with_processor(trimesh_mesh)
    
    # 检查 rolled_error
    print("\n=== 1. rolled_error 检查 ===")
    if hasattr(mesh_processor, 'rolled_error'):
        print(f"rolled_error 存在: len={len(mesh_processor.rolled_error)}")
        print(f"rolled_error 范围: {min(mesh_processor.rolled_error):.4f} - {max(mesh_processor.rolled_error):.4f}")
        print(f"rolled_error 标准差: {np.std(mesh_processor.rolled_error):.4f}")
    else:
        print("rolled_error 不存在")
    
    # 检查 gaussian_curvatures
    print("\n=== 2. gaussian_curvatures 检查 ===")
    if hasattr(mesh_processor, 'gaussian_curvatures'):
        print(f"gaussian_curvatures 存在: len={len(mesh_processor.gaussian_curvatures)}")
        print(f"范围: {min(mesh_processor.gaussian_curvatures):.4f} - {max(mesh_processor.gaussian_curvatures):.4f}")
    else:
        print("gaussian_curvatures 不存在")
    
    # 手动计算属性差异的各个分量
    print("\n=== 3. 属性差异分量分析 ===")
    
    # 选择一个基准点
    test_benchmark = 67
    
    k1_b = mesh_processor.principal_curvatures[test_benchmark][0]
    k2_b = mesh_processor.principal_curvatures[test_benchmark][1]
    n_b = mesh_processor.vertex_normals[test_benchmark]
    
    print(f"\n基准点 {test_benchmark}:")
    print(f"  曲率: k1={k1_b:.4f}, k2={k2_b:.4f}")
    print(f"  平均曲率: {(k1_b + k2_b) / 2:.4f}")
    print(f"  法向量: {n_b}")
    
    # 检查邻居
    neighbors = list(mesh_processor.adjacency[test_benchmark])[:5]
    print(f"\n邻居分析:")
    
    for neighbor in neighbors:
        k1_v = mesh_processor.principal_curvatures[neighbor][0]
        k2_v = mesh_processor.principal_curvatures[neighbor][1]
        n_v = mesh_processor.vertex_normals[neighbor]
        
        # 曲率差异
        K_v = (k1_v + k2_v) / 2
        K_b = (k1_b + k2_b) / 2
        curvature_diff = abs(K_v - K_b)
        
        # 法向差异
        dot_product = np.clip(np.dot(n_v, n_b), -1.0, 1.0)
        normal_angle = np.arccos(dot_product)
        
        # 直纹面误差差异
        if hasattr(mesh_processor, 'rolled_error'):
            e_rolled_v = mesh_processor.rolled_error[neighbor]
            e_rolled_b = mesh_processor.rolled_error[test_benchmark]
            rolled_error_diff = abs(e_rolled_v - e_rolled_b)
        else:
            rolled_error_diff = 0.0
        
        # 使用 sigma=10 计算各分量
        sigma_K = 10.0
        sigma_n = 0.12
        sigma_R = 0.01
        
        component_K = curvature_diff / sigma_K
        component_n = normal_angle / sigma_n
        component_R = rolled_error_diff / sigma_R
        
        # 总差异
        total_diff = np.sqrt(component_K**2 + component_n**2 + component_R**2)
        
        print(f"\n  邻居 {neighbor}:")
        print(f"    曲率: k1={k1_v:.4f}, k2={k2_v:.4f}, avg={K_v:.4f}")
        print(f"    曲率差异: {curvature_diff:.4f}, 归一化分量: {component_K:.4f}")
        print(f"    法向角: {np.degrees(normal_angle):.2f}度, 归一化分量: {component_n:.4f}")
        if hasattr(mesh_processor, 'rolled_error'):
            print(f"    rolled_error差异: {rolled_error_diff:.4f}, 归一化分量: {component_R:.4f}")
        print(f"    总差异 (L2): {total_diff:.4f}")


if __name__ == "__main__":
    detailed_attribute_analysis()
