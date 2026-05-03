"""
诊断脚本：分析分区效果不好的原因
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


def diagnose_cylinder():
    """诊断圆柱网格的分区问题"""
    print("=" * 80)
    print("诊断圆柱网格分区问题")
    print("=" * 80)
    
    # 生成网格
    trimesh_mesh = generate_cylinder(num_u=20, num_v=20)
    print(f"生成网格: {len(trimesh_mesh.vertices)} 顶点, {len(trimesh_mesh.faces)} 面")
    
    mesh_processor = wrap_mesh_with_processor(trimesh_mesh)
    partitioner = NewPartitioner(mesh_processor)
    calculator = partitioner.indicator_calculator
    
    # 打印归一化因子
    print("\n=== 1. 归一化因子 ===")
    print(f"sigma_K (曲率): {calculator.sigma_K:.6f}")
    print(f"sigma_n (法向): {calculator.sigma_n:.6f}")
    print(f"sigma_R (直纹面误差): {calculator.sigma_R:.6f}")
    print(f"avg_edge_length: {calculator.avg_edge_length:.6f}")
    
    # 检查曲率值
    print("\n=== 2. 曲率统计 ===")
    principal_curvatures = mesh_processor.principal_curvatures
    k1 = [k[0] for k in principal_curvatures]
    k2 = [k[1] for k in principal_curvatures]
    print(f"k1 (主曲率1): min={min(k1):.4f}, max={max(k1):.4f}, mean={np.mean(k1):.4f}")
    print(f"k2 (主曲率2): min={min(k2):.4f}, max={max(k2):.4f}, mean={np.mean(k2):.4f}")
    avg_curvatures = [(k1[i] + k2[i]) / 2 for i in range(len(k1))]
    print(f"平均曲率: min={min(avg_curvatures):.4f}, max={max(avg_curvatures):.4f}, std={np.std(avg_curvatures):.4f}")
    
    # 检查法向量
    print("\n=== 3. 法向量统计 ===")
    normals = mesh_processor.vertex_normals
    normal_norms = [np.linalg.norm(n) for n in normals]
    print(f"法向量范数: min={min(normal_norms):.4f}, max={max(normal_norms):.4f}")
    
    # 选取一个基准点进行诊断
    print("\n=== 4. 单点区域生长诊断 ===")
    initializer = BasePointInitializer(mesh_processor, 20)
    benchmarks = initializer.sample('uniform')
    test_benchmark = benchmarks[0]
    
    print(f"\n测试基准点: {test_benchmark}")
    print(f"基准点位置: {mesh_processor.vertices[test_benchmark]}")
    print(f"基准点曲率: k1={principal_curvatures[test_benchmark][0]:.4f}, k2={principal_curvatures[test_benchmark][1]:.4f}")
    print(f"基准点法向量: {normals[test_benchmark]}")
    
    # 测试不同的 theta_attr 值
    print("\n=== 5. 不同 theta_attr 值的区域大小 ===")
    for theta in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        region = calculator.grow_region(test_benchmark, alpha=2.0, R_max=None, theta_attr=theta)
        print(f"theta_attr={theta}: 区域大小={len(region)}")
    
    # 测试不同的 alpha 值
    print("\n=== 6. 不同 alpha 值的区域大小 (theta_attr=1.5) ===")
    for alpha in [0.0, 0.5, 1.0, 2.0]:
        calculator.clear_cache()
        region = calculator.grow_region(test_benchmark, alpha=alpha, R_max=None, theta_attr=1.5)
        print(f"alpha={alpha}: 区域大小={len(region)}")
    
    # 测试不同的 R_max 值
    print("\n=== 7. 不同 R_max 值的区域大小 (alpha=2.0, theta_attr=1.5) ===")
    for r_mult in [5, 10, 20, 50]:
        R_max = r_mult * calculator.avg_edge_length
        calculator.clear_cache()
        region = calculator.grow_region(test_benchmark, alpha=2.0, R_max=R_max, theta_attr=1.5)
        print(f"R_max={R_max:.4f} ({r_mult}x avg_edge): 区域大小={len(region)}")
    
    # 使用 debug 模式打印详细信息
    print("\n=== 8. 详细调试信息 (alpha=2.0, theta_attr=1.5, R_max=5x) ===")
    calculator.clear_cache()
    region = calculator.grow_region(test_benchmark, alpha=2.0, R_max=None, theta_attr=1.5, debug=True)
    
    # 分析邻居的属性差异
    print("\n=== 9. 基准点邻居的属性差异分析 ===")
    neighbors = list(mesh_processor.adjacency[test_benchmark])[:5]
    print(f"前5个邻居: {neighbors}")
    
    for neighbor in neighbors:
        attr_diff = calculator._calculate_attribute_difference(neighbor, test_benchmark)
        edge_len = calculator._calculate_effective_length(test_benchmark, neighbor, 2.0)
        print(f"邻居 {neighbor}: attr_diff={attr_diff:.3f}, edge_len={edge_len:.4f}")


def diagnose_all_shapes():
    """诊断所有形状"""
    shapes = {
        'cylinder': generate_cylinder(num_u=20, num_v=20),
        'cone': generate_cone(num_u=20, num_v=20),
        'wavy_plane': generate_wavy_plane(num_u=20, num_v=20)
    }
    
    for name, trimesh_mesh in shapes.items():
        print(f"\n\n{'=' * 80}")
        print(f"诊断 {name}")
        print(f"{'=' * 80}")
        
        mesh_processor = wrap_mesh_with_processor(trimesh_mesh)
        calculator = NewPartitioner(mesh_processor).indicator_calculator
        
        print(f"\n归一化因子:")
        print(f"  sigma_K={calculator.sigma_K:.6f}")
        print(f"  sigma_n={calculator.sigma_n:.6f}")
        print(f"  sigma_R={calculator.sigma_R:.6f}")
        print(f"  avg_edge_length={calculator.avg_edge_length:.6f}")
        
        # 选取一个基准点
        initializer = BasePointInitializer(mesh_processor, 20)
        benchmarks = initializer.sample('uniform')
        test_benchmark = benchmarks[0]
        
        print(f"\n测试基准点 {test_benchmark}:")
        print(f"  位置: {mesh_processor.vertices[test_benchmark]}")
        print(f"  曲率: {mesh_processor.principal_curvatures[test_benchmark]}")
        
        # 测试不同 theta_attr
        print(f"\n不同 theta_attr 值的区域大小:")
        for theta in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            calculator.clear_cache()
            region = calculator.grow_region(test_benchmark, alpha=2.0, R_max=None, theta_attr=theta)
            print(f"  theta_attr={theta}: {len(region)}")


if __name__ == "__main__":
    diagnose_cylinder()
    print("\n\n")
    diagnose_all_shapes()
