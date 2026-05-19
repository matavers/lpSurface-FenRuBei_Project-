"""
NURBS曲面处理模块测试脚本
测试NURBS曲面的几何属性计算、网格生成等功能
"""
import sys
import os
import time
import numpy as np
import argparse
from typing import Dict, Tuple, Any
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.nurbsProcessor import NURBSProcessor
from core.meshProcessor import MeshProcessor


def test_nurbs_evaluation(nurbs: NURBSProcessor, name: str) -> Dict:
    """
    测试NURBS曲面求值功能
    
    Args:
        nurbs: NURBSProcessor实例
        name: 曲面名称
        
    Returns:
        测试结果字典
    """
    print(f"\n--- Testing {name} evaluation ---")
    
    results = {"shape": name, "tests": []}
    
    test_points = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.25, 0.75), (0.75, 0.25)]
    
    for u, v in test_points:
        point = nurbs.evaluate(u, v)
        normal = nurbs.calculate_normal(u, v)
        k1, k2 = nurbs.calculate_curvature(u, v)
        gaussian = nurbs.calculate_gaussian_curvature(u, v)
        mean = nurbs.calculate_mean_curvature(u, v)
        
        print(f"  (u={u:.2f}, v={v:.2f}):")
        print(f"    Point: {point}")
        print(f"    Normal: {normal} (norm={np.linalg.norm(normal):.6f})")
        print(f"    Principal curvatures: k1={k1:.6f}, k2={k2:.6f}")
        print(f"    Gaussian curvature: {gaussian:.6f}")
        print(f"    Mean curvature: {mean:.6f}")
        
        results["tests"].append({
            "u": u, "v": v,
            "point": point.tolist(),
            "normal": normal.tolist(),
            "k1": float(k1), "k2": float(k2),
            "gaussian": float(gaussian),
            "mean": float(mean)
        })
    
    return results


def test_nurbs_derivatives(nurbs: NURBSProcessor, name: str) -> Dict:
    """
    测试NURBS曲面导数计算功能
    
    Args:
        nurbs: NURBSProcessor实例
        name: 曲面名称
        
    Returns:
        测试结果字典
    """
    print(f"\n--- Testing {name} derivatives ---")
    
    results = {"shape": name, "derivatives": []}
    
    u, v = 0.5, 0.5
    
    S, Su, Sv, Suu, Suv, Svv = nurbs.evaluate_derivatives(u, v)
    
    print(f"  Point S: {S}")
    print(f"  First derivative Su: {Su}")
    print(f"  First derivative Sv: {Sv}")
    print(f"  Second derivative Suu: {Suu}")
    print(f"  Second derivative Suv: {Suv}")
    print(f"  Second derivative Svv: {Svv}")
    
    # 验证法线计算
    normal_analytic = nurbs.calculate_normal_analytic(u, v)
    normal_cross = np.cross(Su, Sv)
    normal_cross /= np.linalg.norm(normal_cross)
    
    print(f"\n  Normal verification:")
    print(f"    Analytic normal: {normal_analytic}")
    print(f"    Cross product normal: {normal_cross}")
    print(f"    Match: {np.allclose(normal_analytic, normal_cross, atol=1e-6)}")
    
    results["derivatives"].append({
        "u": u, "v": v,
        "S": S.tolist(),
        "Su": Su.tolist(), "Sv": Sv.tolist(),
        "Suu": Suu.tolist(), "Suv": Suv.tolist(), "Svv": Svv.tolist(),
        "normal_match": bool(np.allclose(normal_analytic, normal_cross, atol=1e-6))
    })
    
    return results


def test_mesh_generation(nurbs: NURBSProcessor, name: str, resolution_u: int = 50, resolution_v: int = 50) -> Dict:
    """
    测试网格生成功能
    
    Args:
        nurbs: NURBSProcessor实例
        name: 曲面名称
        resolution_u: U方向分辨率
        resolution_v: V方向分辨率
        
    Returns:
        测试结果字典
    """
    print(f"\n--- Testing {name} mesh generation ---")
    
    results = {"shape": name, "mesh": {}}
    
    start_time = time.time()
    mesh = nurbs.generate_mesh(resolution_u, resolution_v)
    gen_time = time.time() - start_time
    
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    print(f"  Resolution: {resolution_u}x{resolution_v}")
    print(f"  Generated {len(vertices)} vertices, {len(faces)} faces")
    print(f"  Generation time: {gen_time:.4f} seconds")
    
    # 检查网格属性
    bounds = mesh.get_axis_aligned_bounding_box()
    print(f"  Bounding box: min={bounds.min_bound}, max={bounds.max_bound}")
    
    results["mesh"] = {
        "resolution_u": resolution_u,
        "resolution_v": resolution_v,
        "num_vertices": len(vertices),
        "num_faces": len(faces),
        "generation_time": float(gen_time),
        "bounds": {
            "min": bounds.min_bound.tolist(),
            "max": bounds.max_bound.tolist()
        }
    }
    
    return results


def test_nurbs_conversion_to_mesh_processor(nurbs: NURBSProcessor, name: str) -> Dict:
    """
    测试NURBS曲面转换为MeshProcessor
    
    Args:
        nurbs: NURBSProcessor实例
        name: 曲面名称
        
    Returns:
        测试结果字典
    """
    print(f"\n--- Testing {name} conversion to MeshProcessor ---")
    
    results = {"shape": name, "conversion": {}}
    
    mesh = nurbs.generate_mesh(30, 30)
    mesh_processor = MeshProcessor(mesh)
    
    print(f"  Number of vertices: {len(mesh_processor.vertices)}")
    print(f"  Number of triangles: {len(mesh_processor.faces)}")
    print(f"  Has normals: {mesh.has_vertex_normals()}")
    
    # 计算一些网格属性
    area = mesh.get_surface_area()
    print(f"  Surface area: {area:.6f}")
    
    results["conversion"] = {
        "num_vertices": len(mesh_processor.vertices),
        "num_triangles": len(mesh_processor.faces),
        "has_normals": mesh.has_vertex_normals(),
        "surface_area": float(area)
    }
    
    return results


def test_surface_properties(nurbs: NURBSProcessor, name: str, sample_count: int = 10) -> Dict:
    """
    测试曲面几何属性统计
    
    Args:
        nurbs: NURBSProcessor实例
        name: 曲面名称
        sample_count: 采样点数量
        
    Returns:
        测试结果字典
    """
    print(f"\n--- Testing {name} surface properties statistics ---")
    
    results = {"shape": name, "statistics": {}}
    
    curvatures = []
    gaussian_curvatures = []
    mean_curvatures = []
    
    for i in range(sample_count):
        for j in range(sample_count):
            u = (i + 0.5) / sample_count
            v = (j + 0.5) / sample_count
            k1, k2 = nurbs.calculate_curvature(u, v)
            gaussian = nurbs.calculate_gaussian_curvature(u, v)
            mean = nurbs.calculate_mean_curvature(u, v)
            
            curvatures.append((k1, k2))
            gaussian_curvatures.append(gaussian)
            mean_curvatures.append(mean)
    
    curvatures = np.array(curvatures)
    gaussian_curvatures = np.array(gaussian_curvatures)
    mean_curvatures = np.array(mean_curvatures)
    
    print(f"  Sampled {sample_count}x{sample_count} = {sample_count*sample_count} points")
    print(f"  Principal curvature k1: min={curvatures[:, 0].min():.6f}, max={curvatures[:, 0].max():.6f}, mean={curvatures[:, 0].mean():.6f}")
    print(f"  Principal curvature k2: min={curvatures[:, 1].min():.6f}, max={curvatures[:, 1].max():.6f}, mean={curvatures[:, 1].mean():.6f}")
    print(f"  Gaussian curvature: min={gaussian_curvatures.min():.6f}, max={gaussian_curvatures.max():.6f}, mean={gaussian_curvatures.mean():.6f}")
    print(f"  Mean curvature: min={mean_curvatures.min():.6f}, max={mean_curvatures.max():.6f}, mean={mean_curvatures.mean():.6f}")
    
    results["statistics"] = {
        "sample_count": sample_count * sample_count,
        "k1": {
            "min": float(curvatures[:, 0].min()),
            "max": float(curvatures[:, 0].max()),
            "mean": float(curvatures[:, 0].mean())
        },
        "k2": {
            "min": float(curvatures[:, 1].min()),
            "max": float(curvatures[:, 1].max()),
            "mean": float(curvatures[:, 1].mean())
        },
        "gaussian": {
            "min": float(gaussian_curvatures.min()),
            "max": float(gaussian_curvatures.max()),
            "mean": float(gaussian_curvatures.mean())
        },
        "mean": {
            "min": float(mean_curvatures.min()),
            "max": float(mean_curvatures.max()),
            "mean": float(mean_curvatures.mean())
        }
    }
    
    return results


def test_cylinder():
    """测试圆柱面"""
    print(f"\n{'=' * 60}")
    print("Testing Cylinder Surface")
    print(f"{'=' * 60}")
    
    cylinder = NURBSProcessor.create_cylinder(radius=1.0, height=2.0)
    
    all_results = {}
    
    all_results["evaluation"] = test_nurbs_evaluation(cylinder, "Cylinder")
    all_results["derivatives"] = test_nurbs_derivatives(cylinder, "Cylinder")
    all_results["mesh"] = test_mesh_generation(cylinder, "Cylinder")
    all_results["conversion"] = test_nurbs_conversion_to_mesh_processor(cylinder, "Cylinder")
    all_results["statistics"] = test_surface_properties(cylinder, "Cylinder")
    
    # 圆柱面特殊验证：高斯曲率应为0
    gaussian_at_center = cylinder.calculate_gaussian_curvature(0.5, 0.5)
    print(f"\n  Cylinder special verification:")
    print(f"    Gaussian curvature at center should be ~0: {gaussian_at_center:.6f}")
    assert abs(gaussian_at_center) < 0.1, f"Cylinder Gaussian curvature should be near 0, got {gaussian_at_center}"
    print("    PASSED")
    
    return all_results


def test_sphere():
    """测试球面"""
    print(f"\n{'=' * 60}")
    print("Testing Sphere Surface")
    print(f"{'=' * 60}")
    
    sphere = NURBSProcessor.create_sphere(radius=1.0, resolution=20)
    
    all_results = {}
    
    all_results["evaluation"] = test_nurbs_evaluation(sphere, "Sphere")
    all_results["derivatives"] = test_nurbs_derivatives(sphere, "Sphere")
    all_results["mesh"] = test_mesh_generation(sphere, "Sphere")
    all_results["conversion"] = test_nurbs_conversion_to_mesh_processor(sphere, "Sphere")
    all_results["statistics"] = test_surface_properties(sphere, "Sphere")
    
    # 球面特殊验证：高斯曲率应为正（凸曲面）
    gaussian_at_center = sphere.calculate_gaussian_curvature(0.5, 0.5)
    print(f"\n  Sphere special verification:")
    print(f"    Gaussian curvature at center: {gaussian_at_center:.6f}")
    assert gaussian_at_center > 0, f"Sphere Gaussian curvature should be positive, got {gaussian_at_center}"
    print("    PASSED (positive Gaussian curvature)")
    
    return all_results


def test_cone():
    """测试圆锥面"""
    print(f"\n{'=' * 60}")
    print("Testing Cone Surface")
    print(f"{'=' * 60}")
    
    cone = NURBSProcessor.create_cone(radius=1.0, height=2.0, resolution_u=20, resolution_v=10)
    
    all_results = {}
    
    all_results["evaluation"] = test_nurbs_evaluation(cone, "Cone")
    all_results["derivatives"] = test_nurbs_derivatives(cone, "Cone")
    all_results["mesh"] = test_mesh_generation(cone, "Cone")
    all_results["conversion"] = test_nurbs_conversion_to_mesh_processor(cone, "Cone")
    all_results["statistics"] = test_surface_properties(cone, "Cone")
    
    # 圆锥面特殊验证：高斯曲率应为0（除顶点外）
    gaussian_at_base = cone.calculate_gaussian_curvature(0.5, 0.1)
    print(f"\n  Cone special verification:")
    print(f"    Gaussian curvature at base should be ~0: {gaussian_at_base:.6f}")
    assert abs(gaussian_at_base) < 0.5, f"Cone Gaussian curvature should be near 0, got {gaussian_at_base}"
    print("    PASSED")
    
    return all_results


def get_timestamped_output_dir(base_dir="nurbs_test_output"):
    """
    创建带时间戳的输出目录
    
    Args:
        base_dir: 基础目录
        
    Returns:
        带时间戳的输出目录路径
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="NURBS曲面处理模块测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
测试内容：
  - 曲面求值（点坐标、法线、曲率）
  - 导数计算（一阶、二阶偏导数）
  - 网格生成
  - MeshProcessor转换
  - 曲面属性统计

示例：
  # 测试所有曲面
  python .\\tests\\test_new_with_nurbs.py
  
  # 仅测试圆柱面
  python .\\tests\\test_new_with_nurbs.py --surfaces cylinder
  
  # 指定输出目录
  python .\\tests\\test_new_with_nurbs.py --output-dir my_results
        """
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="nurbs_test_output",
        help="输出目录 (默认: nurbs_test_output)"
    )
    parser.add_argument(
        "--surfaces", type=str, nargs="+",
        default=["cylinder", "sphere", "cone"],
        help="要测试的曲面类型 (cylinder, sphere, cone)"
    )
    
    args = parser.parse_args()
    
    output_dir = get_timestamped_output_dir(args.output_dir)
    
    print(f"\n{'=' * 60}")
    print("NURBS Processor Test Suite")
    print(f"{'=' * 60}")
    print(f"Output directory: {output_dir}")
    print(f"Surfaces to test: {args.surfaces}")
    print(f"{'=' * 60}\n")
    
    surface_tests = {
        "cylinder": test_cylinder,
        "sphere": test_sphere,
        "cone": test_cone
    }
    
    all_results = {}
    total_start_time = time.time()
    
    for surface_name in args.surfaces:
        if surface_name not in surface_tests:
            print(f"Unknown surface: {surface_name}, skipping")
            continue
        
        test_func = surface_tests[surface_name]
        results = test_func()
        all_results[surface_name] = results
    
    total_time = time.time() - total_start_time
    
    # 保存测试结果
    results_file = os.path.join(output_dir, "test_results.npz")
    np.savez(results_file, **all_results)
    print(f"\n{'=' * 60}")
    print("Test Results Summary")
    print(f"{'=' * 60}")
    
    for surface_name, results in all_results.items():
        print(f"\n{surface_name}:")
        if "statistics" in results:
            stats = results["statistics"].get("statistics", results["statistics"])
            if "gaussian" in stats:
                print(f"  Gaussian curvature: {stats['gaussian']['mean']:.4f} (mean)")
                print(f"  Mean curvature: {stats['mean']['mean']:.4f} (mean)")
        if "mesh" in results:
            mesh = results["mesh"].get("mesh", results["mesh"])
            if "num_vertices" in mesh:
                print(f"  Mesh: {mesh['num_vertices']} vertices, {mesh.get('num_faces', mesh.get('num_triangles', 0))} faces")
    
    print(f"\nTotal test time: {total_time:.2f} seconds")
    print(f"Results saved to: {results_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()