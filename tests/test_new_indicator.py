"""
测试NewIndicatorCalculator类的功能
"""

import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import open3d as o3d
from core.meshProcessor import MeshProcessor
from new.newIndicator import NewIndicatorCalculator

def test_indicator_calculator():
    """
    测试NewIndicatorCalculator的基本功能
    """
    print("开始测试NewIndicatorCalculator...")
    
    # 创建一个简单的测试网格（球体）
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    mesh.compute_vertex_normals()
    
    # 创建MeshProcessor实例
    mesh_processor = MeshProcessor(mesh)
    
    # 创建NewIndicatorCalculator实例
    indicator_calculator = NewIndicatorCalculator(mesh_processor)
    
    # 测试1: 计算平均边长
    avg_edge_length = indicator_calculator.avg_edge_length
    print(f"测试1: 平均边长 = {avg_edge_length:.4f}")
    assert avg_edge_length > 0, "平均边长应该大于0"
    
    # 测试2: 计算度量张量
    vertex_idx = 0
    alpha = 2.0
    metric_tensor = indicator_calculator._calculate_metric_tensor(vertex_idx, alpha)
    print(f"测试2: 度量张量形状 = {metric_tensor.shape}")
    assert metric_tensor.shape == (3, 3), "度量张量应该是3x3矩阵"
    
    # 测试3: 计算有效长度
    if len(mesh_processor.adjacency[vertex_idx]) > 0:
        neighbor_idx = mesh_processor.adjacency[vertex_idx][0]
        effective_length = indicator_calculator._calculate_effective_length(vertex_idx, neighbor_idx, alpha)
        print(f"测试3: 有效长度 = {effective_length:.4f}")
        assert effective_length > 0, "有效长度应该大于0"
    
    # 测试4: 计算属性差异
    if len(mesh_processor.vertices) > 1:
        attr_diff = indicator_calculator._calculate_attribute_difference(0, 1)
        print(f"测试4: 属性差异 = {attr_diff:.4f}")
        assert attr_diff >= 0, "属性差异应该大于等于0"
    
    # 测试5: 区域生长
    benchmark = 0
    region = indicator_calculator.grow_region(benchmark)
    print(f"测试5: 区域大小 = {len(region)}")
    assert len(region) > 0, "区域应该包含至少一个顶点"
    assert benchmark in region, "基准点应该在区域中"
    
    # 测试6: 相似性判断
    if len(mesh_processor.vertices) > 1:
        is_similar = indicator_calculator.is_similar(1, 0)
        print(f"测试6: 顶点1与顶点0是否相似 = {is_similar}")
    
    print("所有测试通过！")

def test_with_different_alpha():
    """
    测试不同alpha值对区域生长的影响
    """
    print("\n测试不同alpha值对区域生长的影响...")
    
    # 创建测试网格
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    mesh.compute_vertex_normals()
    mesh_processor = MeshProcessor(mesh)
    indicator_calculator = NewIndicatorCalculator(mesh_processor)
    
    # 测试不同的alpha值
    alpha_values = [0.0, 1.0, 5.0, 10.0]
    benchmark = 0
    
    for alpha in alpha_values:
        region = indicator_calculator.grow_region(benchmark, alpha=alpha)
        print(f"alpha={alpha}: 区域大小 = {len(region)}")
    
    print("不同alpha值测试完成！")

def test_with_different_R_max():
    """
    测试不同R_max值对区域生长的影响
    """
    print("\n测试不同R_max值对区域生长的影响...")
    
    # 创建测试网格
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    mesh.compute_vertex_normals()
    mesh_processor = MeshProcessor(mesh)
    indicator_calculator = NewIndicatorCalculator(mesh_processor)
    
    # 测试不同的R_max值
    R_max_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    benchmark = 0
    
    for R_max in R_max_values:
        region = indicator_calculator.grow_region(benchmark, R_max=R_max)
        print(f"R_max={R_max}: 区域大小 = {len(region)}")
    
    print("不同R_max值测试完成！")

if __name__ == "__main__":
    test_indicator_calculator()
    test_with_different_alpha()
    test_with_different_R_max()
