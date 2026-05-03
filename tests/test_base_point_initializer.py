"""
测试基准点初始化器
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import open3d as o3d
import numpy as np
from core.meshProcessor import MeshProcessor
from new.basePointDetermine import BasePointInitializer


def visualize_sampling(mesh, sampled_indices, title):
    """
    可视化采样结果
    Args:
        mesh: 原始网格
        sampled_indices: 采样点的索引
        title: 可视化窗口标题
    """
    # 创建网格副本
    mesh_copy = o3d.geometry.TriangleMesh()
    mesh_copy.vertices = mesh.vertices
    mesh_copy.triangles = mesh.triangles
    mesh_copy.vertex_normals = mesh.vertex_normals
    
    # 设置网格颜色为灰色
    mesh_copy.paint_uniform_color([0.7, 0.7, 0.7])
    
    # 创建采样点云
    vertices = np.asarray(mesh.vertices)
    sampled_vertices = vertices[sampled_indices]
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_vertices)
    # 设置采样点颜色为红色
    sampled_pcd.paint_uniform_color([1, 0, 0])
    
    # 使用Visualizer类来自定义点大小
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1024, height=768)
    
    # 添加几何对象
    vis.add_geometry(mesh_copy)
    vis.add_geometry(sampled_pcd)
    
    # 获取渲染选项并设置点大小
    opt = vis.get_render_option()
    opt.point_size = 10.0  # 设置点的大小
    opt.mesh_show_wireframe = False
    opt.mesh_show_back_face = False
    
    # 运行可视化
    vis.run()
    vis.destroy_window()


def test_base_point_initializer():
    """
    测试基准点初始化器
    """
    print("测试基准点初始化器...")
    
    # 创建一个简单的网格（立方体）
    mesh = o3d.geometry.TriangleMesh.create_box()
    
    # 创建网格处理器
    mesh_processor = MeshProcessor(mesh)
    
    # 测试不同的采样数量
    test_nums = [3, 6, 10, 20]
    
    for num in test_nums:
        print(f"\n测试采样数量: {num}")
        
        # 创建基准点初始化器
        initializer = BasePointInitializer(mesh_processor, num)
        
        # 测试随机采样
        print("  测试随机采样:")
        random_indices = initializer.random_sampling()
        print(f"    采样点数: {len(random_indices)}")
        print(f"    采样索引: {random_indices}")
        
        # 测试均匀采样
        print("  测试均匀采样:")
        uniform_indices = initializer.uniform_sampling()
        print(f"    采样点数: {len(uniform_indices)}")
        print(f"    采样索引: {uniform_indices}")
        
        # 测试通用采样方法 - 随机
        sample_random_indices = initializer.sample(method='random')
        print(f"    采样点数: {len(sample_random_indices)}")
        
        print("  测试通用采样方法 - 均匀:")
        sample_uniform_indices = initializer.sample(method='uniform')
        print(f"    采样点数: {len(sample_uniform_indices)}")
        
        print("  测试通用采样方法 - 泊松碟:")
        sample_poisson_indices = initializer.sample(method='poisson')
        print(f"    采样点数: {len(sample_poisson_indices)}")
        
        print("  测试通用采样方法 - 谱聚类:")
        sample_spectral_indices = initializer.sample(method='spectral')
        print(f"    采样点数: {len(sample_spectral_indices)}")
    
    # 测试边界情况 - 采样点数大于顶点数
    print("\n测试边界情况 - 采样点数大于顶点数:")
    large_num = 100  # 大于立方体的顶点数（8个）
    initializer = BasePointInitializer(mesh_processor, large_num)
    random_indices = initializer.random_sampling()
    print(f"  采样点数: {len(random_indices)}")
    print(f"  采样索引: {random_indices}")
    print(f"  验证是否返回所有顶点: {len(random_indices) == len(mesh_processor.vertices)}")
    
    # 测试边界情况 - 采样点数为0
    print("\n测试边界情况 - 采样点数为0:")
    zero_num = 0
    initializer = BasePointInitializer(mesh_processor, zero_num)
    random_indices = initializer.random_sampling()
    print(f"  采样点数: {len(random_indices)}")
    
    print("\n基准点初始化器测试完成！")


def test_visualization():
    """
    测试可视化功能，使用顶点数在1000左右的网格
    """
    print("\n测试可视化功能...")
    
    # 创建一个顶点数在1000左右的网格（球体）
    # 通过调整细分级别来控制顶点数
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    vertices = np.asarray(mesh.vertices)
    print(f"创建的网格顶点数: {len(vertices)}")
    
    # 创建网格处理器
    mesh_processor = MeshProcessor(mesh)
    
    # 测试不同采样方法
    sampling_methods = [('uniform', '均匀采样'), ('poisson', '泊松碟采样'), ('spectral', '谱聚类初始化')]
    sample_num = 50
    
    for method, method_name in sampling_methods:
        print(f"\n测试{method_name} - 采样数量: {sample_num}")
        
        # 创建基准点初始化器
        initializer = BasePointInitializer(mesh_processor, sample_num)
        
        # 执行采样
        sampled_indices = initializer.sample(method=method)
        print(f"  采样点数: {len(sampled_indices)}")
        
        # 可视化采样结果
        visualize_sampling(mesh, sampled_indices, f"{method_name}可视化 - {sample_num} 个点")
    
    print("\n可视化测试完成！")


if __name__ == "__main__":
    test_base_point_initializer()
    test_visualization()

