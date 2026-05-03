"""
测试法向量相关修改与可视化脚本的适配
"""

import numpy as np
import open3d as o3d
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meshProcessor import MeshProcessor
from core.nurbsProcessor import NURBSProcessor
from utils.visualization import Visualizer


def test_normal_visualization():
    """
    测试法向量可视化
    """
    print("测试法向量相关修改与可视化脚本的适配...")
    
    # 创建圆锥面NURBS
    cone_radius = 1.0
    cone_height = 2.0
    nurbs_processor = NURBSProcessor.create_cone(radius=cone_radius, height=cone_height, resolution_u=20, resolution_v=10)
    
    # 生成圆锥面网格
    mesh = nurbs_processor.generate_mesh(resolution_u=50, resolution_v=20)
    
    # 计算法向量
    mesh.compute_vertex_normals()
    
    # 创建MeshProcessor实例
    mesh_processor = MeshProcessor(mesh)
    
    # 找到圆锥顶点（z值最大的点）
    vertices = np.asarray(mesh.vertices)
    vertex_heights = vertices[:, 2]
    apex_idx = np.argmax(vertex_heights)
    apex_pos = vertices[apex_idx]
    
    print(f"圆锥顶点位置: {apex_pos}")
    print(f"圆锥顶点索引: {apex_idx}")
    
    # 测试默认法向量
    default_normal = mesh_processor.get_normal(apex_idx)
    print(f"默认法向量: {default_normal}")
    
    # 为奇点添加多个法向量
    # 圆锥顶点处的法向量可以是任意指向圆锥外部的方向
    # 我们添加几个不同的法向量
    normal1 = np.array([1, 0, 1])  # 指向x+z方向
    normal2 = np.array([0, 1, 1])  # 指向y+z方向
    normal3 = np.array([-1, 0, 1])  # 指向-x+z方向
    normal4 = np.array([0, -1, 1])  # 指向-y+z方向
    
    mesh_processor.add_normal(apex_idx, normal1)
    mesh_processor.add_normal(apex_idx, normal2)
    mesh_processor.add_normal(apex_idx, normal3)
    mesh_processor.add_normal(apex_idx, normal4)
    
    # 获取所有法向量
    all_normals = mesh_processor.get_normals(apex_idx)
    print(f"奇点处的所有法向量: {len(all_normals)}")
    for i, normal in enumerate(all_normals):
        print(f"法向量 {i}: {normal}")
    
    # 更新网格法向量
    print("更新网格法向量...")
    mesh_processor.update_mesh_normals()
    print("网格法向量更新完成")
    
    # 验证更新后的法向量
    updated_normals = np.asarray(mesh.vertex_normals)
    print(f"更新后顶点法向量数量: {len(updated_normals)}")
    print(f"顶点 {apex_idx} 的更新后法向量: {updated_normals[apex_idx]}")
    
    # 验证与添加的法向量是否一致
    first_normal = all_normals[0]
    print(f"第一个法向量: {first_normal}")
    print(f"更新后法向量: {updated_normals[apex_idx]}")
    print(f"法向量是否一致: {np.allclose(first_normal, updated_normals[apex_idx])}")
    
    # 可视化网格和法向量
    print("可视化网格和法向量...")
    visualizer = Visualizer()
    
    # 可视化网格
    print("显示原始网格...")
    o3d.visualization.draw_geometries([mesh], window_name="原始网格")
    
    # 测试C1连续性分区
    print("测试C1连续性分区...")
    from core.advancedSurfacePartitioner import AdvancedSurfacePartitioner
    from core.nonSphericalTool import NonSphericalTool
    
    # 创建刀具
    tool = NonSphericalTool(profile_type='ellipsoidal', params={'semi_axes': [9.0, 3.0], 'shank_diameter': 6.0})
    
    # 创建分区器
    partitioner = AdvancedSurfacePartitioner(mesh_processor, tool)
    
    # 基于C1连续性进行分区
    partition_labels = partitioner.partition_by_c1_continuity(threshold=0.1)
    
    # 分析分区结果
    unique_labels = np.unique(partition_labels)
    print(f"C1连续性分区数量: {len(unique_labels)}")
    
    # 可视化分区
    print("可视化分区...")
    visualizer.visualize_partitions_with_midpoints(mesh, partition_labels, np.array([]))
    
    # 测试工具方向场生成
    print("测试工具方向场生成...")
    from core.toolOrientationField import ToolOrientationField
    
    # 创建工具方向场生成器
    orientation_field = ToolOrientationField(mesh_processor, partition_labels, tool)
    
    # 生成工具方向场
    tool_orientations = orientation_field.generate_field()
    
    # 可视化工具方向场
    print("可视化工具方向场...")
    visualizer.visualize_tool_orientations(mesh, tool_orientations)
    
    print("测试完成！法向量相关修改与可视化脚本适配正常。")


if __name__ == "__main__":
    test_normal_visualization()
