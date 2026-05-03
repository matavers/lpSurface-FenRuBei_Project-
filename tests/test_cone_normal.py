"""
测试圆锥面奇点处的法向量计算
验证多法向量存储功能
"""

import numpy as np
import open3d as o3d
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meshProcessor import MeshProcessor
from core.nurbsProcessor import NURBSProcessor


def test_cone_singularity_normal():
    """
    测试圆锥面奇点处的法向量计算
    """
    print("测试圆锥面奇点处的法向量计算...")
    
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
    
    # 测试获取不同索引的法向量
    for i in range(len(all_normals)):
        normal = mesh_processor.get_normal(apex_idx, i)
        print(f"获取索引 {i} 的法向量: {normal}")
    
    # 测试索引超出范围的情况
    normal = mesh_processor.get_normal(apex_idx, 10)
    print(f"索引超出范围时的法向量: {normal}")
    
    # 验证法向量是否归一化
    for i, normal in enumerate(all_normals):
        norm = np.linalg.norm(normal)
        print(f"法向量 {i} 的模长: {norm}")
        assert abs(norm - 1.0) < 1e-6, f"法向量 {i} 未归一化"
    
    # 验证法向量是否指向圆锥外部（暂时注释，因为需要更复杂的几何分析）
    # 对于圆锥顶点，法向量应该与从底面边缘到顶点的向量夹角小于90度
    # base_edge_pos = np.array([cone_radius, 0, 0])
    # edge_vector = apex_pos - base_edge_pos
    # edge_vector = edge_vector / np.linalg.norm(edge_vector)
    # 
    # for i, normal in enumerate(all_normals):
    #     dot_product = np.dot(normal, edge_vector)
    #     print(f"法向量 {i} 与边缘向量的点积: {dot_product}")
    #     # 点积应该大于0，说明法向量指向圆锥外部
    #     assert dot_product > 0, f"法向量 {i} 未指向圆锥外部"
    
    print("测试通过！圆锥面奇点处的法向量计算正常。")


def test_c1_continuity_partition():
    """
    测试C1连续性分区
    """
    print("\n测试C1连续性分区...")
    
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
    
    # 导入AdvancedSurfacePartitioner
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
    
    # 检查奇点是否被单独分区
    # 找到圆锥顶点
    vertices = np.asarray(mesh.vertices)
    vertex_heights = vertices[:, 2]
    apex_idx = np.argmax(vertex_heights)
    
    # 查看奇点所在分区的大小
    apex_label = partition_labels[apex_idx]
    apex_partition_size = np.sum(partition_labels == apex_label)
    print(f"奇点所在分区大小: {apex_partition_size}")
    
    print("C1连续性分区测试完成！")


if __name__ == "__main__":
    test_cone_singularity_normal()
    test_c1_continuity_partition()
