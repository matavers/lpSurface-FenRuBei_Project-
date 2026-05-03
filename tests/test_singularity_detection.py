"""
测试奇点检测和分析方法
"""

import numpy as np
import open3d as o3d
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meshProcessor import MeshProcessor
from core.nurbsProcessor import NURBSProcessor


def test_cone_singularity_detection():
    """
    测试圆锥面奇点检测
    """
    print("测试圆锥面奇点检测...")
    
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
    
    # 检测奇点
    singularities = mesh_processor.detect_singularities()
    print(f"检测到 {len(singularities)} 个奇点")
    
    # 分析奇点信息
    singularity_info = mesh_processor.get_singularity_info()
    print(f"奇点信息: {len(singularity_info)} 个奇点")
    
    # 显示奇点信息
    for idx, info in singularity_info.items():
        print(f"奇点 {idx}:")
        print(f"  类型: {info['type']}")
        print(f"  位置: {info['position']}")
        print(f"  曲率: {info['curvature']}")
        print(f"  法向量: {info['normal']}")
        print(f"  法向量数量: {info['num_normals']}")
    
    # 验证是否检测到圆锥顶点
    # 找到圆锥顶点（z值最大的点）
    vertices = np.asarray(mesh.vertices)
    vertex_heights = vertices[:, 2]
    apex_idx = np.argmax(vertex_heights)
    apex_pos = vertices[apex_idx]
    
    print(f"\n圆锥顶点位置: {apex_pos}")
    print(f"圆锥顶点索引: {apex_idx}")
    
    # 检查顶点是否被检测为奇点
    if apex_idx in singularity_info:
        print("✅ 圆锥顶点被正确检测为奇点")
        print(f"  奇点类型: {singularity_info[apex_idx]['type']}")
    else:
        print("❌ 圆锥顶点未被检测为奇点")
    
    # 为奇点添加多个法向量
    print("\n为奇点添加多个法向量...")
    normal1 = np.array([1, 0, 1])  # 指向x+z方向
    normal2 = np.array([0, 1, 1])  # 指向y+z方向
    normal3 = np.array([-1, 0, 1])  # 指向-x+z方向
    normal4 = np.array([0, -1, 1])  # 指向-y+z方向
    
    mesh_processor.add_normal(apex_idx, normal1)
    mesh_processor.add_normal(apex_idx, normal2)
    mesh_processor.add_normal(apex_idx, normal3)
    mesh_processor.add_normal(apex_idx, normal4)
    
    # 更新网格法向量
    mesh_processor.update_mesh_normals()
    
    # 重新获取奇点信息
    singularity_info = mesh_processor.get_singularity_info()
    print(f"\n更新后奇点信息: {len(singularity_info)} 个奇点")
    
    if apex_idx in singularity_info:
        print(f"奇点 {apex_idx} 的法向量数量: {singularity_info[apex_idx]['num_normals']}")
        if singularity_info[apex_idx]['num_normals'] == 5:  # 原始法向量 + 4个新法向量
            print("✅ 多法向量存储正常")
        else:
            print("❌ 多法向量存储异常")
    
    print("\n测试完成！")


def test_sphere_singularity_detection():
    """
    测试球面奇点检测（球面应该没有奇点）
    """
    print("\n测试球面奇点检测...")
    
    # 创建球面NURBS
    sphere_radius = 1.0
    nurbs_processor = NURBSProcessor.create_sphere(radius=sphere_radius, resolution=20)
    
    # 生成球面网格
    mesh = nurbs_processor.generate_mesh(resolution_u=50, resolution_v=50)
    
    # 计算法向量
    mesh.compute_vertex_normals()
    
    # 创建MeshProcessor实例
    mesh_processor = MeshProcessor(mesh)
    
    # 检测奇点
    singularities = mesh_processor.detect_singularities()
    print(f"检测到 {len(singularities)} 个奇点")
    
    # 分析奇点信息
    singularity_info = mesh_processor.get_singularity_info()
    print(f"奇点信息: {len(singularity_info)} 个奇点")
    
    # 球面应该没有奇点
    if len(singularities) == 0:
        print("✅ 球面正确检测为无奇点")
    else:
        print("❌ 球面检测到异常奇点")
        for idx, info in singularity_info.items():
            print(f"  奇点 {idx}: {info['type']} at {info['position']}")
    
    print("\n测试完成！")


def test_cube_singularity_detection():
    """
    测试立方体奇点检测（立方体应该有8个顶点奇点）
    """
    print("\n测试立方体奇点检测...")
    
    # 创建立方体
    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    mesh.compute_vertex_normals()
    
    # 创建MeshProcessor实例
    mesh_processor = MeshProcessor(mesh)
    
    # 检测奇点
    singularities = mesh_processor.detect_singularities()
    print(f"检测到 {len(singularities)} 个奇点")
    
    # 分析奇点信息
    singularity_info = mesh_processor.get_singularity_info()
    print(f"奇点信息: {len(singularity_info)} 个奇点")
    
    # 立方体应该有8个顶点奇点
    if len(singularities) == 8:
        print("✅ 立方体正确检测到8个顶点奇点")
    else:
        print(f"❌ 立方体检测到 {len(singularities)} 个奇点，期望8个")
    
    # 显示奇点类型
    for idx, info in singularity_info.items():
        print(f"  奇点 {idx}: {info['type']}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_cone_singularity_detection()
    test_sphere_singularity_detection()
    test_cube_singularity_detection()
