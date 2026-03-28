"""
测试球面
通过参数方程直接采样点云，并使用解析公式计算几何特性
"""

import numpy as np
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meshProcessor import MeshProcessor
from core.nonSphericalTool import NonSphericalTool
from core.advancedSurfacePartitioner import AdvancedSurfacePartitioner
from core.pathGenerator import PathGenerator
from utils.visualization import Visualizer
import open3d as o3d


def create_sphere_mesh(radius=1.0, resolution=50):
    """
    通过参数方程直接生成球面三角网格
    Args:
        radius: 球半径
        resolution: 采样分辨率
    Returns:
        mesh: Open3D网格对象
        vertex_params: 每个顶点对应的参数 (theta, phi) 列表
    """
    vertices = []
    vertex_params = []  # 存储每个顶点的参数 (theta, phi)
    
    # 1. 生成球面顶点
    # 使用球坐标系，theta从0到pi，phi从0到2pi
    n_theta = resolution + 1
    n_phi = resolution + 1
    
    for i in range(n_theta):
        theta = np.pi * i / resolution  # 极角
        for j in range(n_phi):
            phi = 2 * np.pi * j / resolution  # 方位角
            
            # 参数方程
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            
            vertices.append([x, y, z])
            vertex_params.append((theta, phi))
    
    # 2. 构造三角形索引
    triangles = []
    # 处理非极点区域（四边形拆分为两个三角形）
    for i in range(n_theta - 1):  # 从第0层到第resolution-1层
        for j in range(resolution):  # 每层有 resolution 个四边形（因为首尾点重复，但只连到第一个）
            # 当前层索引
            idx_00 = i * n_phi + j
            idx_01 = i * n_phi + (j + 1)
            # 下一层索引
            idx_10 = (i + 1) * n_phi + j
            idx_11 = (i + 1) * n_phi + (j + 1)
            
            # 第一个三角形 (左下-右下-左上)
            triangles.append([idx_00, idx_01, idx_10])
            # 第二个三角形 (右上-左上-右下)
            triangles.append([idx_01, idx_11, idx_10])
    
    # 创建Open3D网格对象
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.compute_vertex_normals()
    
    return mesh, vertex_params


def compute_sphere_properties(mesh, vertex_params, radius=1.0):
    """
    为每个顶点计算解析几何特性（法向量、高斯曲率、主曲率）
    Args:
        mesh: Open3D网格
        vertex_params: 每个顶点的 (theta, phi) 参数列表
        radius: 球半径
    Returns:
        normals, gaussian_curvatures, principal_curvatures
    """
    vertices = np.asarray(mesh.vertices)
    n_vertices = len(vertices)
    normals = np.zeros((n_vertices, 3))
    gaussian_curvatures = np.zeros(n_vertices)
    principal_curvatures = np.zeros((n_vertices, 2))
    
    # 球面的高斯曲率和主曲率（解析解）
    K = 1 / (radius**2)  # 高斯曲率
    k = 1 / radius  # 主曲率（球面的两个主曲率相等）
    
    for i, (theta, phi) in enumerate(vertex_params):
        # 法向量（球面的法向量指向径向）
        normal = vertices[i] / radius
        normals[i] = normal
        
        # 高斯曲率
        gaussian_curvatures[i] = K
        
        # 主曲率
        principal_curvatures[i] = [k, k]
    
    return normals, gaussian_curvatures, principal_curvatures


def run_test():
    """
    运行球面测试
    """
    print("=== 球面测试 ===")
    
    # 1. 创建球面点云
    print("1. 创建球面点云...")
    # 使用参数方程直接生成球面网格
    mesh, vertex_params = create_sphere_mesh(
        radius=1.0, 
        resolution=95
    )
    print(f"采样完成，得到 {len(mesh.vertices)} 个点")
    
    # 2. 创建网格
    print("2. 创建网格...")
    print(f"网格创建完成: {len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个三角形")
    
    # 3. 创建网格处理器
    print("3. 创建网格处理器...")
    mesh_processor = MeshProcessor(mesh)
    
    # 4. 添加几何特性
    print("4. 添加几何特性...")
    # 使用解析方法计算几何特性
    normals, gaussian_curvatures, principal_curvatures = compute_sphere_properties(
        mesh, vertex_params, radius=1.0
    )
    
    mesh_processor.vertex_normals = normals
    mesh_processor.gaussian_curvatures = gaussian_curvatures
    mesh_processor.principal_curvatures = principal_curvatures
    
    # 5. 创建刀具
    print("5. 创建刀具...")
    tool = NonSphericalTool(
        profile_type='ellipsoidal',
        params={'semi_axes': [0.5, 0.25]}
    )
    
    # 6. 计算切削宽度和直纹面逼近误差
    print("6. 计算切削宽度和直纹面逼近误差...")
    mesh_processor.calculate_max_cutting_width(tool)
    mesh_processor.calculate_rolled_error()
    
    # 7. 创建分区器
    print("7. 创建分区器...")
    partitioner = AdvancedSurfacePartitioner(
        mesh_processor, 
        tool, 
        resolution=0.1, 
        alpha=0.3,  # 全局引导强度
        global_field='rolled_error'  # 使用直纹面逼近误差作为全局场
    )
    
    # 8. 执行分区
    print("8. 执行分区...")
    labels, edge_midpoints = partitioner.partition_surface()
    num_partitions = len(np.unique(labels))
    print(f"分区完成: {num_partitions} 个分区")
    
    # 9. 生成刀具路径
    print("9. 生成刀具路径...")
    # 生成简    # 使用IsoScallopFieldGenerator生成标量场并提取真正的等距线
    from core.isoScallopField import IsoScallopFieldGenerator
    tool_orientations = mesh_processor.vertex_normals
    iso_field = IsoScallopFieldGenerator(mesh_processor, tool_orientations, tool, scallop_height=0.4)
    scalar_field = iso_field.generate_scalar_field()
    iso_curves = iso_field.extract_iso_curves(scalar_field, spacing=0.05)
    
    # 创建路径生成器
    path_generator = PathGenerator(mesh_processor, iso_curves, tool_orientations, tool)
    tool_paths = path_generator.generate_final_path()
    
    # 10. 计算路径总长度
    total_length = 0
    for path_data in tool_paths['paths']:
        path_points = path_data['points']
        for i in range(len(path_points) - 1):
            total_length += np.linalg.norm(path_points[i+1] - path_points[i])
    
    print(f"刀具路径生成完成: {len(tool_paths['paths'])} 条路径, 总长度 {total_length:.2f} mm")
    
    # 11. 可视化结果
    print("11. 可视化结果...")
    visualizer = Visualizer()
    
    # 可视化分区情况
    visualizer.visualize_partitions_with_midpoints(mesh, labels, edge_midpoints)
    
    # 可视化刀具方向场
    visualizer.visualize_tool_orientations(mesh, tool_orientations, scale=0.1)
    
    # 可视化刀具路径
    visualizer.visualize_tool_paths(tool_paths['paths'])
    
    # 12. 验证结果
    print("12. 验证结果...")
    print(f"- 点云数量: {len(mesh.vertices)}")
    print(f"- 网格顶点数: {len(mesh.vertices)}")
    print(f"- 分区数量: {num_partitions}")
    print(f"- 刀具路径数: {len(tool_paths['paths'])}")
    print(f"- 路径总长度: {total_length:.2f} mm")
    
    # 13. 保存结果
    print("13. 保存结果...")
    output_dir = os.path.join("output", "test_sphere")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存点云（从网格顶点生成）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(os.path.join(output_dir, "sphere_points.ply"), pcd)
    
    # 保存网格
    o3d.io.write_triangle_mesh(os.path.join(output_dir, "sphere_mesh.obj"), mesh)
    
    # 保存网格数据（用于visualize_results.py）
    np.save(os.path.join(output_dir, "vertices.npy"), np.asarray(mesh.vertices))
    np.save(os.path.join(output_dir, "triangles.npy"), np.asarray(mesh.triangles))
    
    # 保存分区结果
    np.save(os.path.join(output_dir, "partition_labels.npy"), labels)
    np.save(os.path.join(output_dir, "edge_midpoints.npy"), edge_midpoints)
    
    # 保存方向场（用于visualize_results.py）
    np.save(os.path.join(output_dir, "orientation_field.npy"), tool_orientations)
    
    # 保存刀具路径（用于visualize_results.py）
    import json
    tool_paths_json = {
        'paths': []
    }
    for path_data in tool_paths['paths']:
        tool_paths_json['paths'].append({
            'points': path_data['points'].tolist(),
            'orientations': path_data.get('orientations', []).tolist()
        })
    with open(os.path.join(output_dir, "tool_paths.json"), 'w') as f:
        json.dump(tool_paths_json, f, indent=2)
    
    # 保存指标（用于visualize_results.py）
    metrics = {
        'num_partitions': num_partitions,
        'num_tool_paths': len(tool_paths['paths']),
        'total_path_length': total_length,
        'num_vertices': len(mesh.vertices),
        'num_triangles': len(mesh.triangles)
    }
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"结果保存到: {output_dir}")
    print("=== 球面测试完成 ===")


if __name__ == "__main__":
    run_test()
