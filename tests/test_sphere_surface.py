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


def create_sphere_points(radius=1.0, resolution=50):
    """
    通过参数方程创建球面点云
    Args:
        radius: 球半径
        resolution: 采样分辨率
    Returns:
        points: 点云
        normals: 法向量
        gaussian_curvatures: 高斯曲率
        principal_curvatures: 主曲率
    """
    points = []
    normals = []
    gaussian_curvatures = []
    principal_curvatures = []
    
    # 使用球坐标系采样
    for i in range(resolution + 1):
        theta = np.pi * i / resolution  # 极角
        for j in range(resolution + 1):
            phi = 2 * np.pi * j / resolution  # 方位角
            
            # 参数方程
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            
            # 法向量（球面的法向量指向径向）
            normal = np.array([x, y, z]) / radius
            
            # 球面的高斯曲率和主曲率（解析解）
            K = 1 / (radius**2)  # 高斯曲率
            k1 = 1 / radius  # 主曲率1
            k2 = 1 / radius  # 主曲率2
            
            points.append([x, y, z])
            normals.append(normal)
            gaussian_curvatures.append(K)
            principal_curvatures.append([k1, k2])
    
    return np.array(points), np.array(normals), np.array(gaussian_curvatures), np.array(principal_curvatures)


def create_mesh_from_points(points):
    """
    从点云创建网格
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 估计法向量，使用较大的搜索半径以获得更一致的法线方向
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # 法线方向一致性调整
    pcd.orient_normals_consistent_tangent_plane(100)
    
    # 使用Ball Pivoting Algorithm (BPA)创建表面网格
    # BPA更适合创建表面网格，不会生成内部面
    print("使用BPA算法创建网格...")
    # 计算点云密度，用于确定球半径
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [avg_dist * 1.5, avg_dist * 2.0, avg_dist * 2.5]
    
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        print(f"BPA网格创建完成: {len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个三角形")
        
        # 移除BPA创建的多余平面
        # 计算每个三角形的中心
        triangle_centers = []
        for triangle in mesh.triangles:
            v0 = mesh.vertices[triangle[0]]
            v1 = mesh.vertices[triangle[1]]
            v2 = mesh.vertices[triangle[2]]
            center = (v0 + v1 + v2) / 3
            triangle_centers.append(center)
        
        # 只保留外部的三角形
        filtered_triangles = []
        for i, center in enumerate(triangle_centers):
            center_np = np.array(center)
            # 对于球面，检查点到原点的距离
            center_dist = np.linalg.norm(center_np)
            if center_dist > 0.95 and center_dist < 1.05:  # 只保留球面表面的三角形
                filtered_triangles.append(mesh.triangles[i])
        
        # 创建新的网格
        if filtered_triangles:
            new_mesh = o3d.geometry.TriangleMesh()
            new_mesh.vertices = mesh.vertices
            new_mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)
            new_mesh.compute_vertex_normals()
            mesh = new_mesh
            print(f"过滤后网格: {len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个三角形")
        else:
            mesh.compute_vertex_normals()
    except Exception as e:
        print(f"BPA算法失败: {e}，使用泊松重建作为备选")
        # 使用泊松重建作为备选
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
        
        # 使用点云的边界框裁剪
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
        
        # 移除内部面和多余的平面
        # 计算每个三角形的中心
        triangle_centers = []
        for triangle in mesh.triangles:
            v0 = mesh.vertices[triangle[0]]
            v1 = mesh.vertices[triangle[1]]
            v2 = mesh.vertices[triangle[2]]
            center = (v0 + v1 + v2) / 3
            triangle_centers.append(center)
        
        # 只保留外部的三角形
        filtered_triangles = []
        for i, center in enumerate(triangle_centers):
            center_np = np.array(center)
            # 对于球面，检查点到原点的距离
            center_dist = np.linalg.norm(center_np)
            if center_dist > 0.95 and center_dist < 1.05:  # 只保留球面表面的三角形
                filtered_triangles.append(mesh.triangles[i])
        
        # 创建新的网格
        if filtered_triangles:
            new_mesh = o3d.geometry.TriangleMesh()
            new_mesh.vertices = mesh.vertices
            new_mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)
            new_mesh.compute_vertex_normals()
            mesh = new_mesh
        else:
            mesh.compute_vertex_normals()
    
    # 清理网格
    # 移除重复顶点
    mesh.remove_duplicated_vertices()
    # 移除重复三角形
    mesh.remove_duplicated_triangles()
    # 移除非流形边
    mesh.remove_non_manifold_edges()
    
    print(f"最终网格: {len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个三角形")
    return mesh


def run_test():
    """
    运行球面测试
    """
    print("=== 球面测试 ===")
    
    # 1. 创建球面点云
    print("1. 创建球面点云...")
    points, normals, gaussian_curvatures, principal_curvatures = create_sphere_points(
        radius=1.0, 
        resolution=30
    )
    print(f"采样完成，得到 {len(points)} 个点")
    
    # 2. 创建网格
    print("2. 创建网格...")
    mesh = create_mesh_from_points(points)
    print(f"网格创建完成: {len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个三角形")
    
    # 3. 创建网格处理器
    print("3. 创建网格处理器...")
    mesh_processor = MeshProcessor(mesh)
    
    # 4. 添加几何特性
    print("4. 添加几何特性...")
    # 为所有网格顶点计算几何特性
    mesh_vertices = np.asarray(mesh.vertices)
    vertex_normals = []
    gaussian_curvatures = []
    principal_curvatures = []
    
    for vertex in mesh_vertices:
        # 计算球面上点的法向量（指向径向）
        normal = vertex / np.linalg.norm(vertex)
        
        # 球面的高斯曲率和主曲率（解析解）
        K = 1 / (1.0**2)  # 高斯曲率
        k1 = 1 / 1.0  # 主曲率1
        k2 = 1 / 1.0  # 主曲率2
        
        vertex_normals.append(normal)
        gaussian_curvatures.append(K)
        principal_curvatures.append([k1, k2])
    
    mesh_processor.vertex_normals = np.array(vertex_normals)
    mesh_processor.gaussian_curvatures = np.array(gaussian_curvatures)
    mesh_processor.principal_curvatures = np.array(principal_curvatures)
    
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
    # 生成简单的等值线
    iso_curves = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        partition_vertices = np.where(labels == label)[0]
        if len(partition_vertices) > 3:
            curve = []
            for i in range(0, len(partition_vertices), 5):
                vertex_idx = partition_vertices[i]
                curve.append(mesh_processor.vertices[vertex_idx])
            if len(curve) > 1:
                iso_curves.append(curve)
    
    # 使用顶点法向量作为工具方向
    tool_orientations = mesh_processor.vertex_normals
    
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
    print(f"- 点云数量: {len(points)}")
    print(f"- 网格顶点数: {len(mesh.vertices)}")
    print(f"- 分区数量: {num_partitions}")
    print(f"- 刀具路径数: {len(tool_paths['paths'])}")
    print(f"- 路径总长度: {total_length:.2f} mm")
    
    # 13. 保存结果
    print("13. 保存结果...")
    output_dir = os.path.join("output", "test_sphere")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
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
