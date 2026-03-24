"""
测试圆锥面
通过参数方程直接生成网格，并使用解析公式计算几何特性
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


def create_cone_mesh(height=2.0, radius=1.0, resolution=50):
    """
    通过参数方程直接生成圆锥面三角网格
    Args:
        height: 圆锥高度
        radius: 圆锥底面半径
        resolution: 角度方向采样数（高度方向也使用相同分辨率）
    Returns:
        mesh: Open3D网格对象
        vertex_params: 每个顶点对应的参数 (t, theta) 列表
    """
    vertices = []
    vertex_params = []  # 存储每个顶点的参数 (t, theta)
    
    # 1. 生成底部到顶部（不含顶部）的顶点
    # 每一层有 resolution+1 个点，首尾角度相同（theta=0 和 theta=2π），以保证网格闭合
    n_theta = resolution + 1
    n_height = resolution  # 底部到顶部前一层的高度层数
    
    for i in range(n_height):
        t = i / resolution  # 高度参数 0 ≤ t < 1
        r = radius * (1 - t)
        for j in range(n_theta):
            theta = 2 * np.pi * j / resolution
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = height * t
            vertices.append([x, y, z])
            vertex_params.append((t, theta))
    
    # 2. 添加顶部点（只有一个）
    top_idx = len(vertices)
    vertices.append([0, 0, height])
    vertex_params.append((1.0, 0.0))  # 顶部点参数 t=1, theta 任意
    
    # 3. 构造三角形索引
    triangles = []
    # 处理非顶部区域（四边形拆分为两个三角形）
    for i in range(n_height - 1):  # 从第0层到第resolution-2层
        for j in range(resolution):  # 每层有 resolution 个四边形（因为首尾点重复，但只连到第一个）
            # 当前层索引
            idx_00 = i * n_theta + j
            idx_01 = i * n_theta + (j + 1)
            # 下一层索引
            idx_10 = (i + 1) * n_theta + j
            idx_11 = (i + 1) * n_theta + (j + 1)
            
            # 第一个三角形 (左下-右下-左上)
            triangles.append([idx_00, idx_01, idx_10])
            # 第二个三角形 (右上-左上-右下)
            triangles.append([idx_01, idx_11, idx_10])
    
    # 处理顶层（第 resolution-1 层）与顶部点的连接
    i = n_height - 1  # 最后一层（高度接近顶部）
    for j in range(resolution):
        idx_00 = i * n_theta + j
        idx_01 = i * n_theta + (j + 1)
        # 与顶部点构成三角形
        triangles.append([idx_00, idx_01, top_idx])
    
    # 创建Open3D网格对象
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.compute_vertex_normals()  # 后续将用解析法向量覆盖，但先计算默认值
    
    return mesh, vertex_params


def compute_vertex_properties(mesh, vertex_params, height=2.0, radius=1.0):
    """
    为每个顶点计算解析几何特性（法向量、高斯曲率、主曲率）
    Args:
        mesh: Open3D网格
        vertex_params: 每个顶点的 (t, theta) 参数列表
        height: 圆锥高度
        radius: 圆锥底面半径
    Returns:
        normals, gaussian_curvatures, principal_curvatures
    """
    vertices = np.asarray(mesh.vertices)
    normals = []
    gaussian_curvatures = []
    principal_curvatures = []
    
    for idx, (t, theta) in enumerate(vertex_params):
        # 对于顶部点单独处理
        if t >= 1.0:
            normals.append(np.array([0, 0, 1]))
            gaussian_curvatures.append(0.0)
            principal_curvatures.append([0.0, 0.0])
            continue
        
        # 计算圆锥面上的点坐标（可用于验证）
        r = radius * (1 - t)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = height * t
        
        # 一阶偏导数
        dr_dt = -radius
        dx_dt = dr_dt * np.cos(theta)
        dy_dt = dr_dt * np.sin(theta)
        dz_dt = height
        
        dx_dtheta = -r * np.sin(theta)
        dy_dtheta = r * np.cos(theta)
        dz_dtheta = 0
        
        # 法向量
        du = np.array([dx_dt, dy_dt, dz_dt])
        dv = np.array([dx_dtheta, dy_dtheta, dz_dtheta])
        normal = np.cross(du, dv)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-6:
            normal = np.array([0, 0, 1])
        else:
            normal /= normal_norm
        
        # 二阶偏导数
        d2x_dt2 = 0
        d2y_dt2 = 0
        d2z_dt2 = 0
        
        d2x_dtheta2 = -r * np.cos(theta)
        d2y_dtheta2 = -r * np.sin(theta)
        d2z_dtheta2 = 0
        
        d2x_dt_dtheta = -dr_dt * np.sin(theta)
        d2y_dt_dtheta = dr_dt * np.cos(theta)
        d2z_dt_dtheta = 0
        
        # 第一基本形式系数
        E = np.dot(du, du)
        F = np.dot(du, dv)
        G = np.dot(dv, dv)
        
        # 第二基本形式系数
        d2r_dt2 = np.array([d2x_dt2, d2y_dt2, d2z_dt2])
        d2r_dtheta2 = np.array([d2x_dtheta2, d2y_dtheta2, d2z_dtheta2])
        d2r_dt_dtheta = np.array([d2x_dt_dtheta, d2y_dt_dtheta, d2z_dt_dtheta])
        
        L = np.dot(d2r_dt2, normal)
        M = np.dot(d2r_dt_dtheta, normal)
        N = np.dot(d2r_dtheta2, normal)
        
        # 高斯曲率
        denominator = E * G - F**2
        if abs(denominator) < 1e-6:
            K = 0
        else:
            K = (L * N - M**2) / denominator
        
        # 平均曲率
        if abs(denominator) < 1e-6:
            H = 0
        else:
            H = (E * N - 2 * F * M + G * L) / (2 * denominator)
        
        # 主曲率
        discriminant = H**2 - K
        if discriminant < 0:
            k1 = H
            k2 = H
        else:
            sqrt_disc = np.sqrt(discriminant)
            k1 = H + sqrt_disc
            k2 = H - sqrt_disc
        
        normals.append(normal)
        gaussian_curvatures.append(K)
        principal_curvatures.append([k1, k2])
    
    return np.array(normals), np.array(gaussian_curvatures), np.array(principal_curvatures)


def run_test():
    """
    运行圆锥面测试
    """
    print("=== 圆锥面测试 ===")
    
    # 1. 直接生成圆锥面网格
    print("1. 直接生成圆锥面网格...")
    height = 2.0
    radius = 1.0
    resolution = 50  # 控制网格密度
    mesh, vertex_params = create_cone_mesh(height, radius, resolution)
    print(f"网格生成完成: {len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个三角形")
    
    # 2. 计算顶点几何特性
    print("2. 计算顶点几何特性...")
    normals, gaussian_curvatures, principal_curvatures = compute_vertex_properties(
        mesh, vertex_params, height, radius
    )
    
    # 3. 创建网格处理器
    print("3. 创建网格处理器...")
    mesh_processor = MeshProcessor(mesh)
    mesh_processor.vertex_normals = normals
    mesh_processor.gaussian_curvatures = gaussian_curvatures
    mesh_processor.principal_curvatures = principal_curvatures
    
    # 4. 创建刀具
    print("4. 创建刀具...")
    tool = NonSphericalTool(
        profile_type='ellipsoidal',
        params={'semi_axes': [0.5, 0.25]}
    )
    
    # 5. 计算切削宽度和直纹面逼近误差
    print("5. 计算切削宽度和直纹面逼近误差...")
    mesh_processor.calculate_max_cutting_width(tool)
    mesh_processor.calculate_rolled_error()
    
    # 6. 创建分区器
    print("6. 创建分区器...")
    partitioner = AdvancedSurfacePartitioner(
        mesh_processor, 
        tool, 
        resolution=0.5, 
        alpha=0.7,  # 增加全局引导强度
        global_field='rolled_error'  # 使用直纹面逼近误差作为全局场
    )
    
    # 7. 执行分区
    print("7. 执行分区...")
    labels, edge_midpoints = partitioner.partition_surface()
    num_partitions = len(np.unique(labels))
    print(f"分区完成: {num_partitions} 个分区")
    
    # 8. 生成刀具路径（简化示例）
    print("8. 生成刀具路径...")
    # 生成简单的等值线（基于分区标签）
    iso_curves = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        partition_vertices = np.where(labels == label)[0]
        if len(partition_vertices) > 3:
            curve = []
            # 简单采样：每5个点取一个
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
    
    # 9. 计算路径总长度
    total_length = 0
    for path_data in tool_paths['paths']:
        path_points = path_data['points']
        for i in range(len(path_points) - 1):
            total_length += np.linalg.norm(path_points[i+1] - path_points[i])
    
    print(f"刀具路径生成完成: {len(tool_paths['paths'])} 条路径, 总长度 {total_length:.2f} mm")
    
    # 10. 可视化结果
    print("10. 可视化结果...")
    visualizer = Visualizer()
    
    # 可视化分区情况
    visualizer.visualize_partitions_with_midpoints(mesh, labels, edge_midpoints)
    
    # 可视化刀具方向场
    visualizer.visualize_tool_orientations(mesh, tool_orientations, scale=0.1)
    
    # 可视化刀具路径
    visualizer.visualize_tool_paths(tool_paths['paths'])
    
    # 11. 验证结果
    print("11. 验证结果...")
    print(f"- 网格顶点数: {len(mesh.vertices)}")
    print(f"- 网格三角形数: {len(mesh.triangles)}")
    print(f"- 分区数量: {num_partitions}")
    print(f"- 刀具路径数: {len(tool_paths['paths'])}")
    print(f"- 路径总长度: {total_length:.2f} mm")
    
    # 12. 保存结果
    print("12. 保存结果...")
    output_dir = os.path.join("output", "test_cone")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(os.path.join(output_dir, "cone_points.ply"), pcd)
    
    # 保存网格
    o3d.io.write_triangle_mesh(os.path.join(output_dir, "cone_mesh.obj"), mesh)
    
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
    print("=== 圆锥面测试完成 ===")


if __name__ == "__main__":
    run_test()