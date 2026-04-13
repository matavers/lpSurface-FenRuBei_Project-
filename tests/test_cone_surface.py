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
    resolution = 40  # 控制网格密度，减少到约2000个顶点
    print(f"设置分辨率: {resolution}")
    mesh, vertex_params = create_cone_mesh(height, radius, resolution)
    num_vertices = len(mesh.vertices)
    num_triangles = len(mesh.triangles)
    print(f"网格生成完成: {num_vertices} 个顶点, {num_triangles} 个三角形")
    print(f"顶点数量: {num_vertices}, 目标: 约2000个")
    
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
    # 测试不同的对称性类型
    symmetry_types_list = [
        None,  # 无对称性（使用原有算法）
        ['rotation'],  # 旋转对称
        ['combined']  # 组合对称性
    ]
    
    for symmetry_types in symmetry_types_list:
        print(f"\n--- 测试对称性类型: {symmetry_types} ---")
        
        # 记录开始时间
        import time
        start_time = time.time()
        
        print(f"创建分区器...")
        partitioner = AdvancedSurfacePartitioner(
            mesh_processor, 
            tool, 
            resolution=0.5, 
            alpha=0.7,  # 增加全局引导强度
            global_field='rolled_error',  # 使用直纹面逼近误差作为全局场
            symmetry_types=symmetry_types
        )
        
        # 7. 执行分区
        print("执行分区...")
        print(f"分区前时间: {time.time() - start_time:.2f}秒")
        
        labels, edge_midpoints = partitioner.partition_surface()
        partition_time = time.time() - start_time
        
        num_partitions = len(np.unique(labels))
        print(f"分区完成: {num_partitions} 个分区")
        print(f"分区耗时: {partition_time:.2f}秒")
        print(f"边缘中点数量: {len(edge_midpoints)}")
        
        # 8. 生成刀具路径
        print("8. 生成刀具路径...")
        
        # 8.1 生成刀具方向场
        print("   8.1 生成刀具方向场...")
        from core.toolOrientationField import ToolOrientationField
        orientation_field = ToolOrientationField(
            mesh_processor,
            labels,
            tool
        )
        tool_orientations = orientation_field.generate_field()
        
        # 8.2 生成等残留高度场
        print("   8.2 生成等残留高度场...")
        from core.isoScallopField import IsoScallopFieldGenerator
        iso_scallop_generator = IsoScallopFieldGenerator(
            mesh_processor,
            tool_orientations,
            tool,
            scallop_height=0.05  # 设置残留高度
        )
        scalar_field = iso_scallop_generator.generate_scalar_field()
        
        # 统计残留高度数据
        if scalar_field is not None and len(scalar_field) > 0:
            max_scallop = np.max(scalar_field)
            avg_scallop = np.mean(scalar_field)
            std_scallop = np.std(scalar_field)
            print(f"   残留高度统计: 最大={max_scallop:.4f} mm, 平均={avg_scallop:.4f} mm, 标准差={std_scallop:.4f} mm")
        else:
            max_scallop = 0.0
            avg_scallop = 0.0
            std_scallop = 0.0
            print("   残留高度数据不可用")
        
        # 8.3 提取等值线
        print("   8.3 提取等值线...")
        iso_curves = iso_scallop_generator.extract_iso_curves(scalar_field)
        
        # 8.4 使用PathGenerator生成最终刀具路径
        print("   8.4 使用PathGenerator生成刀具路径...")
        path_generator = PathGenerator(
            mesh_processor,
            iso_curves,
            tool_orientations,
            tool
        )
        tool_paths = path_generator.generate_final_path()
        
        print(f"   生成了 {len(tool_paths['paths'])} 条刀具路径")
        
        # 9. 计算路径总长度和质量指标
        total_length = 0
        path_lengths = []
        path_smoothness = []
        
        for path_data in tool_paths['paths']:
            path_points = path_data['points']
            if len(path_points) < 2:
                continue
            
            # 计算路径长度
            path_length = 0
            for i in range(len(path_points) - 1):
                segment_length = np.linalg.norm(path_points[i+1] - path_points[i])
                path_length += segment_length
            total_length += path_length
            path_lengths.append(path_length)
            
            # 计算路径平滑度（相邻线段的角度变化）
            if len(path_points) >= 3:
                smoothness_sum = 0
                for i in range(1, len(path_points) - 1):
                    v1 = np.array(path_points[i]) - np.array(path_points[i-1])
                    v2 = np.array(path_points[i+1]) - np.array(path_points[i])
                    if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_theta = np.clip(cos_theta, -1, 1)
                        angle = np.arccos(cos_theta)
                        smoothness_sum += angle
                avg_smoothness = smoothness_sum / (len(path_points) - 2) if (len(path_points) - 2) > 0 else 0
                path_smoothness.append(avg_smoothness)
        
        # 计算路径质量指标
        avg_path_length = np.mean(path_lengths) if path_lengths else 0
        avg_smoothness = np.mean(path_smoothness) if path_smoothness else 0
        
        print(f"刀具路径生成完成: {len(tool_paths['paths'])} 条路径, 总长度 {total_length:.2f} mm")
        print(f"路径质量: 平均路径长度={avg_path_length:.2f} mm, 平均平滑度={avg_smoothness:.4f} rad")
        
        # 10. 可视化结果
        print("10. 可视化结果...")
        visualizer = Visualizer()
        
        # 可视化分区情况
        visualizer.visualize_partitions_with_midpoints(mesh, labels, edge_midpoints)
        
        # 可视化刀具方向场
        visualizer.visualize_tool_orientations(mesh, tool_orientations, scale=0.05)
        
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
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        symmetry_str = "_" + "_".join(symmetry_types) if symmetry_types else "_no_symmetry"
        output_dir = os.path.join("output", f"test_cone{symmetry_str}_{timestamp}")
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
            # 处理路径点，确保是列表格式
            points = path_data['points']
            if isinstance(points[0], np.ndarray):
                points_list = [point.tolist() for point in points]
            else:
                points_list = points
            
            # 处理方向，确保是列表格式
            orientations = path_data.get('orientations', [])
            # 检查orientations是否有效
            if orientations is not None:
                # 检查是否是numpy数组
                if isinstance(orientations, np.ndarray):
                    # 如果是二维数组且不为空，转换为列表
                    if len(orientations.shape) == 2 and orientations.size > 0:
                        orientations_list = orientations.tolist()
                    else:
                        orientations_list = []
                else:
                    # 非numpy数组情况
                    try:
                        if orientations and isinstance(orientations[0], np.ndarray):
                            # 如果是numpy数组列表
                            orientations_list = [orientation.tolist() for orientation in orientations]
                        else:
                            # 其他情况
                            orientations_list = orientations
                    except:
                        # 处理可能的索引错误
                        orientations_list = []
            else:
                orientations_list = []
            
            tool_paths_json['paths'].append({
                'points': points_list,
                'orientations': orientations_list
            })
        with open(os.path.join(output_dir, "tool_paths.json"), 'w') as f:
            json.dump(tool_paths_json, f, indent=2)
        
        # 保存指标（用于visualize_results.py）
        metrics = {
            'num_partitions': num_partitions,
            'num_tool_paths': len(tool_paths['paths']),
            'total_path_length': total_length,
            'avg_path_length': avg_path_length,
            'avg_smoothness': avg_smoothness,
            'max_scallop': max_scallop,
            'avg_scallop': avg_scallop,
            'std_scallop': std_scallop,
            'partition_time': partition_time,
            'num_vertices': len(mesh.vertices),
            'num_triangles': len(mesh.triangles),
            'symmetry_types': symmetry_types if symmetry_types else []
        }
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"结果保存到: {output_dir}")

    print("=== 圆锥面测试完成 ===")


if __name__ == "__main__":
    run_test()