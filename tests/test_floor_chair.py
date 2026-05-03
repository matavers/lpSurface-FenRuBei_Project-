"""
测试FloorChair网格
使用重建的网格执行算法流程，不考虑对称性
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


def run_test(skip_visualization=False):
    """
    运行FloorChair测试
    Args:
        skip_visualization: 是否跳过可视化步骤
    """
    print("=== FloorChair测试 ===")
    
    # 1. 加载重建的网格
    print("1. 加载重建的网格...")
    mesh_path = "D:\Projects\lpSurface\GM\data\models\FloorChair(1)_mesh.obj"
    if not os.path.exists(mesh_path):
        print(f"错误: 网格文件不存在: {mesh_path}")
        return
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print("错误: 加载的网格无效")
        return
    
    # 计算法线
    print("计算网格法线...")
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    num_vertices = len(mesh.vertices)
    num_triangles = len(mesh.triangles)
    print(f"网格加载完成: {num_vertices} 个顶点, {num_triangles} 个三角形")
    
    # 2. 创建网格处理器
    print("2. 创建网格处理器...")
    mesh_processor = MeshProcessor(mesh)
    
    # 3. 创建刀具
    print("3. 创建刀具...")
    tool = NonSphericalTool(
        profile_type='ellipsoidal',
        params={'semi_axes': [9.0, 3.0], 'shank_diameter': 6.0}
    )
    
    # 4. 计算切削宽度和直纹面逼近误差
    print("4. 计算切削宽度和直纹面逼近误差...")
    mesh_processor.calculate_max_cutting_width(tool)
    mesh_processor.calculate_rolled_error()
    
    # 5. 创建分区器
    print("5. 创建分区器...")
    partitioner = AdvancedSurfacePartitioner(
        mesh_processor, 
        tool, 
        resolution=0.5, 
        alpha=0.3,  # 全局引导强度
        global_field='rolled_error',  # 使用直纹面逼近误差作为全局场
        symmetry_types=None  # 不考虑对称性
    )
    
    # 6. 执行分区
    print("6. 执行分区...")
    import time
    start_time = time.time()
    
    labels, edge_midpoints = partitioner.partition_surface()
    partition_time = time.time() - start_time
    
    num_partitions = len(np.unique(labels))
    print(f"分区完成: {num_partitions} 个分区")
    print(f"分区耗时: {partition_time:.2f}秒")
    print(f"边缘中点数量: {len(edge_midpoints)}")
    
    # 7. 生成刀具方向场
    print("7. 生成刀具方向场...")
    from core.toolOrientationField import ToolOrientationField
    orientation_field = ToolOrientationField(
        mesh_processor,
        labels,
        tool
    )
    tool_orientations = orientation_field.generate_field()
    
    # 8. 生成等残留高度场
    print("8. 生成等残留高度场...")
    from core.isoScallopField import IsoScallopFieldGenerator
    iso_scallop_generator = IsoScallopFieldGenerator(
        mesh_processor,
        tool_orientations,
        tool,
        scallop_height=0.4  # 设置残留高度
    )
    scalar_field = iso_scallop_generator.generate_scalar_field()
    
    # 统计残留高度数据
    if scalar_field is not None and len(scalar_field) > 0:
        max_scallop = np.max(scalar_field)
        avg_scallop = np.mean(scalar_field)
        std_scallop = np.std(scalar_field)
        print(f"残留高度统计: 最大={max_scallop:.4f} mm, 平均={avg_scallop:.4f} mm, 标准差={std_scallop:.4f} mm")
    else:
        max_scallop = 0.0
        avg_scallop = 0.0
        std_scallop = 0.0
        print("残留高度数据不可用")
    
    # 9. 提取等值线
    print("9. 提取等值线...")
    iso_curves = iso_scallop_generator.extract_iso_curves(scalar_field)
    
    # 10. 使用PathGenerator生成最终刀具路径
    print("10. 使用PathGenerator生成刀具路径...")
    path_generator = PathGenerator(
        mesh_processor,
        iso_curves,
        tool_orientations,
        tool
    )
    tool_paths = path_generator.generate_final_path()
    
    print(f"生成了 {len(tool_paths['paths'])} 条刀具路径")
    
    # 11. 计算路径总长度和质量指标
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
    
    # 12. 可视化结果
    if not skip_visualization:
        print("12. 可视化结果...")
        visualizer = Visualizer()
        
        # 可视化分区情况
        visualizer.visualize_partitions_with_midpoints(mesh, labels, edge_midpoints)
        
        # 可视化刀具方向场
        visualizer.visualize_tool_orientations(mesh, tool_orientations, scale=0.05)
        
        # 可视化刀具路径
        visualizer.visualize_tool_paths(tool_paths['paths'])
    else:
        print("12. 跳过可视化步骤...")
    
    # 13. 验证结果
    print("13. 验证结果...")
    print(f"- 网格顶点数: {len(mesh.vertices)}")
    print(f"- 网格三角形数: {len(mesh.triangles)}")
    print(f"- 分区数量: {num_partitions}")
    print(f"- 刀具路径数: {len(tool_paths['paths'])}")
    print(f"- 路径总长度: {total_length:.2f} mm")
    
    # 14. 保存结果
    print("14. 保存结果...")
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", f"test_floor_chair_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
    o3d.io.write_point_cloud(os.path.join(output_dir, "floor_chair_points.ply"), pcd)
    
    # 保存网格
    o3d.io.write_triangle_mesh(os.path.join(output_dir, "floor_chair_mesh.obj"), mesh)
    
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
        'symmetry_types': []
    }
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"结果保存到: {output_dir}")

    print("=== FloorChair测试完成 ===")


if __name__ == "__main__":
    import sys
    skip_visualization = False
    for arg in sys.argv[1:]:
        if arg == "--skip-visualization":
            skip_visualization = True
    run_test(skip_visualization=skip_visualization)
