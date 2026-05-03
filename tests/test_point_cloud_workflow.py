import numpy as np
import os
import sys
import open3d as o3d

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meshProcessor import MeshProcessor
from core.nonSphericalTool import NonSphericalTool
from core.advancedSurfacePartitioner import AdvancedSurfacePartitioner
from core.toolOrientationField import ToolOrientationField
from core.isoScallopField import IsoScallopFieldGenerator
from core.pathGenerator import PathGenerator
from utils.visualization import Visualizer

def test_point_cloud_workflow():
    """
    测试点云数据是否能够适配完整的工作流程
    """
    print("=== 测试点云数据工作流程 ===")
    
    # 1. 加载点云
    print("1. 加载点云...")
    ply_file = r"D:\Projects\lpSurface\GM\data\models\FloorChair(1).ply"
    
    if not os.path.exists(ply_file):
        print(f"错误：文件 {ply_file} 不存在")
        return False
    
    point_cloud = o3d.io.read_point_cloud(ply_file)
    print(f"成功加载点云：{len(point_cloud.points)} 个点")
    
    # 2. 重建网格
    print("2. 重建网格...")
    try:
        # 使用泊松重建算法
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud, depth=8
        )
        
        # 裁剪低密度区域
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print(f"网格重建完成：{len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个三角形")
        
        # 计算法线
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        # 跳过可视化步骤
        print("3. 跳过可视化步骤...")
        
    except Exception as e:
        print(f"网格重建失败：{e}")
        return False
    
    # 3. 创建网格处理器
    print("4. 创建网格处理器...")
    mesh_processor = MeshProcessor(mesh)
    
    # 4. 计算几何特性
    print("5. 计算几何特性...")
    mesh_processor._estimate_curvatures()
    mesh_processor.principal_curvatures = np.zeros((len(mesh_processor.vertices), 2))
    for i in range(len(mesh_processor.vertices)):
        mesh_processor.principal_curvatures[i] = [mesh_processor.curvatures[i], mesh_processor.curvatures[i]]
    mesh_processor.gaussian_curvatures = mesh_processor.curvatures
    
    # 5. 创建刀具
    print("6. 创建刀具...")
    tool = NonSphericalTool(
        profile_type='ellipsoidal',
        params={'semi_axes': [9.0, 3.0], 'shank_diameter': 6.0}
    )
    
    # 6. 计算切削宽度和直纹面逼近误差
    print("7. 计算切削宽度和直纹面逼近误差...")
    mesh_processor.calculate_max_cutting_width(tool)
    mesh_processor.calculate_rolled_error()
    
    # 7. 创建分区器
    print("8. 创建分区器...")
    partitioner = AdvancedSurfacePartitioner(
        mesh_processor,
        tool,
        resolution=0.05,
        alpha=0.3,
        global_field='rolled_error'
    )
    
    # 8. 执行分区
    print("9. 执行分区...")
    try:
        labels, edge_midpoints = partitioner.partition_surface()
        num_partitions = len(np.unique(labels))
        print(f"分区完成：{num_partitions} 个分区")
        print(f"边缘中点数量：{len(edge_midpoints)}")
        
        # 跳过可视化步骤
        print("10. 跳过可视化步骤...")
        
    except Exception as e:
        print(f"分区失败：{e}")
        return False
    
    # 9. 生成刀具方向场
    print("11. 生成刀具方向场...")
    try:
        orientation_field = ToolOrientationField(
            mesh_processor,
            labels,
            tool
        )
        tool_orientations = orientation_field.generate_field()
        print("刀具方向场生成完成")
        
        # 跳过可视化步骤
        print("12. 跳过可视化步骤...")
        
    except Exception as e:
        print(f"方向场生成失败：{e}")
        return False
    
    # 10. 生成等残留高度场
    print("13. 生成等残留高度场...")
    try:
        iso_scallop_generator = IsoScallopFieldGenerator(
            mesh_processor,
            tool_orientations,
            tool,
            scallop_height=0.4
        )
        scalar_field = iso_scallop_generator.generate_scalar_field()
        print("等残留高度场生成完成")
        
    except Exception as e:
        print(f"等残留高度场生成失败：{e}")
        return False
    
    # 11. 提取等值线
    print("14. 提取等值线...")
    try:
        iso_curves = iso_scallop_generator.extract_iso_curves(scalar_field)
        print("等值线提取完成")
        
    except Exception as e:
        print(f"等值线提取失败：{e}")
        return False
    
    # 12. 生成刀具路径
    print("15. 生成刀具路径...")
    try:
        path_generator = PathGenerator(
            mesh_processor,
            iso_curves,
            tool_orientations,
            tool
        )
        tool_paths = path_generator.generate_final_path()
        print(f"刀具路径生成完成：{len(tool_paths['paths'])} 条路径")
        
        # 跳过可视化步骤
        print("16. 跳过可视化步骤...")
        
    except Exception as e:
        print(f"刀具路径生成失败：{e}")
        return False
    
    # 13. 保存结果
    print("17. 保存结果...")
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", f"test_point_cloud_{timestamp}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存网格
        o3d.io.write_triangle_mesh(os.path.join(output_dir, "floor_chair_mesh.obj"), mesh)
        
        # 保存点云
        o3d.io.write_point_cloud(os.path.join(output_dir, "floor_chair_points.ply"), point_cloud)
        
        # 保存网格数据
        np.save(os.path.join(output_dir, "vertices.npy"), np.asarray(mesh.vertices))
        np.save(os.path.join(output_dir, "triangles.npy"), np.asarray(mesh.triangles))
        
        # 保存分区结果
        np.save(os.path.join(output_dir, "partition_labels.npy"), labels)
        np.save(os.path.join(output_dir, "edge_midpoints.npy"), edge_midpoints)
        
        # 保存方向场
        np.save(os.path.join(output_dir, "orientation_field.npy"), tool_orientations)
        
        # 保存标量场
        np.save(os.path.join(output_dir, "scalar_field.npy"), scalar_field)
        
        # 保存刀具路径
        import json
        tool_paths_json = {
            'paths': []
        }
        for path_data in tool_paths['paths']:
            points = path_data['points']
            if isinstance(points[0], np.ndarray):
                points_list = [point.tolist() for point in points]
            else:
                points_list = points
            
            orientations = path_data.get('orientations', [])
            if orientations and isinstance(orientations[0], np.ndarray):
                orientations_list = [orientation.tolist() for orientation in orientations]
            else:
                orientations_list = orientations
            
            tool_paths_json['paths'].append({
                'points': points_list,
                'orientations': orientations_list
            })
        with open(os.path.join(output_dir, "tool_paths.json"), 'w') as f:
            json.dump(tool_paths_json, f, indent=2)
        
        # 保存指标
        metrics = {
            'num_partitions': num_partitions,
            'num_tool_paths': len(tool_paths['paths']),
            'num_vertices': len(mesh.vertices),
            'num_triangles': len(mesh.triangles),
            'num_points': len(point_cloud.points)
        }
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"结果保存到：{output_dir}")
        
    except Exception as e:
        print(f"保存结果失败：{e}")
        return False
    
    print("=== 点云数据工作流程测试完成 ===")
    return True

if __name__ == "__main__":
    success = test_point_cloud_workflow()
    if success:
        print("测试成功！点云数据能够适配完整的工作流程")
    else:
        print("测试失败！点云数据无法适配完整的工作流程")
