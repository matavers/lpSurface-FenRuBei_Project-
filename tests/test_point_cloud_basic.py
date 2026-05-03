import numpy as np
import os
import sys
import open3d as o3d

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meshProcessor import MeshProcessor
from core.nonSphericalTool import NonSphericalTool

def test_point_cloud_basic():
    """
    测试点云数据的基本处理，验证是否能够适配工作流程
    """
    print("=== 测试点云数据基本处理 ===")
    
    # 1. 加载点云
    print("1. 加载点云...")
    ply_file = r"D:\Projects\lpSurface\GM\data\models\FloorChair(1).ply"
    
    if not os.path.exists(ply_file):
        print(f"错误：文件 {ply_file} 不存在")
        return False
    
    point_cloud = o3d.io.read_point_cloud(ply_file)
    print(f"成功加载点云：{len(point_cloud.points)} 个点")
    
    # 2. 检查点云属性
    print("2. 检查点云属性...")
    if not point_cloud.has_points():
        print("错误：点云没有点数据")
        return False
    
    # 计算法线
    print("3. 计算点云法线...")
    point_cloud.estimate_normals()
    if not point_cloud.has_normals():
        print("错误：无法计算点云法线")
        return False
    print("点云法线计算完成")
    
    # 3. 简化点云（可选）
    print("4. 简化点云...")
    downsampled = point_cloud.voxel_down_sample(voxel_size=0.1)
    print(f"点云简化完成：{len(downsampled.points)} 个点")
    
    # 4. 重建网格
    print("5. 重建网格...")
    try:
        # 使用泊松重建算法
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            downsampled, depth=7  # 降低深度以减少计算量
        )
        
        # 裁剪低密度区域
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print(f"网格重建完成：{len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个三角形")
        
        # 计算法线
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
    except Exception as e:
        print(f"网格重建失败：{e}")
        return False
    
    # 5. 创建网格处理器
    print("6. 创建网格处理器...")
    try:
        mesh_processor = MeshProcessor(mesh)
        print("网格处理器创建成功")
    except Exception as e:
        print(f"网格处理器创建失败：{e}")
        return False
    
    # 6. 计算几何特性
    print("7. 计算几何特性...")
    try:
        mesh_processor._estimate_curvatures()
        print("曲率计算完成")
        
        # 初始化其他几何特性
        mesh_processor.principal_curvatures = np.zeros((len(mesh_processor.vertices), 2))
        for i in range(len(mesh_processor.vertices)):
            mesh_processor.principal_curvatures[i] = [mesh_processor.curvatures[i], mesh_processor.curvatures[i]]
        mesh_processor.gaussian_curvatures = mesh_processor.curvatures
        print("几何特性初始化完成")
    except Exception as e:
        print(f"几何特性计算失败：{e}")
        return False
    
    # 7. 创建刀具
    print("8. 创建刀具...")
    try:
        tool = NonSphericalTool(
            profile_type='ellipsoidal',
            params={'semi_axes': [9.0, 3.0], 'shank_diameter': 6.0}
        )
        print("刀具创建成功")
    except Exception as e:
        print(f"刀具创建失败：{e}")
        return False
    
    # 8. 计算切削宽度和直纹面逼近误差
    print("9. 计算切削宽度和直纹面逼近误差...")
    try:
        mesh_processor.calculate_max_cutting_width(tool)
        mesh_processor.calculate_rolled_error()
        print("切削宽度和直纹面逼近误差计算完成")
    except Exception as e:
        print(f"计算失败：{e}")
        return False
    
    # 9. 保存测试结果
    print("10. 保存测试结果...")
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("output", f"test_point_cloud_basic_{timestamp}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存点云
        o3d.io.write_point_cloud(os.path.join(output_dir, "floor_chair_points.ply"), point_cloud)
        
        # 保存简化后的点云
        o3d.io.write_point_cloud(os.path.join(output_dir, "floor_chair_points_downsampled.ply"), downsampled)
        
        # 保存网格
        o3d.io.write_triangle_mesh(os.path.join(output_dir, "floor_chair_mesh.obj"), mesh)
        
        # 保存网格数据
        np.save(os.path.join(output_dir, "vertices.npy"), np.asarray(mesh.vertices))
        np.save(os.path.join(output_dir, "triangles.npy"), np.asarray(mesh.triangles))
        
        # 保存指标
        metrics = {
            'num_points': len(point_cloud.points),
            'num_points_downsampled': len(downsampled.points),
            'num_vertices': len(mesh.vertices),
            'num_triangles': len(mesh.triangles)
        }
        import json
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"测试结果保存到：{output_dir}")
        
    except Exception as e:
        print(f"保存测试结果失败：{e}")
        return False
    
    print("=== 点云数据基本处理测试完成 ===")
    return True

if __name__ == "__main__":
    success = test_point_cloud_basic()
    if success:
        print("测试成功！点云数据能够适配基本处理流程")
    else:
        print("测试失败！点云数据无法适配基本处理流程")
