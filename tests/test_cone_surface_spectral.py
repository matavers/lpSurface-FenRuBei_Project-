"""
圆锥面测试脚本 - 使用谱聚类
"""

import numpy as np
import open3d as o3d
import os
import sys

# 添加根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meshProcessor import MeshProcessor
from core.nonSphericalTool import NonSphericalTool
from core.advancedSurfacePartitioner import AdvancedSurfacePartitioner
from core.pathGenerator import PathGenerator
from utils.visualization import Visualizer


def create_cone_mesh(height=2.0, radius=1.0, resolution=50):
    """
    通过参数方程直接构建圆锥面网格
    Args:
        height: 圆锥高度
        radius: 圆锥底面半径
        resolution: 角度分辨率（圆周和高度均使用相同分辨率）
    Returns:
        mesh: Open3D三角网格
        vertex_params: 每个顶点对应的参数 (t, theta) 列表
    """
    vertices = []
    vertex_params = []  # 存储每个顶点的参数 (t, theta)
    
    n_theta = resolution + 1
    n_height = resolution
    
    for i in range(n_height):
        t = i / resolution
        r = radius * (1 - t)
        for j in range(n_theta):
            theta = 2 * np.pi * j / resolution
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = height * t
            vertices.append([x, y, z])
            vertex_params.append((t, theta))
    
    # 添加顶点
    top_idx = len(vertices)
    vertices.append([0, 0, height])
    vertex_params.append((1.0, 0.0))
    
    triangles = []
    for i in range(n_height - 1):
        for j in range(resolution):
            idx_00 = i * n_theta + j
            idx_01 = i * n_theta + (j + 1)
            idx_10 = (i + 1) * n_theta + j
            idx_11 = (i + 1) * n_theta + (j + 1)
            triangles.append([idx_00, idx_01, idx_10])
            triangles.append([idx_01, idx_11, idx_10])
    
    # 连接最后一层到顶点
    i = n_height - 1
    for j in range(resolution):
        idx_00 = i * n_theta + j
        idx_01 = i * n_theta + (j + 1)
        triangles.append([idx_00, idx_01, top_idx])
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.compute_vertex_normals()
    
    return mesh, vertex_params


def compute_vertex_properties(mesh, vertex_params, height=2.0, radius=1.0):
    """
    为每个顶点计算解析几何性质（法向量、高斯曲率、主曲率）
    """
    vertices = np.asarray(mesh.vertices)
    n_vertices = len(vertices)
    normals = np.zeros((n_vertices, 3))
    gaussian_curvatures = np.zeros(n_vertices)
    principal_curvatures = np.zeros((n_vertices, 2))
    
    alpha = np.arctan(radius / height)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    
    for i, (t, theta) in enumerate(vertex_params):
        if t < 1.0:
            r = radius * (1 - t)
            if r > 1e-6:
                nx = vertices[i][0] / r
                ny = vertices[i][1] / r
                nz = -sin_alpha / cos_alpha
                norm = np.sqrt(nx**2 + ny**2 + nz**2)
                normals[i] = [nx/norm, ny/norm, nz/norm]
            else:
                normals[i] = [0, 0, 1]
            gaussian_curvatures[i] = 0.0
            if r > 1e-6:
                k1 = sin_alpha / (r * cos_alpha**2)
            else:
                k1 = 0.0
            principal_curvatures[i] = [k1, 0.0]
        else:
            normals[i] = [0, 0, 1]
            gaussian_curvatures[i] = 0.0
            principal_curvatures[i] = [0.0, 0.0]
    
    return normals, gaussian_curvatures, principal_curvatures


def run_test():
    """运行圆锥面测试"""
    print("=" * 50)
    print("圆锥面测试 - 使用谱聚类")
    print("=" * 50)
    
    try:
        # 1. 直接构建圆锥面网格
        print("1. 直接构建圆锥面网格...")
        decimated_mesh, vertex_params = create_cone_mesh(height=2.0, radius=1.0, resolution=150)
        print(f"网格构建完成: {len(decimated_mesh.vertices)} 个顶点, {len(decimated_mesh.triangles)} 个三角形")
        
        # 2. 计算顶点几何性质
        print("2. 计算顶点几何性质...")
        normals, gaussian_curvatures, principal_curvatures = compute_vertex_properties(
            decimated_mesh, vertex_params, height=2.0, radius=1.0
        )
        decimated_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        
        # 3. 构建网格处理器
        print("3. 构建网格处理器...")
        mesh_processor = MeshProcessor(decimated_mesh)
        mesh_processor.gaussian_curvatures = gaussian_curvatures
        mesh_processor.principal_curvatures = principal_curvatures
        
        # 4. 配置刀具
        print("4. 配置刀具...")
        tool = NonSphericalTool(
            profile_type='ellipsoidal',
            params={'semi_axes': [0.5, 0.25]}
        )
        
        # 5. 计算切削宽度族和滚球误差
        print("5. 计算切削宽度族和滚球误差...")
        mesh_processor.calculate_max_cutting_width(tool)
        mesh_processor.calculate_rolled_error()
        
        # 6. 配置分区器
        print("6. 配置分区器...")
        partitioner = AdvancedSurfacePartitioner(
            mesh_processor, 
            tool, 
            resolution=0.5, 
            alpha=0.7,
            global_field='rolled_error'
        )
        
        # 7. 执行分区 - 使用谱聚类
        print("7. 执行分区...")
        labels, edge_midpoints = partitioner.partition_surface(clustering_method='spectral')
        num_partitions = len(np.unique(labels))
        print(f"分区完成: {num_partitions} 个分区")
        
        # 8. 生成刀具路径
        print("8. 生成刀具路径...")
        from core.isoScallopField import IsoScallopFieldGenerator
        tool_orientations = mesh_processor.vertex_normals
        iso_field = IsoScallopFieldGenerator(mesh_processor, tool_orientations, tool, scallop_height=0.4)
        scalar_field = iso_field.generate_scalar_field()
        iso_curves = iso_field.extract_iso_curves(scalar_field, spacing=0.05)
        
        # 创建路径生成器
        path_generator = PathGenerator(mesh_processor, iso_curves, tool_orientations, tool)
        tool_paths = path_generator.generate_final_path()
        
        # 9. 计算路径总长度
        print("9. 计算路径总长度...")
        total_length = 0
        for path_data in tool_paths['paths']:
            points = path_data['points']
            for i in range(len(points) - 1):
                total_length += np.linalg.norm(points[i + 1] - points[i])
        print(f"路径总长度: {total_length:.2f} mm")
        
        # 10. 可视化结果
        print("10. 可视化结果...")
        visualizer = Visualizer()
        if len(edge_midpoints) > 0:
            visualizer.visualize_partitions_with_midpoints(decimated_mesh, labels, edge_midpoints)
        visualizer.visualize_tool_paths(tool_paths['paths'])

        # 11. 验证结果
        print("11. 验证结果...")
        print(f"- 网格顶点数: {len(mesh_processor.vertices)}")
        print(f"- 网格三角形数: {len(mesh_processor.faces)}")
        print(f"- 分区数量: {num_partitions}")
        print(f"- 刀具路径数: {len(iso_curves)}")
        print(f"- 路径总长度: {total_length:.2f} mm")
        
        # 12. 保存结果
        print("12. 保存结果...")
        output_dir = "output/test_cone_spectral"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存网格
        mesh_path = os.path.join(output_dir, "cone_mesh.obj")
        o3d.io.write_triangle_mesh(mesh_path, decimated_mesh)
        print(f"网格保存到: {mesh_path}")
        
        # 保存分区标签
        labels_path = os.path.join(output_dir, "partition_labels.npy")
        np.save(labels_path, labels)
        print(f"分区标签保存到: {labels_path}")
        
        # 保存边缘中点
        edge_midpoints_path = os.path.join(output_dir, "edge_midpoints.npy")
        np.save(edge_midpoints_path, edge_midpoints)
        print(f"边缘中点保存到: {edge_midpoints_path}")
        
        print("\n=== 圆锥面测试完成 ===")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_test()
