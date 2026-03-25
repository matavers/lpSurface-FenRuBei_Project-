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
from utils.visualization import Visualizer


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
    n_vertices = len(vertices)
    normals = np.zeros((n_vertices, 3))
    gaussian_curvatures = np.zeros(n_vertices)
    principal_curvatures = np.zeros((n_vertices, 2))
    
    # 计算圆锥的半顶角 α
    alpha = np.arctan(radius / height)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    
    for i, (t, theta) in enumerate(vertex_params):
        if t < 1.0:  # 非顶点
            # 解析法向量：圆锥面上任意点的法向量与圆锥轴线夹角为 α
            # 圆锥轴线沿 z 轴，所以法向量可以表示为 (x/r, y/r, -sin(alpha)/cos(alpha)) 归一化
            r = radius * (1 - t)
            if r > 1e-6:  # 避免除以零
                nx = vertices[i][0] / r
                ny = vertices[i][1] / r
                nz = -sin_alpha / cos_alpha
                norm = np.sqrt(nx**2 + ny**2 + nz**2)
                normals[i] = [nx/norm, ny/norm, nz/norm]
            else:  # 接近顶点，使用近似法向量
                normals[i] = [0, 0, 1]
            
            # 圆锥面的高斯曲率为零（可展曲面）
            gaussian_curvatures[i] = 0.0
            
            # 主曲率：一个为主曲率 k1 = (sin(alpha))/(r * cos(alpha)^2)，另一个 k2 = 0
            if r > 1e-6:
                k1 = sin_alpha / (r * cos_alpha**2)
            else:
                k1 = 0.0
            principal_curvatures[i] = [k1, 0.0]
        else:  # 顶点
            # 顶点处法向量沿 z 轴
            normals[i] = [0, 0, 1]
            # 顶点处曲率为无穷大，这里设为 0 作为近似
            gaussian_curvatures[i] = 0.0
            principal_curvatures[i] = [0.0, 0.0]
    
    return normals, gaussian_curvatures, principal_curvatures


def run_test():
    """运行圆锥面测试"""
    print("=" * 50)
    print("圆锥面测试 - 使用谱聚类")
    print("=" * 50)
    
    try:
        # 1. 直接生成圆锥面网格
        print("1. 直接生成圆锥面网格...")
        
        # 使用参数方程直接生成圆锥面网格
        decimated_mesh, vertex_params = create_cone_mesh(height=2.0, radius=1.0, resolution=150)
        
        print(f"网格生成完成: {len(decimated_mesh.vertices)} 个顶点, {len(decimated_mesh.triangles)} 个三角形")
        
        # 2. 计算顶点几何特性
        print("2. 计算顶点几何特性...")
        
        # 使用解析方法计算几何特性
        normals, gaussian_curvatures, principal_curvatures = compute_vertex_properties(
            decimated_mesh, vertex_params, height=2.0, radius=1.0
        )
        
        # 更新网格的法向量
        decimated_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        
        # 3. 创建网格处理器
        print("3. 创建网格处理器...")
        mesh_processor = MeshProcessor(decimated_mesh)
        
        # 设置曲率属性
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
            alpha=0.7,  # 全局引导强度
            global_field='rolled_error'  # 使用直纹面逼近误差作为全局场
        )
        
        # 7. 执行分区 - 使用谱聚类
        print("7. 执行分区...")
        labels, edge_midpoints = partitioner.partition_surface(clustering_method='spectral')
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
                    v_idx = partition_vertices[i]
                    curve.append(mesh_processor.vertices[v_idx])
                if len(curve) > 1:
                    iso_curves.append(np.array(curve))
        
        print(f"刀具路径生成完成: {len(iso_curves)} 条路径")
        
        # 9. 计算路径总长度
        print("9. 计算路径总长度...")
        total_length = 0
        for curve in iso_curves:
            if len(curve) > 1:
                for i in range(len(curve) - 1):
                    total_length += np.linalg.norm(curve[i+1] - curve[i])
        print(f"路径总长度: {total_length:.2f} mm")
        
        # 10. 可视化结果
        print("10. 可视化结果...")
        visualizer = Visualizer()
        
        # 可视化分区和边缘中点
        if len(edge_midpoints) > 0:
            visualizer.visualize_partitions_with_midpoints(decimated_mesh, labels, edge_midpoints)
        
        # 可视化刀具路径
        if iso_curves:
            # 转换为可视化所需的格式
            tool_paths = []
            for i, curve in enumerate(iso_curves):
                tool_path = {
                    'points': curve,
                    'type': 'machining'
                }
                tool_paths.append(tool_path)
            visualizer.visualize_tool_paths(tool_paths)
        
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
