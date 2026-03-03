"""
测试分区边缘拟合脚本
直接使用data/models目录下的modified_sphere.obj文件，运行Leiden聚类分区，拟合分区边缘，可视化分区结果，不再执行刀具路径计算
"""

import open3d as o3d
import numpy as np
import os
import sys

# 添加根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import FiveAxisMachiningSystem
from core.surfaceGenerator import SurfaceGenerator

def run_partition_edge_fitting():
    """运行分区边缘拟合测试"""
    print("=" * 70)
    print("运行分区边缘拟合测试")
    print("=" * 70)
    
    try:
        # 加载已有的网格文件
        mesh_path = "data/models/modified_sphere.obj"
        if not os.path.exists(mesh_path):
            print(f"错误: 网格文件 {mesh_path} 不存在")
            return
        
        print(f"加载网格文件: {mesh_path}")
        
        # 初始化系统
        print("\n初始化五轴加工系统...")
        system = FiveAxisMachiningSystem()
        
        # 加载网格
        print("\n加载网格...")
        system.load_mesh_from_file(mesh_path)
        
        # 设置刀具
        print("\n设置刀具...")
        system.setup_tool()
        
        # 运行分区
        print("\n运行表面分区...")
        # 创建基于时间戳的output子目录
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = os.path.join("output", timestamp)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        print(f"输出目录: {output_subdir}")
        
        # 尝试加载已有的分区结果（优先从output子目录加载）
        partition_file = os.path.join(output_subdir, "partition_labels.npy")
        edge_midpoints_file = os.path.join(output_subdir, "edge_midpoints.npy")
        
        if os.path.exists(partition_file):
            print("加载已有的分区结果...")
            partition_labels = np.load(partition_file)
            # 将分区结果存储到系统中，用于可视化
            system.results['partition_labels'] = partition_labels
            # 尝试加载edge_midpoints
            if os.path.exists(edge_midpoints_file):
                edge_midpoints = np.load(edge_midpoints_file)
                system.results['edge_midpoints'] = edge_midpoints
                print(f"加载成功，分区数量: {len(np.unique(partition_labels))}, 边缘中点数量: {len(edge_midpoints)}")
            else:
                print(f"加载成功，分区数量: {len(np.unique(partition_labels))}")
        else:
            # 尝试从旧位置加载
            old_partition_file = "data/models/partition_labels.npy"
            old_edge_midpoints_file = "data/models/edge_midpoints.npy"
            
            if os.path.exists(old_partition_file):
                print("从旧位置加载分区结果...")
                partition_labels = np.load(old_partition_file)
                # 将分区结果存储到系统中，用于可视化
                system.results['partition_labels'] = partition_labels
                # 尝试加载edge_midpoints
                if os.path.exists(old_edge_midpoints_file):
                    edge_midpoints = np.load(old_edge_midpoints_file)
                    system.results['edge_midpoints'] = edge_midpoints
                    print(f"加载成功，分区数量: {len(np.unique(partition_labels))}, 边缘中点数量: {len(edge_midpoints)}")
                else:
                    print(f"加载成功，分区数量: {len(np.unique(partition_labels))}")
                # 复制到新位置
                print("将分区结果复制到新的output目录...")
                np.save(partition_file, partition_labels)
                if os.path.exists(old_edge_midpoints_file):
                    np.save(edge_midpoints_file, edge_midpoints)
            else:
                    print("计算新的分区结果...")
                    partition_labels, edge_midpoints = system.run_partitioning()
                    # 将分区结果存储到系统中，用于可视化
                    system.results['partition_labels'] = partition_labels
                    system.results['edge_midpoints'] = edge_midpoints
                    # 保存分区结果到output子目录
                    print("保存分区结果...")
                    np.save(partition_file, partition_labels)
                    np.save(edge_midpoints_file, edge_midpoints)
                    print(f"保存成功，分区数量: {len(np.unique(partition_labels))}, 边缘中点数量: {len(edge_midpoints)}")
        
        # 暂时不运行边缘拟合功能
        print("\n暂时不运行边缘拟合功能...")
        surface_generator = SurfaceGenerator()
        
        # 获取原始网格的顶点和面
        vertices = np.asarray(system.mesh.vertices)
        faces = np.asarray(system.mesh.triangles)
        
        # 提取每个分区的边缘，确保每条边缘只被处理一次
        unique_labels = np.unique(partition_labels)
        all_edge_points = []
        all_new_vertices = []
        all_new_vertices_labels = []  # 存储每个新顶点对应的分区标签
        processed_edges = set()  # 用于跟踪已处理的边缘
        
        # 首先收集所有唯一的边缘，确保每条边缘只被处理一次
        unique_edges = {}  # 存储每条唯一边缘的点和相关分区
        
        for v in range(len(vertices)):
            # 检查邻居是否属于不同分区
            neighbors = system.mesh_processor.adjacency[v]
            for neighbor in neighbors:
                if partition_labels[neighbor] != partition_labels[v]:
                    # 创建边缘的唯一标识（按顺序排列顶点索引）
                    edge_key = tuple(sorted([v, neighbor]))
                    if edge_key not in processed_edges:
                        processed_edges.add(edge_key)
                        # 存储边缘点和相关分区
                        label1 = partition_labels[v]
                        label2 = partition_labels[neighbor]
                        # 选择一个分区来处理这条边缘（选择标签较小的）
                        processing_label = min(label1, label2)
                        # 存储边缘点
                        edge_points = [vertices[v], vertices[neighbor]]
                        if processing_label not in unique_edges:
                            unique_edges[processing_label] = []
                        unique_edges[processing_label].extend(edge_points)
        
        # 只收集边缘点，不进行拟合
        for label in unique_labels:
            if label in unique_edges:
                edge_points = unique_edges[label]
                
                # 去重边缘点
                unique_edge_points = []
                seen = set()
                for point in edge_points:
                    point_tuple = tuple(point)
                    if point_tuple not in seen:
                        seen.add(point_tuple)
                        unique_edge_points.append(point)
                
                print(f"分区 {label}: 收集边缘点，{len(unique_edge_points)} 个边缘点")
                all_edge_points.extend(unique_edge_points)
            else:
                print(f"分区 {label}: 无边缘点")
        
        print("\n跳过边缘拟合，直接保存结果...")
        
        # 保存分区结果
        print("\n4. 保存分区结果...")
        # 保存分区标签
        partition_file = os.path.join(output_subdir, "partition_labels.npy")
        np.save(partition_file, partition_labels)
        print(f"分区标签保存到: {partition_file}")
        print(f"分区数量: {len(np.unique(partition_labels))}")
        
        # 保存网格数据
        print("\n5. 保存网格数据...")
        # 保存顶点
        vertices_file = os.path.join(output_subdir, "vertices.npy")
        np.save(vertices_file, vertices)
        print(f"顶点数据保存到: {vertices_file}")
        
        # 保存面
        faces_file = os.path.join(output_subdir, "triangles.npy")
        np.save(faces_file, faces)
        print(f"面数据保存到: {faces_file}")
        
        # 对边缘点进行全局去重
        unique_all_edge_points = []
        seen = set()
        for point in all_edge_points:
            point_tuple = tuple(point)
            if point_tuple not in seen:
                seen.add(point_tuple)
                unique_all_edge_points.append(point)
        
        # 保存边缘点
        print("\n6. 提取和保存边缘点...")
        edge_points_file = os.path.join(output_subdir, "edge_points.npy")
        np.save(edge_points_file, unique_all_edge_points)
        print(f"边缘点数据保存到: {edge_points_file}")
        print(f"去重后边缘点数量: {len(unique_all_edge_points)}")
        
        # 保存投影点
        if all_new_vertices:
            projected_points_file = os.path.join(output_subdir, "projected_points.npy")
            np.save(projected_points_file, all_new_vertices)
            print(f"投影点数据保存到: {projected_points_file}")
        
        # 保存新顶点标签
        if all_new_vertices_labels:
            new_vertices_labels_file = os.path.join(output_subdir, "new_vertices_labels.npy")
            np.save(new_vertices_labels_file, all_new_vertices_labels)
            print(f"新顶点标签保存到: {new_vertices_labels_file}")
        
        # 保存分区统计信息
        print("\n7. 保存分区统计信息...")
        metrics = {
            'num_partitions': len(np.unique(partition_labels)),
            'num_vertices': len(vertices),
            'num_faces': len(faces),
            'num_edge_points': len(all_edge_points),
            'num_projected_points': len(all_new_vertices)
        }
        metrics_file = os.path.join(output_subdir, "metrics.json")
        import json
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"分区统计信息保存到: {metrics_file}")
        
        # 检查分区数量是否与三角面数量相同
        print("\n9. 检查分区结果...")
        num_partitions = len(np.unique(partition_labels))
        num_faces = len(faces)
        print(f"三角面数量: {num_faces}")
        print(f"分区数量: {num_partitions}")
        
        if num_partitions == num_faces:
            print("警告: 分区数量与三角面数量相同，每个三角面可能被作为一个分区")
            print("这可能是因为Leiden聚类参数设置不当，或者网格过于简单")
            print("建议调整AdvancedSurfacePartitioner中的resolution参数或聚类参数")
        else:
            print("分区数量与三角面数量不同，分区结果看起来合理")
        
        # 可视化分区结果，在一个窗口中同时显示颜色块分区和中点
        print("\n可视化分区结果...")
        # 使用新的可视化方法，在一个窗口中同时显示颜色块分区和中点
        if 'edge_midpoints' in system.results:
            system.visualizer.visualize_partitions_with_midpoints(
                system.mesh, 
                partition_labels, 
                system.results['edge_midpoints']
            )
        else:
            # 如果没有边缘中点，使用原来的可视化方法
            system.visualizer.visualize_partitions(system.mesh, partition_labels)
        
        # 跳过点云可视化
        # print("\n使用备用方案：点云可视化...")
        # import open3d as o3d
        # 
        # # 创建点云
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(vertices)
        # 
        # # 为每个点设置颜色：非边缘点为灰白色，边缘点为红色
        # colors = []
        # for i, vertex in enumerate(vertices):
        #     # 检查是否是边缘点
        #     is_edge = False
        #     neighbors = system.mesh_processor.adjacency[i]
        #     for neighbor in neighbors:
        #         if partition_labels[neighbor] != partition_labels[i]:
        #             is_edge = True
        #             break
        #     
        #     if is_edge:
        #         colors.append([1, 0, 0])  # 红色边缘点
        #     else:
        #         colors.append([0.8, 0.8, 0.8])  # 灰白色非边缘点
        # 
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # 
        # # 可视化点云
        # print(f"点云可视化：{len(vertices)} 个点，其中 {sum(1 for c in colors if c == [1, 0, 0])} 个边缘点")
        # o3d.visualization.draw_geometries(
        #     [pcd], 
        #     window_name="分区点云可视化（边缘点为红色）"
        # )
        print("\n跳过所有可视化步骤，测试完成...")
        
        print("\n分区边缘拟合测试完成!")
        print(f"所有结果已保存到: {output_subdir}")
        print("测试已终止，未执行刀具路径计算")
        
    except Exception as e:
        print(f"运行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_partition_edge_fitting()
