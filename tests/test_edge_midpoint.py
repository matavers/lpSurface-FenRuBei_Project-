"""
测试边缘中点提取脚本
取不同分区内对应的边缘点的中点，作为分区边缘
"""

import open3d as o3d
import numpy as np
import os
import sys

# 添加根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import FiveAxisMachiningSystem
from core.surfaceGenerator import SurfaceGenerator

def run_edge_midpoint_extraction():
    """运行边缘中点提取测试"""
    print("=" * 70)
    print("运行边缘中点提取测试")
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
        
        # 提取分区边缘并计算中点
        print("\n提取分区边缘并计算中点...")
        surface_generator = SurfaceGenerator()
        
        # 获取原始网格的顶点和面
        vertices = np.asarray(system.mesh.vertices)
        faces = np.asarray(system.mesh.triangles)
        
        # 提取每个分区的边缘点
        unique_labels = np.unique(partition_labels)
        all_edge_points = []
        all_midpoints = []
        all_new_vertices = []
        all_new_vertices_labels = []  # 存储每个新顶点对应的分区标签
        
        # 首先收集所有边缘点对
        edge_pairs = {}
        
        for v in range(len(vertices)):
            # 检查邻居是否属于不同分区
            neighbors = system.mesh_processor.adjacency[v]
            for neighbor in neighbors:
                if partition_labels[neighbor] != partition_labels[v]:
                    # 创建边缘的唯一标识（按顺序排列顶点索引）
                    edge_key = tuple(sorted([v, neighbor]))
                    if edge_key not in edge_pairs:
                        label1 = partition_labels[v]
                        label2 = partition_labels[neighbor]
                        edge_pairs[edge_key] = {
                            'point1': vertices[v],
                            'point2': vertices[neighbor],
                            'label1': label1,
                            'label2': label2
                        }
        
        # 计算每个边缘的中点
        print(f"找到 {len(edge_pairs)} 条唯一边缘")
        
        for edge_key, edge_data in edge_pairs.items():
            # 计算中点
            midpoint = (edge_data['point1'] + edge_data['point2']) / 2
            all_midpoints.append(midpoint)
            all_edge_points.append(edge_data['point1'])
            all_edge_points.append(edge_data['point2'])
        
        # 去重边缘点
        unique_all_edge_points = []
        seen = set()
        for point in all_edge_points:
            point_tuple = tuple(point)
            if point_tuple not in seen:
                seen.add(point_tuple)
                unique_all_edge_points.append(point)
        
        print(f"原始边缘点数量: {len(all_edge_points)}")
        print(f"去重后边缘点数量: {len(unique_all_edge_points)}")
        print(f"中点数量: {len(all_midpoints)}")
        
        # 将中点作为新的边缘点
        all_new_vertices = all_midpoints
        # 为每个中点分配标签（使用较小的分区标签）
        for edge_key, edge_data in edge_pairs.items():
            label = min(edge_data['label1'], edge_data['label2'])
            all_new_vertices_labels.append(label)
        
        # 跳过边缘中点可视化，保留分区结果可视化
        print("\n跳过边缘中点可视化...")
        
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
        
        # 保存边缘点
        print("\n6. 提取和保存边缘点...")
        edge_points_file = os.path.join(output_subdir, "edge_points.npy")
        np.save(edge_points_file, unique_all_edge_points)
        print(f"边缘点数据保存到: {edge_points_file}")
        print(f"去重后边缘点数量: {len(unique_all_edge_points)}")
        
        # 保存中点作为投影点
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
            'num_projected_points': len(all_new_vertices),
            'num_midpoints': len(all_midpoints)
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
            edge_midpoints = system.results['edge_midpoints']
        else:
            edge_midpoints = np.array([])
        
        system.visualizer.visualize_partitions_with_midpoints(
            system.mesh, 
            partition_labels, 
            edge_midpoints
        )
        
        print("\n分区边缘中点提取测试完成!")
        print(f"所有结果已保存到: {output_subdir}")
        print("测试已终止，未执行刀具路径计算")
        
    except Exception as e:
        print(f"运行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_edge_midpoint_extraction()
