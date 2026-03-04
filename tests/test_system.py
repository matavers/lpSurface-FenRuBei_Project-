"""
运行五轴加工路径规划系统主流程
"""

import open3d as o3d
import numpy as np
import os
import sys

# 添加根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import FiveAxisMachiningSystem
from utils.validation import MachiningValidator
from core.surfaceGenerator import SurfaceGenerator


def run_main_pipeline():
    """运行主流程"""
    print("=" * 70)
    print("运行五轴加工路径规划系统主流程")
    print("=" * 70)
    
    try:
        # 初始化系统 - 增加分区大小
        print("\n初始化五轴加工系统...")
        
        # 修改配置以增加分区大小
        config = {
            'algorithm': {
                'partition_resolution': 0.05,  # 降低分辨率，减少分区数量
                'tar_sampling_resolution': 30
            }
        }
        
        system = FiveAxisMachiningSystem()
        
        # 直接修改系统配置中的分区参数
        system.config['algorithm']['partition_resolution'] = 0.05  # 降低分辨率，减少分区数量
        print("已设置分区参数以增加分区大小")
        
        # 运行完整流程
        print("\n运行完整加工路径规划流程...")
        
        # 使用曲面函数生成网格，测试新的算法选择机制
        surface_func = "sphere"
        surface_params = {"resolution": 50}
        mesh_algorithm = "delaunay_cocone"  # 使用默认的Delaunay + Cocone算法
        
        print(f"使用曲面函数: {surface_func}")
        print(f"使用网格算法: {mesh_algorithm}")
        
        # 先加载网格（使用曲面函数生成）
        system.load_mesh_from_file("sphere", mesh_algorithm=mesh_algorithm, surface_func=surface_func, surface_params=surface_params)
        system.setup_tool()
        
        # 运行分区
        print("\n运行表面分区...")
        partition_labels, edge_midpoints = system.run_partitioning()
        
        # 暂时不运行边缘拟合功能
        print("\n暂时不运行边缘拟合功能...")
        surface_generator = SurfaceGenerator()
        
        # 获取原始网格的顶点和面
        vertices = np.asarray(system.mesh.vertices)
        faces = np.asarray(system.mesh.triangles)
        
        # 提取每个分区的边缘
        unique_labels = np.unique(partition_labels)
        
        # 收集所有边缘点
        all_edge_points = []
        for label in unique_labels:
            # 找到属于当前分区的顶点
            partition_vertices = np.where(partition_labels == label)[0]
            
            # 提取边缘点（与其他分区相邻的顶点）
            edge_points = []
            for v in partition_vertices:
                # 检查邻居是否属于不同分区
                neighbors = system.mesh_processor.adjacency[v]
                has_other_partition = False
                for neighbor in neighbors:
                    if partition_labels[neighbor] != label:
                        has_other_partition = True
                        break
                if has_other_partition:
                    edge_points.append(vertices[v])
            
            print(f"分区 {label}: 收集边缘点，{len(edge_points)} 个边缘点")
            all_edge_points.extend(edge_points)
        
        # 创建基于时间戳的output子目录
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = os.path.join("output", timestamp)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        print(f"输出目录: {output_subdir}")
        
        # 保存分区结果
        print("\n4. 保存分区结果...")
        # 保存分区标签
        partition_file = os.path.join(output_subdir, "partition_labels.npy")
        np.save(partition_file, partition_labels)
        print(f"分区标签保存到: {partition_file}")
        print(f"分区数量: {len(unique_labels)}")
        
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
        np.save(edge_points_file, all_edge_points)
        print(f"边缘点数据保存到: {edge_points_file}")
        print(f"边缘点数量: {len(all_edge_points)}")
        
        # 保存分区统计信息
        print("\n7. 保存分区统计信息...")
        metrics = {
            'num_partitions': len(unique_labels),
            'num_vertices': len(vertices),
            'num_faces': len(faces),
            'num_edge_points': len(all_edge_points)
        }
        metrics_file = os.path.join(output_subdir, "metrics.json")
        import json
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"分区统计信息保存到: {metrics_file}")
        
        # 检查分区数量是否与三角面数量相同
        print("\n9. 检查分区结果...")
        num_partitions = len(unique_labels)
        num_faces = len(faces)
        print(f"三角面数量: {num_faces}")
        print(f"分区数量: {num_partitions}")
        
        if num_partitions == num_faces:
            print("警告: 分区数量与三角面数量相同，每个三角面可能被作为一个分区")
            print("这可能是因为Leiden聚类参数设置不当，或者网格过于简单")
            print("建议调整AdvancedSurfacePartitioner中的resolution参数或聚类参数")
        else:
            print("分区数量与三角面数量不同，分区结果看起来合理")
        
        # 跳过边缘拟合可视化
        print("\n跳过边缘拟合可视化...")
        
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
        
        # 使用曲面函数运行完整流程
        print("\n使用曲面函数运行完整流程...")
        success = system.run_full_pipeline("sphere", skip_visualization=True, mesh_algorithm=mesh_algorithm, surface_func=surface_func, surface_params=surface_params)  # 禁用可视化，避免重复显示
        
        if success:
            # 运行验证
            print("\n运行加工验证...")
            validator = MachiningValidator(system.mesh_processor, system.tool)
            tool_paths = system.results['tool_paths']
            
            if tool_paths:
                report = validator.generate_report(tool_paths)
                print("\n验证报告:")
                print(report['summary'])
                
                # 跳过可视化验证结果，避免卡住
                validator.visualize_validation(tool_paths)
            
            print("\n主流程运行成功!")
        else:
            print("\n主流程运行失败!")
            
    except Exception as e:
        print(f"运行过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理临时文件
        temp_files = ["temp_sphere.obj", "temp_torus.obj", "temp_saddle.obj"]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"清理临时文件: {temp_file}")


if __name__ == "__main__":
    run_main_pipeline()
