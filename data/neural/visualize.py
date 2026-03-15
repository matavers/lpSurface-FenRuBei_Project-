"""
点云可视化脚本

用于可视化模型预测的直纹面和目标直纹面的点云，以及采样点。
"""

import numpy as np
import open3d as o3d
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from train import DevelopableSurfaceDataset


def visualize_sample(sample, sample_id):
    """
    可视化单个样本
    Args:
        sample: 样本数据
        sample_id: 样本ID
    """
    # 获取数据
    interior_points = sample['interior_points']  # 采样点
    edge_points = sample['edge_points']  # 边缘点
    
    # 生成目标直纹面的点云
    # 只处理四边形样本
    curve_A = sample['curve_A_control']
    curve_B = sample['curve_B_control']
    # 生成直纹面点云
    surface_points = []
    u_values = np.linspace(0, 1, 32)
    v_values = np.linspace(0, 1, 32)
    for u in u_values:
        for v in v_values:
            # 简单线性插值
            t = u * (len(curve_A) - 1)
            k = int(np.floor(t))
            if k >= len(curve_A) - 1:
                k = len(curve_A) - 2
            t_local = t - k
            if k == 0:
                p0, p1, p2 = curve_A[0], curve_A[1], curve_A[2]
            elif k == len(curve_A) - 2:
                p0, p1, p2 = curve_A[-3], curve_A[-2], curve_A[-1]
            else:
                p0, p1, p2 = curve_A[k], curve_A[k+1], curve_A[k+2]
            point_A = (1 - t_local)**2 / 2 * p0 + (1 - 2*t_local + t_local**2) * p1 + t_local**2 / 2 * p2
            
            if k == 0:
                p0, p1, p2 = curve_B[0], curve_B[1], curve_B[2]
            elif k == len(curve_B) - 2:
                p0, p1, p2 = curve_B[-3], curve_B[-2], curve_B[-1]
            else:
                p0, p1, p2 = curve_B[k], curve_B[k+1], curve_B[k+2]
            point_B = (1 - t_local)**2 / 2 * p0 + (1 - 2*t_local + t_local**2) * p1 + t_local**2 / 2 * p2
            
            point = (1 - v) * point_A + v * point_B
            surface_points.append(point)
    
    surface_points = np.array(surface_points)
    
    # 加载预测点云
    pred_points = None
    pred_file = f'pred_surface_{sample_id}.npy'
    if os.path.exists(pred_file):
        pred_points = np.load(pred_file)
    
    # 创建Open3D可视化窗口
    # 1. 采样点和边缘点
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name=f'采样点和边缘点 - 样本 {sample_id}', width=800, height=600)
    
    # 添加采样点
    interior_pcd = o3d.geometry.PointCloud()
    interior_pcd.points = o3d.utility.Vector3dVector(interior_points)
    interior_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
    vis1.add_geometry(interior_pcd)
    
    # 添加边缘点
    for i, edge in enumerate(edge_points):
        edge_pcd = o3d.geometry.PointCloud()
        edge_pcd.points = o3d.utility.Vector3dVector(edge)
        edge_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
        vis1.add_geometry(edge_pcd)
    
    # 设置视图
    vis1.get_render_option().point_size = 3
    vis1.run()
    vis1.destroy_window()
    
    # 2. 原始直纹面
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name=f'原始直纹面 - 样本 {sample_id}', width=800, height=600)
    
    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(surface_points)
    surface_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
    vis2.add_geometry(surface_pcd)
    
    # 设置视图
    vis2.get_render_option().point_size = 2
    vis2.run()
    vis2.destroy_window()
    
    # 3. 神经网络输出
    if pred_points is not None:
        vis3 = o3d.visualization.Visualizer()
        vis3.create_window(window_name=f'神经网络输出 - 样本 {sample_id}', width=800, height=600)
        
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(pred_points)
        pred_pcd.paint_uniform_color([0.5, 0.0, 0.5])  # 紫色
        vis3.add_geometry(pred_pcd)
        
        # 设置视图
        vis3.get_render_option().point_size = 2
        vis3.run()
        vis3.destroy_window()


def visualize_training_data(data_path, num_samples=5):
    """
    可视化训练数据
    Args:
        data_path: 训练数据路径
        num_samples: 要可视化的样本数量
    """
    # 加载数据集
    dataset = DevelopableSurfaceDataset(data_path)
    
    # 可视化前num_samples个样本
    for i in range(min(num_samples, len(dataset))):
        sample = dataset.data[i]
        visualize_sample(sample, i+1)


def visualize_test_results(num_samples=5):
    """
    可视化测试结果
    Args:
        num_samples: 要可视化的样本数量
    """
    # 加载测试数据集
    test_dataset = DevelopableSurfaceDataset('dataset/test/test_dataset.npy')
    
    # 可视化前num_samples个样本
    for i in range(min(num_samples, len(test_dataset))):
        sample = test_dataset.data[i]
        visualize_sample(sample, i+1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化直纹面点云')
    parser.add_argument('--type', type=str, default='test', choices=['train', 'test'],
                        help='可视化类型：train（训练数据）或 test（测试结果）')
    parser.add_argument('--data_path', type=str, default='train_dataset.npy',
                        help='数据文件路径')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='要可视化的样本数量')
    
    args = parser.parse_args()
    
    if args.type == 'train':
        visualize_training_data(args.data_path, args.num_samples)
    else:
        visualize_test_results(args.num_samples)

