#!/usr/bin/env python3
"""
验证神经网络模型

该脚本加载验证数据集，运行神经网络模型，然后使用Open3D可视化结果。
"""

import numpy as np
import os
import json
import sys
import open3d as o3d

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.edgePointToNURBSSurfaceNet import EdgePointToNURBSSurfaceWrapper


def load_validation_data(validation_dir='data/neural/validation'):
    """
    加载验证数据集
    Args:
        validation_dir: 验证数据目录
    Returns:
        验证数据列表
    """
    validation_data = []
    
    # 加载所有样本文件
    for filename in os.listdir(validation_dir):
        if filename.startswith('sample_') and filename.endswith('.json'):
            file_path = os.path.join(validation_dir, filename)
            with open(file_path, 'r') as f:
                sample = json.load(f)
                validation_data.append(sample)
    
    print(f"加载了 {len(validation_data)} 个验证样本")
    return validation_data


def generate_surface_points(surface, num_points=None):
    """
    生成直纹面上的点
    Args:
        surface: 直纹面参数
        num_points: 生成的点数量，如果为None则使用默认值
    Returns:
        直纹面点云
    """
    # 如果没有指定点数量，使用默认值
    if num_points is None:
        num_u, num_v = 32, 32
    else:
        # 计算合适的u和v方向采样点数
        num_u = int(np.sqrt(num_points))
        num_v = (num_points + num_u - 1) // num_u
    
    surface_points = []
    
    # 创建临时包装器来评估NURBS曲线
    wrapper = EdgePointToNURBSSurfaceWrapper()
    
    for u in np.linspace(0, 1, num_u):
        # 评估两条NURBS曲线上的点
        curve0_point = wrapper._evaluate_nurbs_curve(surface['curve0'], u)
        curve1_point = wrapper._evaluate_nurbs_curve(surface['curve1'], u)
        
        # 沿 v 方向插值生成直纹面
        for v in np.linspace(0, 1, num_v):
            point = (1 - v) * curve0_point + v * curve1_point
            surface_points.append(point)
    
    return np.array(surface_points)


def generate_curve_points(curve, num_points=100):
    """
    生成曲线上的点
    Args:
        curve: NURBS曲线参数
        num_points: 采样点数
    Returns:
        曲线上的点云
    """
    wrapper = EdgePointToNURBSSurfaceWrapper()
    points = []
    for u in np.linspace(0, 1, num_points):
        point = wrapper._evaluate_nurbs_curve(curve, u)
        points.append(point)
    return np.array(points)

def visualize_result(original_points, predicted_surface):
    """
    可视化结果
    Args:
        original_points: 原始点云
        predicted_surface: 预测的直纹面参数
    """
    # 生成预测直纹面上的点，使用与原始点云相同数量的点
    predicted_points = generate_surface_points(predicted_surface, num_points=len(original_points))
    
    # 生成准线上的点
    curve0_points = generate_curve_points(predicted_surface['curve0'])
    curve1_points = generate_curve_points(predicted_surface['curve1'])
    
    # 创建原点云（绿色）
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(original_points)
    original_pcd.paint_uniform_color([0, 1, 0])  # 绿色
    
    # 创建预测直纹面点云（红色）
    predicted_pcd = o3d.geometry.PointCloud()
    predicted_pcd.points = o3d.utility.Vector3dVector(predicted_points)
    predicted_pcd.paint_uniform_color([1, 0, 0])  # 红色
    
    # 创建准线（深紫色和青色，更容易看清）
    curve0_pcd = o3d.geometry.PointCloud()
    curve0_pcd.points = o3d.utility.Vector3dVector(curve0_points)
    curve0_pcd.paint_uniform_color([0.5, 0, 0.5])  # 深紫色
    
    curve1_pcd = o3d.geometry.PointCloud()
    curve1_pcd.points = o3d.utility.Vector3dVector(curve1_points)
    curve1_pcd.paint_uniform_color([0, 1, 1])  # 青色
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="验证结果可视化", width=1024, height=768)
    
    # 添加点云
    vis.add_geometry(original_pcd)
    vis.add_geometry(predicted_pcd)
    vis.add_geometry(curve0_pcd)
    vis.add_geometry(curve1_pcd)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1, 1, 1])  # 白色背景
    render_option.point_size = 2.0
    
    # 运行可视化
    print("正在显示验证结果...")
    print("绿色：原始点云")
    print("红色：拟合的直纹面")
    print("深紫色：第一条准线")
    print("青色：第二条准线")
    print("按ESC键关闭窗口")
    vis.run()
    vis.destroy_window()


def validate_model(model_path='checkpoints/best_nurbs_model.pth', validation_dir='data/neural/validation'):
    """
    验证神经网络模型
    Args:
        model_path: 模型权重文件路径
        validation_dir: 验证数据目录
    """
    # 加载验证数据
    validation_data = load_validation_data(validation_dir)
    
    # 初始化神经网络模型
    wrapper = EdgePointToNURBSSurfaceWrapper(model_path=model_path)
    
    # 验证每个样本
    for i, sample in enumerate(validation_data):
        print(f"验证样本 {i+1}/{len(validation_data)}")
        
        # 提取输入数据
        edge_points = sample['edge_points']
        interior_points = sample['interior_points']
        original_points = np.array(sample['surface_points'])
        
        # 转换内部点云为numpy数组
        interior_points = np.array(interior_points)
        
        # 预测直纹面
        try:
            predicted_surface = wrapper.predict(edge_points, interior_points)
            
            # 可视化结果
            visualize_result(original_points, predicted_surface)
            
        except Exception as e:
            print(f"验证样本 {i+1} 失败: {e}")
            continue


if __name__ == "__main__":
    validate_model()
