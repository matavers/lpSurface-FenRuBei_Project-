#!/usr/bin/env python3
"""
生成验证数据集

该脚本生成用于验证神经网络模型的数据集，包含直纹面和对应的点云。
"""

import numpy as np
import os
import json
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.edgePointToNURBSSurfaceNet import EdgePointToNURBSSurfaceWrapper


def generate_random_ruled_surface(straight=False):
    """
    生成随机直纹面参数
    Args:
        straight: 是否生成较为平直的直纹面
    Returns:
        直纹面参数
    """
    # 生成随机NURBS曲线参数
    def generate_random_nurbs_curve(straight=False):
        M = 16  # 控制点数量
        degree = 3  # 曲线次数
        
        if straight:
            # 生成较为平直的曲线
            # 创建一条基本直线
            start_point = np.random.rand(3) * 2 - 1
            end_point = np.random.rand(3) * 2 - 1
            
            # 生成控制点，大部分在直线上，小部分有轻微扰动
            control_points = np.zeros((M, 3))
            for i in range(M):
                t = i / (M - 1)
                # 基本直线
                base_point = (1 - t) * start_point + t * end_point
                # 添加轻微扰动
                perturbation = np.random.rand(3) * 0.1 - 0.05
                control_points[i] = base_point + perturbation
        else:
            # 生成随机控制点，范围与训练数据一致
            control_points = np.random.rand(M, 3) * 2 - 1  # 范围 [-1, 1]
        
        # 生成随机权重
        weights = np.random.rand(M) + 0.5  # 范围 [0.5, 1.5]
        
        # 生成节点向量
        knot_count = M + degree + 1
        knot_vector = np.zeros(knot_count)
        for i in range(knot_count):
            if i < degree + 1:
                knot_vector[i] = 0.0
            elif i > knot_count - degree - 2:
                knot_vector[i] = 1.0
            else:
                knot_vector[i] = (i - degree) / (knot_count - 2 * degree - 1)
        
        return {
            'type': 'nurbs',
            'control_points': control_points.tolist(),
            'weights': weights.tolist(),
            'knot_vector': knot_vector.tolist(),
            'degree': degree
        }
    
    # 生成两条NURBS曲线
    curve0 = generate_random_nurbs_curve(straight=straight)
    curve1 = generate_random_nurbs_curve(straight=straight)
    
    # 确保两条曲线的节点向量相同
    curve1['knot_vector'] = curve0['knot_vector']
    
    return {
        'type': 'developable',
        'curve0': curve0,
        'curve1': curve1
    }

def _evaluate_nurbs_curve(curve, u):
    """
    评估NURBS曲线上的点
    Args:
        curve: NURBS曲线参数
        u: 参数值
    Returns:
        曲线上的点
    """
    control_points = np.array(curve['control_points'])
    weights = np.array(curve['weights'])
    knot_vector = np.array(curve['knot_vector'])
    degree = curve['degree']
    
    n = len(control_points) - 1
    numerator = np.zeros(3)
    denominator = 0.0
    
    def _compute_basis_function(u, knot_vector, degree, i):
        if degree == 0:
            return 1.0 if knot_vector[i] <= u < knot_vector[i+1] else 0.0
        else:
            denominator1 = knot_vector[i+degree] - knot_vector[i]
            denominator2 = knot_vector[i+degree+1] - knot_vector[i+1]
            
            term1 = 0.0
            if denominator1 > 1e-8:
                term1 = (u - knot_vector[i]) / denominator1 * _compute_basis_function(u, knot_vector, degree-1, i)
            
            term2 = 0.0
            if denominator2 > 1e-8:
                term2 = (knot_vector[i+degree+1] - u) / denominator2 * _compute_basis_function(u, knot_vector, degree-1, i+1)
            
            return term1 + term2
    
    for i in range(n + 1):
        basis = _compute_basis_function(u, knot_vector, degree, i)
        weighted_basis = basis * weights[i]
        numerator += weighted_basis * control_points[i]
        denominator += weighted_basis
    
    if denominator > 1e-8:
        return numerator / denominator
    else:
        return np.zeros(3)

def generate_ruled_surface_points(curve0, curve1, num_u=32, num_v=32):
    """
    生成直纹面上的点
    Args:
        curve0: 第一条曲线参数
        curve1: 第二条曲线参数
        num_u: u方向采样点数
        num_v: v方向采样点数
    Returns:
        直纹面点云 (num_u*num_v, 3)
    """
    surface_points = []
    
    for u in np.linspace(0, 1, num_u):
        # 评估两条NURBS曲线上的点
        curve0_point = _evaluate_nurbs_curve(curve0, u)
        curve1_point = _evaluate_nurbs_curve(curve1, u)
        
        # 沿 v 方向插值生成直纹面
        for v in np.linspace(0, 1, num_v):
            point = (1 - v) * curve0_point + v * curve1_point
            surface_points.append(point)
    
    return np.array(surface_points)

def generate_edge_points(surface, num_points_per_edge=64):
    """
    生成边缘点列
    Args:
        surface: 直纹面参数
        num_points_per_edge: 每条边的点数
    Returns:
        边缘点列
    """
    edges = []
    
    # 生成四条边
    for i in range(4):
        edge_points = []
        for t in np.linspace(0, 1, num_points_per_edge):
            if i % 2 == 0:
                # 第一条曲线
                point = _evaluate_nurbs_curve(surface['curve0'], t)
            else:
                # 第二条曲线
                point = _evaluate_nurbs_curve(surface['curve1'], t)
            edge_points.append(point)
        edges.append(np.array(edge_points))
    
    return edges

def generate_interior_points(surface, num_points=1000):
    """
    生成内部点云
    Args:
        surface: 直纹面参数
        num_points: 点云数量
    Returns:
        内部点云
    """
    points = []
    
    for _ in range(num_points):
        u = np.random.rand()
        v = np.random.rand()
        
        # 计算直纹面上的点
        p1 = _evaluate_nurbs_curve(surface['curve0'], u)
        p2 = _evaluate_nurbs_curve(surface['curve1'], u)
        point = (1 - v) * p1 + v * p2
        
        points.append(point)
    
    return np.array(points)

def generate_validation_data(num_samples=100, output_dir='data/neural/validation', straight_ratio=0.5):
    """
    生成验证数据集
    Args:
        num_samples: 样本数量
        output_dir: 输出目录
        straight_ratio: 生成平直直纹面的比例
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    validation_data = []
    
    for i in range(num_samples):
        print(f"生成样本 {i+1}/{num_samples}")
        
        # 随机决定是否生成平直直纹面
        straight = np.random.rand() < straight_ratio
        
        # 生成直纹面
        surface = generate_random_ruled_surface(straight=straight)
        
        # 生成直纹面点云
        surface_points = generate_ruled_surface_points(surface['curve0'], surface['curve1'])
        
        # 生成边缘点列（与训练数据一致）
        edge_points = generate_edge_points(surface)
        edge_points = [edge.tolist() for edge in edge_points]
        
        # 生成内部点云（与训练数据一致）
        interior_points = generate_interior_points(surface).tolist()
        
        # 保存样本
        sample = {
            'surface': surface,
            'edge_points': edge_points,
            'interior_points': interior_points,
            'surface_points': surface_points.tolist(),
            'is_straight': straight
        }
        
        validation_data.append(sample)
        
        # 保存单个样本
        with open(os.path.join(output_dir, f'sample_{i:03d}.json'), 'w') as f:
            json.dump(sample, f, indent=2)
    
    # 保存所有样本
    with open(os.path.join(output_dir, 'validation_data.json'), 'w') as f:
        json.dump(validation_data, f, indent=2)
    
    print(f"验证数据集生成完成，共 {num_samples} 个样本")
    print(f"其中平直直纹面样本数量: {int(num_samples * straight_ratio)}")


if __name__ == "__main__":
    generate_validation_data()
