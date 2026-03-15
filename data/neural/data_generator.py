"""
NURBS直纹面训练数据生成器

本脚本用于生成训练EdgePointToNURBSSurfaceNet所需的数据集。
"""

import numpy as np
import os
import json
from datetime import datetime


class NURBSSurfaceDataGenerator:
    """
    NURBS直纹面数据生成器
    """
    
    def __init__(self, output_dir='dataset'):
        """
        初始化数据生成器
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.train_dir = os.path.join(output_dir, 'train')
        self.test_dir = os.path.join(output_dir, 'test')
        
        # 创建目录
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)
    
    def generate_nurbs_curve(self, num_control_points=16, degree=3, straight=False):
        """
        生成随机NURBS曲线
        Args:
            num_control_points: 控制点数量
            degree: 曲线次数
            straight: 是否生成较为平直的曲线
        Returns:
            控制点、权重、节点向量
        """
        if straight:
            # 生成较为平直的曲线
            # 创建一条基本直线
            start_point = np.random.rand(3) * 2 - 1
            end_point = np.random.rand(3) * 2 - 1
            
            # 生成控制点，大部分在直线上，小部分有轻微扰动
            control_points = np.zeros((num_control_points, 3))
            for i in range(num_control_points):
                t = i / (num_control_points - 1)
                # 基本直线
                base_point = (1 - t) * start_point + t * end_point
                # 添加轻微扰动
                perturbation = np.random.rand(3) * 0.1 - 0.05
                control_points[i] = base_point + perturbation
        else:
            # 生成随机控制点
            control_points = np.random.rand(num_control_points, 3) * 2 - 1  # 范围 [-1, 1]
        
        # 生成随机权重（确保为正）
        weights = np.random.rand(num_control_points) + 0.5  # 范围 [0.5, 1.5]
        
        # 生成节点向量
        knot_vector = self._generate_knot_vector(num_control_points, degree)
        
        return control_points, weights, knot_vector
    
    def _generate_knot_vector(self, num_control_points, degree):
        """
        生成节点向量
        Args:
            num_control_points: 控制点数量
            degree: 曲线次数
        Returns:
            节点向量
        """
        knot_count = num_control_points + degree + 1
        knot_vector = np.zeros(knot_count)
        
        # 均匀节点向量
        for i in range(knot_count):
            if i < degree + 1:
                knot_vector[i] = 0.0
            elif i > knot_count - degree - 2:
                knot_vector[i] = 1.0
            else:
                knot_vector[i] = (i - degree) / (knot_count - 2 * degree - 1)
        
        return knot_vector
    
    def generate_developable_surface(self, straight=False):
        """
        生成直纹面
        Args:
            straight: 是否生成较为平直的直纹面
        Returns:
            两条NURBS曲线的参数
        """
        # 生成两条NURBS曲线
        curve0_ctrl, curve0_weights, curve0_knots = self.generate_nurbs_curve(straight=straight)
        curve1_ctrl, curve1_weights, curve1_knots = self.generate_nurbs_curve(straight=straight)
        
        # 确保两条曲线的节点向量相同
        curve1_knots = curve0_knots.copy()
        
        return {
            'curve0': {
                'control_points': curve0_ctrl,
                'weights': curve0_weights,
                'knot_vector': curve0_knots,
                'degree': 3
            },
            'curve1': {
                'control_points': curve1_ctrl,
                'weights': curve1_weights,
                'knot_vector': curve1_knots,
                'degree': 3
            }
        }
    
    def generate_edge_points(self, surface, num_points_per_edge=64):
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
                    point = self._evaluate_nurbs_curve(surface['curve0'], t)
                else:
                    # 第二条曲线
                    point = self._evaluate_nurbs_curve(surface['curve1'], t)
                edge_points.append(point)
            edges.append(np.array(edge_points))
        
        return edges
    
    def generate_interior_points(self, surface, num_points=1000):
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
            p1 = self._evaluate_nurbs_curve(surface['curve0'], u)
            p2 = self._evaluate_nurbs_curve(surface['curve1'], u)
            point = (1 - v) * p1 + v * p2
            
            points.append(point)
        
        return np.array(points)
    
    def _evaluate_nurbs_curve(self, curve, u):
        """
        评估NURBS曲线上的点
        Args:
            curve: NURBS曲线参数
            u: 参数值
        Returns:
            曲线上的点
        """
        control_points = curve['control_points']
        weights = curve['weights']
        knot_vector = curve['knot_vector']
        degree = curve['degree']
        
        n = len(control_points) - 1
        numerator = np.zeros(3)
        denominator = 0.0
        
        for i in range(n + 1):
            basis = self._compute_basis_function(u, knot_vector, degree, i)
            weighted_basis = basis * weights[i]
            numerator += weighted_basis * control_points[i]
            denominator += weighted_basis
        
        if denominator > 1e-8:
            return numerator / denominator
        else:
            return np.zeros(3)
    
    def _compute_basis_function(self, u, knot_vector, degree, i):
        """
        计算B样条基函数
        Args:
            u: 参数值
            knot_vector: 节点向量
            degree: 曲线次数
            i: 基函数索引
        Returns:
            基函数值
        """
        if degree == 0:
            return 1.0 if knot_vector[i] <= u < knot_vector[i+1] else 0.0
        else:
            denominator1 = knot_vector[i+degree] - knot_vector[i]
            denominator2 = knot_vector[i+degree+1] - knot_vector[i+1]
            
            term1 = 0.0
            if denominator1 > 1e-8:
                term1 = (u - knot_vector[i]) / denominator1 * self._compute_basis_function(u, knot_vector, degree-1, i)
            
            term2 = 0.0
            if denominator2 > 1e-8:
                term2 = (knot_vector[i+degree+1] - u) / denominator2 * self._compute_basis_function(u, knot_vector, degree-1, i+1)
            
            return term1 + term2
    
    def generate_sample(self, straight_ratio=0.5):
        """
        生成一个训练样本
        Args:
            straight_ratio: 生成平直直纹面的比例
        Returns:
            样本字典
        """
        # 随机决定是否生成平直直纹面
        straight = np.random.rand() < straight_ratio
        
        # 生成直纹面
        surface = self.generate_developable_surface(straight=straight)
        
        # 生成边缘点列
        edges = self.generate_edge_points(surface)
        
        # 生成内部点云
        point_cloud = self.generate_interior_points(surface)
        
        # 提取NURBS参数
        curve0 = surface['curve0']
        curve1 = surface['curve1']
        
        # 构建NURBS参数向量
        nurbs_params = []
        
        # 曲线0参数
        nurbs_params.extend(curve0['control_points'].flatten())
        nurbs_params.extend(curve0['weights'])
        nurbs_params.extend(curve0['knot_vector'])
        nurbs_params.append(curve0['degree'])
        
        # 曲线1参数
        nurbs_params.extend(curve1['control_points'].flatten())
        nurbs_params.extend(curve1['weights'])
        nurbs_params.extend(curve1['knot_vector'])
        nurbs_params.append(curve1['degree'])
        
        nurbs_params = np.array(nurbs_params)
        
        return {
            'edges': np.array(edges),
            'point_cloud': point_cloud,
            'nurbs_params': nurbs_params,
            'surface': surface,
            'is_straight': straight
        }
    
    def generate_dataset(self, num_train=1000, num_test=200, straight_ratio=0.5):
        """
        生成数据集
        Args:
            num_train: 训练样本数量
            num_test: 测试样本数量
            straight_ratio: 生成平直直纹面的比例
        """
        print(f"生成训练数据集...")
        train_data = []
        for i in range(num_train):
            if (i + 1) % 100 == 0:
                print(f"生成训练样本 {i+1}/{num_train}")
            sample = self.generate_sample(straight_ratio=straight_ratio)
            train_data.append(sample)
        
        print(f"生成测试数据集...")
        test_data = []
        for i in range(num_test):
            if (i + 1) % 50 == 0:
                print(f"生成测试样本 {i+1}/{num_test}")
            sample = self.generate_sample(straight_ratio=straight_ratio)
            test_data.append(sample)
        
        # 保存数据
        train_path = os.path.join(self.train_dir, 'train_dataset.npy')
        test_path = os.path.join(self.test_dir, 'test_dataset.npy')
        
        np.save(train_path, train_data)
        np.save(test_path, test_data)
        
        print(f"训练数据保存到: {train_path}")
        print(f"测试数据保存到: {test_path}")
        print(f"训练样本数量: {len(train_data)}")
        print(f"测试样本数量: {len(test_data)}")
        print(f"平直直纹面比例: {straight_ratio}")


if __name__ == '__main__':
    # 创建数据生成器
    generator = NURBSSurfaceDataGenerator()
    
    # 生成数据集
    generator.generate_dataset(num_train=1000, num_test=200)
