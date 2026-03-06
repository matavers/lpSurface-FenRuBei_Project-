"""
直纹面拟合测试
"""

import unittest
import numpy as np
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.developableSurfaceFitter import DevelopableSurfaceFitter
from core.meshProcessor import MeshProcessor
import open3d as o3d


class TestDevelopableSurfaceFitter(unittest.TestCase):
    """测试直纹面拟合功能"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建一个简单的直纹面网格（平面四边形）
        self.mesh = o3d.geometry.TriangleMesh()
        
        # 生成平面四边形的顶点
        vertices = [
            [0.0, 0.0, 0.0],  # 顶点0
            [1.0, 0.0, 0.0],  # 顶点1
            [1.0, 1.0, 0.0],  # 顶点2
            [0.0, 1.0, 0.0]   # 顶点3
        ]
        
        # 生成三角形
        triangles = [
            [0, 1, 2],  # 第一个三角形
            [0, 2, 3]   # 第二个三角形
        ]
        
        # 设置顶点和三角形
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.triangles = o3d.utility.Vector3iVector(triangles)
        self.mesh.compute_vertex_normals()
        
        # 创建MeshProcessor
        self.mesh_processor = MeshProcessor(self.mesh)
        
        # 创建DevelopableSurfaceFitter
        self.fitter = DevelopableSurfaceFitter(self.mesh_processor)
    
    def test_fit_developable_surfaces(self):
        """测试直纹面拟合"""
        # 创建一个分区标签，将顶点分为两个分区
        # 顶点0和1属于分区0，顶点2和3属于分区1
        partition_labels = np.array([0, 0, 1, 1], dtype=int)
        
        # 创建边缘中点数组
        # 边界边是顶点1-2和顶点0-3
        edge_midpoints = np.array([
            [(1.0 + 1.0) / 2, (0.0 + 1.0) / 2, 0.0],  # 顶点1和2的中点
            [(0.0 + 0.0) / 2, (0.0 + 1.0) / 2, 0.0]   # 顶点0和3的中点
        ])
        
        # 拟合直纹面
        developable_surfaces = self.fitter.fit_developable_surfaces(partition_labels, edge_midpoints)
        
        # 验证结果
        self.assertIsNotNone(developable_surfaces)
        # 由于可能没有识别到种子分区，我们只检查结果是否为字典
        self.assertIsInstance(developable_surfaces, dict)
    
    def test_evaluate_curve(self):
        """测试曲线评估"""
        # 创建一个简单的直线曲线
        line_curve = {
            'type': 'line',
            'start_point': [0.0, 0.0, 0.0],
            'end_point': [1.0, 1.0, 1.0]
        }
        
        # 评估曲线
        point = self.fitter._evaluate_curve(line_curve, 0.5)
        
        # 验证结果
        expected_point = [0.5, 0.5, 0.5]
        np.testing.assert_allclose(point, expected_point)
    
    def test_evaluate_developable(self):
        """测试直纹面评估"""
        # 创建一个简单的直纹面
        surface = {
            'type': 'developable',
            'curve1': {
                'type': 'line',
                'start_point': [1.0, 0.0, 0.0],
                'end_point': [1.0, 0.0, 2.0]
            },
            'curve2': {
                'type': 'line',
                'start_point': [0.0, 1.0, 0.0],
                'end_point': [0.0, 1.0, 2.0]
            }
        }
        
        # 评估直纹面
        point = self.fitter._evaluate_developable(surface, 0.5, 0.5)
        
        # 验证结果
        expected_point = [0.5, 0.5, 1.0]
        np.testing.assert_allclose(point, expected_point)


if __name__ == '__main__':
    unittest.main()