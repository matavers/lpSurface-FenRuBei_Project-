"""
神经网络推理脚本

用于加载训练好的模型并进行直纹面拟合推理。
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.neuralDevelopableSurfaceFitter import NeuralDevelopableSurfaceFitter


class NeuralDevelopableSurfaceInference:
    """
    神经网络直纹面拟合推理器
    """
    
    def __init__(self, model_path: str, device: torch.device = None):
        """
        初始化推理器
        Args:
            model_path: 模型权重文件路径
            device: 设备（CPU或GPU）
        """
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 创建模型
        self.model = NeuralDevelopableSurfaceFitter(M=16)
        self.model.to(self.device)
        
        # 加载模型权重
        self._load_model(model_path)
        
        # 模型设置为评估模式
        self.model.eval()
        
    def _load_model(self, model_path: str):
        """
        加载模型权重
        """
        try:
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"模型已从 {model_path} 加载")
            else:
                print(f"警告：模型文件 {model_path} 不存在")
        except Exception as e:
            print(f"加载模型时出错: {e}")
    
    def preprocess_sample(self, interior_points: np.ndarray, edge_points: list, 
                         partition_type: str) -> dict:
        """
        预处理样本
        Args:
            interior_points: 内部点云
            edge_points: 边缘点列列表
            partition_type: 分区类型 ('triangle' 或 'quadrilateral')
        Returns:
            预处理后的样本字典
        """
        # 处理内部点云
        max_interior_points = 1000
        if len(interior_points) > max_interior_points:
            indices = np.random.choice(len(interior_points), max_interior_points, replace=False)
            interior_points = interior_points[indices]
        elif len(interior_points) < max_interior_points:
            padding = np.zeros((max_interior_points - len(interior_points), 3))
            interior_points = np.vstack([interior_points, padding])
        
        # 处理边缘点列
        max_edge_points = 32
        padded_edges = []
        for edge in edge_points:
            edge = np.array(edge)
            if len(edge) > max_edge_points:
                indices = np.random.choice(len(edge), max_edge_points, replace=False)
                edge = edge[indices]
            elif len(edge) < max_edge_points:
                padding = np.zeros((max_edge_points - len(edge), 3))
                edge = np.vstack([edge, padding])
            padded_edges.append(edge)
        
        # 填充到4条边
        while len(padded_edges) < 4:
            padding = np.zeros((max_edge_points, 3))
            padded_edges.append(padding)
        
        # 分区类型
        if partition_type == 'triangle':
            partition_type_tensor = np.array([1, 0], dtype=np.float32)
        else:
            partition_type_tensor = np.array([0, 1], dtype=np.float32)
        
        return {
            'interior_points': torch.tensor(interior_points, dtype=torch.float32).to(self.device),
            'edge_points': torch.tensor(np.array(padded_edges), dtype=torch.float32).to(self.device),
            'partition_type': torch.tensor(partition_type_tensor, dtype=torch.float32).to(self.device)
        }
    
    def inference(self, interior_points: np.ndarray, edge_points: list, 
                 partition_type: str) -> dict:
        """
        执行推理
        Args:
            interior_points: 内部点云
            edge_points: 边缘点列列表
            partition_type: 分区类型 ('triangle' 或 'quadrilateral')
        Returns:
            直纹面参数
        """
        # 预处理
        sample = self.preprocess_sample(interior_points, edge_points, partition_type)
        
        # 推理
        with torch.no_grad():
            # 扩展维度以匹配模型输入形状 (B, N, 3)
            interior_points = sample['interior_points'].unsqueeze(0)
            edge_points = sample['edge_points'].unsqueeze(0)
            partition_type = sample['partition_type'].unsqueeze(0)
            
            # 前向传播
            control_points = self.model(interior_points, edge_points, partition_type)
            
            # 转换为numpy数组
            control_points = control_points.squeeze(0).cpu().numpy()
        
        # 生成直纹面参数
        if partition_type[0, 0].item() > 0.5:
            # 三角形分区
            vertex = control_points[-1]
            curve_points = control_points[:16]
            
            surface = {
                'type': 'conical',
                'vertex': vertex.tolist(),
                'curve': {
                    'type': 'b-spline',
                    'control_points': curve_points.tolist()
                }
            }
        else:
            # 四边形分区
            curve0_points = control_points[:16]
            curve1_points = control_points[16:32]
            
            surface = {
                'type': 'developable',
                'curve0': {
                    'type': 'b-spline',
                    'control_points': curve0_points.tolist()
                },
                'curve1': {
                    'type': 'b-spline',
                    'control_points': curve1_points.tolist()
                }
            }
        
        return surface
    
    def generate_surface_points(self, surface: dict, num_u: int = 32, 
                               num_v: int = 32) -> np.ndarray:
        """
        根据直纹面参数生成点云
        Args:
            surface: 直纹面参数
            num_u: u方向采样点数
            num_v: v方向采样点数
        Returns:
            直纹面点云
        """
        surface_points = []
        
        if surface['type'] == 'conical':
            # 锥面
            vertex = np.array(surface['vertex'])
            curve = np.array(surface['curve']['control_points'])
            M = len(curve)
            
            for u in np.linspace(0, 1, num_u):
                # 评估曲线上的点
                t = u * (M - 1)
                k = int(np.floor(t))
                if k >= M - 1:
                    k = M - 2
                t_local = t - k
                
                if k == 0:
                    p0, p1, p2 = curve[0], curve[1], curve[2]
                elif k == M - 2:
                    p0, p1, p2 = curve[-3], curve[-2], curve[-1]
                else:
                    p0, p1, p2 = curve[k], curve[k+1], curve[k+2]
                
                # 二次B样条
                curve_point = (1 - t_local)**2 / 2 * p0 + \
                             (1 - 2*t_local + t_local**2) * p1 + \
                             t_local**2 / 2 * p2
                
                for v in np.linspace(0, 1, num_v):
                    point = vertex + v * (curve_point - vertex)
                    surface_points.append(point)
        else:
            # 直纹面
            curve0 = np.array(surface['curve0']['control_points'])
            curve1 = np.array(surface['curve1']['control_points'])
            M = len(curve0)
            
            for u in np.linspace(0, 1, num_u):
                # 评估曲线0上的点
                t = u * (M - 1)
                k = int(np.floor(t))
                if k >= M - 1:
                    k = M - 2
                t_local = t - k
                
                if k == 0:
                    p0, p1, p2 = curve0[0], curve0[1], curve0[2]
                elif k == M - 2:
                    p0, p1, p2 = curve0[-3], curve0[-2], curve0[-1]
                else:
                    p0, p1, p2 = curve0[k], curve0[k+1], curve0[k+2]
                
                # 二次B样条
                curve0_point = (1 - t_local)**2 / 2 * p0 + \
                              (1 - 2*t_local + t_local**2) * p1 + \
                              t_local**2 / 2 * p2
                
                # 评估曲线1上的点
                if k == 0:
                    p0, p1, p2 = curve1[0], curve1[1], curve1[2]
                elif k == M - 2:
                    p0, p1, p2 = curve1[-3], curve1[-2], curve1[-1]
                else:
                    p0, p1, p2 = curve1[k], curve1[k+1], curve1[k+2]
                
                curve1_point = (1 - t_local)**2 / 2 * p0 + \
                              (1 - 2*t_local + t_local**2) * p1 + \
                              t_local**2 / 2 * p2
                
                for v in np.linspace(0, 1, num_v):
                    point = (1 - v) * curve0_point + v * curve1_point
                    surface_points.append(point)
        
        return np.array(surface_points)


def load_sample_from_file(file_path: str) -> dict:
    """
    从文件加载样本
    """
    sample = np.load(file_path, allow_pickle=True).item()
    return sample


def save_inference_result(result: dict, file_path: str):
    """
    保存推理结果
    """
    np.save(file_path, result)
    print(f"推理结果已保存到 {file_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='直纹面拟合神经网络推理')
    parser.add_argument('--model', type=str, default='data/neural/checkpoints/best_model.pth',
                        help='模型权重文件路径')
    parser.add_argument('--input', type=str, default=None,
                        help='输入样本文件路径')
    parser.add_argument('--output', type=str, default='inference_result.npy',
                        help='输出结果文件路径')
    
    args = parser.parse_args()
    
    # 初始化推理器
    inference = NeuralDevelopableSurfaceInference(args.model)
    
    if args.input:
        # 从文件加载样本
        sample = load_sample_from_file(args.input)
        
        # 执行推理
        result = inference.inference(
            sample['interior_points'],
            sample['edge_points'],
            sample['partition_type']
        )
        
        # 保存结果
        save_inference_result(result, args.output)
        
        print("推理完成！")
        print(f"直纹面类型: {result['type']}")
        if result['type'] == 'conical':
            print(f"顶点: {result['vertex']}")
            print(f"曲线控制点数量: {len(result['curve']['control_points'])}")
        else:
            print(f"曲线0控制点数量: {len(result['curve0']['control_points'])}")
            print(f"曲线1控制点数量: {len(result['curve1']['control_points'])}")
    else:
        # 示例推理
        print("执行示例推理...")
        
        # 生成示例数据
        from data.neural.data_generator import DevelopableSurfaceDataGenerator
        
        generator = DevelopableSurfaceDataGenerator(M=16)
        sample = generator.generate_quadrilateral_sample()
        sample = generator.normalize_sample(sample)
        
        # 执行推理
        result = inference.inference(
            sample['interior_points'],
            sample['edge_points'],
            sample['partition_type']
        )
        
        # 生成点云
        surface_points = inference.generate_surface_points(result)
        
        print("示例推理完成！")
        print(f"直纹面类型: {result['type']}")
        print(f"生成的点云数量: {len(surface_points)}")
        print(f"点云范围: {np.min(surface_points, axis=0)} - {np.max(surface_points, axis=0)}")
