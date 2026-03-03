"""
参数优化模块
用于优化分区算法的参数
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from core.advancedSurfacePartitioner import AdvancedSurfacePartitioner
from core.meshProcessor import MeshProcessor
from utils.visualization import Visualizer


class ParameterOptimizer:
    def __init__(self, mesh_processor: MeshProcessor, tool):
        """
        初始化参数优化器
        Args:
            mesh_processor: 网格处理器
            tool: 刀具对象
        """
        self.mesh_processor = mesh_processor
        self.tool = tool
        self.visualizer = Visualizer()
    
    def optimize_resolution(self, initial_resolution: float = 0.1, 
                          min_resolution: float = 0.01, 
                          max_resolution: float = 1.0, 
                          step: float = 0.05) -> float:
        """
        优化分辨率参数
        Args:
            initial_resolution: 初始分辨率
            min_resolution: 最小分辨率
            max_resolution: 最大分辨率
            step: 步长
        Returns:
            最优分辨率
        """
        print("优化分辨率参数...")
        
        best_resolution = initial_resolution
        best_quality = 0.0
        
        # 尝试不同的分辨率值
        resolutions = np.arange(min_resolution, max_resolution + step, step)
        
        for resolution in resolutions:
            print(f"测试分辨率: {resolution}")
            
            # 创建分区器
            partitioner = AdvancedSurfacePartitioner(
                self.mesh_processor, 
                self.tool, 
                resolution=resolution
            )
            
            # 执行分区
            labels, edge_midpoints = partitioner.partition_surface()
            
            # 评估分区质量
            mesh = self.mesh_processor.get_open3d_mesh()
            quality_metrics = self.visualizer.evaluate_partition_quality(mesh, labels)
            
            # 更新最优参数
            if quality_metrics['overall_quality'] > best_quality:
                best_quality = quality_metrics['overall_quality']
                best_resolution = resolution
                print(f"发现更优分辨率: {best_resolution} (质量: {best_quality:.4f})")
        
        print(f"分辨率优化完成: 最优分辨率 = {best_resolution} (质量: {best_quality:.4f})")
        return best_resolution
    
    def optimize_all_parameters(self, param_ranges: Dict[str, Tuple[float, float, float]] = None) -> Dict[str, float]:
        """
        优化所有参数
        Args:
            param_ranges: 参数范围字典，格式为 {param_name: (min, max, step)}
        Returns:
            最优参数字典
        """
        print("优化所有参数...")
        
        # 默认参数范围
        if param_ranges is None:
            param_ranges = {
                'resolution': (0.01, 1.0, 0.1),
            }
        
        best_params = {}
        best_quality = 0.0
        
        # 网格搜索所有参数组合
        # 这里简化为逐个优化参数
        for param_name, (min_val, max_val, step) in param_ranges.items():
            print(f"优化参数: {param_name}")
            
            if param_name == 'resolution':
                best_params[param_name] = self.optimize_resolution(
                    initial_resolution=0.1,
                    min_resolution=min_val,
                    max_resolution=max_val,
                    step=step
                )
        
        # 评估最优参数组合
        partitioner = AdvancedSurfacePartitioner(
            self.mesh_processor, 
            self.tool, 
            resolution=best_params.get('resolution', 0.1)
        )
        
        labels, edge_midpoints = partitioner.partition_surface()
        mesh = self.mesh_processor.get_open3d_mesh()
        quality_metrics = self.visualizer.evaluate_partition_quality(mesh, labels)
        
        print("参数优化完成:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"  最佳质量: {quality_metrics['overall_quality']:.4f}")
        
        return best_params
    
    def interactive_parameter_tuning(self):
        """
        交互式参数调整
        """
        print("交互式参数调整")
        print("================")
        
        # 初始参数
        resolution = 0.1
        
        while True:
            print(f"当前分辨率: {resolution}")
            
            # 创建分区器
            partitioner = AdvancedSurfacePartitioner(
                self.mesh_processor, 
                self.tool, 
                resolution=resolution
            )
            
            # 执行分区
            labels, edge_midpoints = partitioner.partition_surface()
            
            # 可视化结果
            mesh = self.mesh_processor.get_open3d_mesh()
            self.visualizer.visualize_partitions_with_midpoints(mesh, labels, edge_midpoints)
            
            # 询问用户
            user_input = input("输入新的分辨率值（按Enter保持当前值，输入'q'退出）: ")
            
            if user_input.lower() == 'q':
                break
            elif user_input:
                try:
                    new_resolution = float(user_input)
                    if new_resolution > 0:
                        resolution = new_resolution
                except ValueError:
                    print("无效的输入，请输入数字")
        
        print("交互式参数调整完成")
        return resolution
