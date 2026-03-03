"""
加工仿真验证工具
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from core.meshProcessor import MeshProcessor
from core.nonSphericalTool import NonSphericalTool

# 条件导入matplotlib
matplotlib_available = False
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    print("警告: matplotlib 未安装，可视化功能不可用")


class MachiningValidator:
    def __init__(self, mesh: MeshProcessor, tool: NonSphericalTool):
        """
        初始化加工验证器
        Args:
            mesh: 网格处理器
            tool: 刀具模型
        """
        self.mesh = mesh
        self.tool = tool
        
        # 验证结果
        self.validation_results = {
            'collision_check': False,
            'scallop_height': [],
            'path_smoothness': 0.0,
            'tool_orientation_change': 0.0,
            'metrics': {}
        }
    
    def check_collision(self, tool_paths: Dict[str, Any]) -> bool:
        """
        检查刀具路径是否与表面发生碰撞
        Args:
            tool_paths: 刀具路径数据
        Returns:
            True表示无碰撞，False表示有碰撞
        """
        print("检查碰撞...")
        
        collision_detected = False
        
        for path in tool_paths['paths']:
            if path['type'] == 'connection':
                continue  # 跳过连接路径
            
            points = path['points']
            orientations = path['orientations']
            
            for i, (point, orientation) in enumerate(zip(points, orientations)):
                # 找到最近的顶点
                distances = np.linalg.norm(self.mesh.vertices - point, axis=1)
                nearest_vertex = np.argmin(distances)
                normal = self.mesh.vertex_normals[nearest_vertex]
                
                # 检查碰撞
                is_collision = self.tool.check_collision_simple(
                    point,
                    normal,
                    orientation
                )
                
                if is_collision:
                    print(f"碰撞检测在路径点 {i}")
                    collision_detected = True
                    break
            
            if collision_detected:
                break
        
        self.validation_results['collision_check'] = not collision_detected
        return not collision_detected
    
    def calculate_scallop_height(self, tool_paths: Dict[str, Any]) -> List[float]:
        """
        计算残留高度
        Args:
            tool_paths: 刀具路径数据
        Returns:
            残留高度列表
        """
        print("计算残留高度...")
        
        scallop_heights = []
        
        # 提取所有路径的点
        all_points = []
        for path in tool_paths['paths']:
            if path['type'] == 'cc_path':
                all_points.extend(path['points'])
        
        # 计算相邻路径点之间的距离
        for i in range(len(all_points) - 1):
            for j in range(i + 1, min(i + 10, len(all_points))):  # 只检查附近的点
                distance = np.linalg.norm(all_points[i] - all_points[j])
                if 0.1 < distance < 5.0:  # 只考虑合理范围内的点
                    # 计算残留高度
                    tool_radius = self.tool.calculate_effective_radius(
                        gamma=np.pi / 4,
                        tilt_angle=0
                    )
                    if tool_radius > 0:
                        scallop_height = (distance ** 2) / (8 * tool_radius)
                        scallop_heights.append(scallop_height)
        
        self.validation_results['scallop_height'] = scallop_heights
        return scallop_heights
    
    def evaluate_path_smoothness(self, tool_paths: Dict[str, Any]) -> float:
        """
        评估路径平滑度
        Args:
            tool_paths: 刀具路径数据
        Returns:
            平滑度评分（0-1，越高越平滑）
        """
        print("评估路径平滑度...")
        
        total_curvature = 0.0
        total_segments = 0
        
        for path in tool_paths['paths']:
            points = path['points']
            
            for i in range(1, len(points) - 1):
                # 计算三点的曲率
                p_prev = points[i-1]
                p_curr = points[i]
                p_next = points[i+1]
                
                # 计算向量
                v1 = p_curr - p_prev
                v2 = p_next - p_curr
                
                # 计算曲率
                cross_product = np.cross(v1, v2)
                cross_norm = np.linalg.norm(cross_product)
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm > 0 and v2_norm > 0:
                    curvature = cross_norm / (v1_norm * v2_norm * np.linalg.norm(p_next - p_prev))
                    total_curvature += curvature
                    total_segments += 1
        
        if total_segments > 0:
            avg_curvature = total_curvature / total_segments
            # 转换为平滑度评分（曲率越小越平滑）
            smoothness = 1.0 / (1.0 + avg_curvature * 100)
        else:
            smoothness = 1.0
        
        self.validation_results['path_smoothness'] = smoothness
        return smoothness
    
    def evaluate_tool_orientation_change(self, tool_paths: Dict[str, Any]) -> float:
        """
        评估工具方向变化
        Args:
            tool_paths: 刀具路径数据
        Returns:
            方向变化评分（0-1，越高越稳定）
        """
        print("评估工具方向变化...")
        
        total_angle_change = 0.0
        total_transitions = 0
        
        for path in tool_paths['paths']:
            orientations = path['orientations']
            
            for i in range(1, len(orientations)):
                # 计算方向变化角度
                dot_product = np.dot(orientations[i-1], orientations[i])
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                total_angle_change += angle
                total_transitions += 1
        
        if total_transitions > 0:
            avg_angle_change = total_angle_change / total_transitions
            # 转换为方向稳定性评分（角度变化越小越稳定）
            stability = 1.0 / (1.0 + avg_angle_change * 10)
        else:
            stability = 1.0
        
        self.validation_results['tool_orientation_change'] = stability
        return stability
    
    def generate_report(self, tool_paths: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成验证报告
        Args:
            tool_paths: 刀具路径数据
        Returns:
            验证报告
        """
        print("生成验证报告...")
        
        # 运行所有验证
        collision_free = self.check_collision(tool_paths)
        scallop_heights = self.calculate_scallop_height(tool_paths)
        smoothness = self.evaluate_path_smoothness(tool_paths)
        orientation_stability = self.evaluate_tool_orientation_change(tool_paths)
        
        # 计算统计信息
        metrics = {
            'collision_free': collision_free,
            'average_scallop_height': np.mean(scallop_heights) if scallop_heights else 0,
            'max_scallop_height': np.max(scallop_heights) if scallop_heights else 0,
            'path_smoothness': smoothness,
            'orientation_stability': orientation_stability,
            'total_path_length': tool_paths.get('total_length', 0),
            'num_paths': tool_paths.get('num_paths', 0),
            'num_points': tool_paths.get('num_points', 0)
        }
        
        self.validation_results['metrics'] = metrics
        
        # 生成报告
        report = {
            'validation_results': self.validation_results,
            'metrics': metrics,
            'summary': self._generate_summary(metrics)
        }
        
        return report
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> str:
        """
        生成验证摘要
        Args:
            metrics: 验证指标
        Returns:
            摘要字符串
        """
        summary = "加工验证报告\n"
        summary += "=" * 50 + "\n"
        summary += f"碰撞检测: {'通过' if metrics['collision_free'] else '失败'}\n"
        summary += f"平均残留高度: {metrics['average_scallop_height']:.3f} mm\n"
        summary += f"最大残留高度: {metrics['max_scallop_height']:.3f} mm\n"
        summary += f"路径平滑度: {metrics['path_smoothness']:.3f} (0-1)\n"
        summary += f"方向稳定性: {metrics['orientation_stability']:.3f} (0-1)\n"
        summary += f"总路径长度: {metrics['total_path_length']:.2f} mm\n"
        summary += f"路径数量: {metrics['num_paths']}\n"
        summary += f"路径点数量: {metrics['num_points']}\n"
        summary += "=" * 50 + "\n"
        
        # 评估整体质量
        if metrics['collision_free']:
            if metrics['max_scallop_height'] < 0.5:
                summary += "整体评估: 优秀\n"
            elif metrics['max_scallop_height'] < 1.0:
                summary += "整体评估: 良好\n"
            else:
                summary += "整体评估: 需要改进\n"
        else:
            summary += "整体评估: 失败（存在碰撞）\n"
        
        return summary
    
    def visualize_validation(self, tool_paths: Dict[str, Any]):
        """
        可视化验证结果
        Args:
            tool_paths: 刀具路径数据
        """
        if not matplotlib_available:
            print("警告: matplotlib 未安装，跳过可视化")
            return
        
        print("可视化验证结果...")
        
        # 创建子图
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Machining Validation Results', fontsize=16)
        
        # 1. Scallop height distribution
        axs[0, 0].hist(self.validation_results['scallop_height'], bins=20)
        axs[0, 0].set_title('Scallop Height Distribution')
        axs[0, 0].set_xlabel('Scallop Height (mm)')
        axs[0, 0].set_ylabel('Frequency')
        
        # 2. Path smoothness
        axs[0, 1].bar(['Path Smoothness', 'Orientation Stability'], 
                      [self.validation_results['path_smoothness'], 
                       self.validation_results['tool_orientation_change']])
        axs[0, 1].set_title('Path Quality Metrics')
        axs[0, 1].set_ylim(0, 1)
        
        # 3. Path length distribution
        path_lengths = []
        for path in tool_paths['paths']:
            points = path['points']
            length = 0
            for i in range(len(points) - 1):
                length += np.linalg.norm(points[i+1] - points[i])
            path_lengths.append(length)
        
        axs[1, 0].hist(path_lengths, bins=20)
        axs[1, 0].set_title('Path Length Distribution')
        axs[1, 0].set_xlabel('Path Length (mm)')
        axs[1, 0].set_ylabel('Frequency')
        
        # 4. Collision detection result
        collision_result = "Pass" if self.validation_results['collision_check'] else "Fail"
        axs[1, 1].text(0.5, 0.5, f'Collision Check: {collision_result}', 
                      fontsize=16, ha='center', va='center')
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
