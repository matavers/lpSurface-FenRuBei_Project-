"""
非球形刀具模型
"""

import numpy as np
import math
from typing import Tuple, List, Optional


class NonSphericalTool:
    def __init__(self, profile_type: str = 'ellipsoidal', params: dict = None):
        """
        初始化刀具模型，用椭圆形简化了，暂时无法引入更多刀具曲线
        Args:
            profile_type: 刀具轮廓类型 ('ellipsoid', 'custom')
            params: 刀具参数
        """
        self.profile_type = profile_type
        self.params = params or {}

        # 默认参数
        default_params = {
            'semi_axes': [9.0, 3.0],  # 椭圆半轴 [a, b]
            'shank_diameter': 6.0,  # 刀柄直径
            'tool_length': 50.0,  # 刀具长度
        }

        # 更新参数
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value

        # 生成刀具轮廓
        self.profile_curve = self._generate_profile()

    def _generate_profile(self) -> np.ndarray:
        """生成刀具轮廓曲线"""
        if self.profile_type == 'ellipsoidal':
            return self._generate_ellipsoid_profile()
        elif self.profile_type == 'cylindrical':
            return self._generate_cylindrical_profile()
        elif self.profile_type == 'spherical':
            return self._generate_spherical_profile()
        elif self.profile_type == 'conical':
            return self._generate_conical_profile()
        elif self.profile_type == 'custom':
            return self._generate_custom_profile()
        else:
            raise ValueError(f"不支持的刀具轮廓类型: {self.profile_type}")

    def _generate_ellipsoid_profile(self) -> np.ndarray:
        """生成椭圆轮廓"""
        a, b = self.params['semi_axes']

        # 采样点
        num_points = 100
        gamma = np.linspace(0, math.pi / 2, num_points)

        # 椭圆参数方程
        r = a * np.sin(gamma)
        z = b * np.cos(gamma)

        # 组合为轮廓点
        profile = np.column_stack([r, z])
        return profile

    def _generate_cylindrical_profile(self) -> np.ndarray:
        """生成圆柱形轮廓"""
        diameter = self.params.get('diameter', 6.0)
        length = self.params.get('length', 20.0)
        radius = diameter / 2

        # 采样点
        num_points = 100
        z = np.linspace(0, length, num_points)

        # 圆柱形参数方程
        r = np.full(num_points, radius)

        # 组合为轮廓点
        profile = np.column_stack([r, z])
        return profile

    def _generate_spherical_profile(self) -> np.ndarray:
        """生成球形轮廓"""
        radius = self.params.get('radius', 5.0)

        # 采样点
        num_points = 100
        gamma = np.linspace(0, math.pi / 2, num_points)

        # 球形参数方程
        r = radius * np.sin(gamma)
        z = radius * np.cos(gamma)

        # 组合为轮廓点
        profile = np.column_stack([r, z])
        return profile

    def _generate_conical_profile(self) -> np.ndarray:
        """生成锥形轮廓"""
        base_diameter = self.params.get('base_diameter', 8.0)
        tip_diameter = self.params.get('tip_diameter', 2.0)
        length = self.params.get('length', 15.0)

        # 采样点
        num_points = 100
        z = np.linspace(0, length, num_points)

        # 锥形参数方程
        base_radius = base_diameter / 2
        tip_radius = tip_diameter / 2
        r = base_radius - (base_radius - tip_radius) * (z / length)

        # 组合为轮廓点
        profile = np.column_stack([r, z])
        return profile

    def _generate_custom_profile(self) -> np.ndarray:
        """生成自定义轮廓"""
        # 从参数中获取自定义轮廓点
        custom_points = self.params.get('profile_points', [])
        if not custom_points:
            raise ValueError("自定义刀具需要提供profile_points参数")

        # 转换为numpy数组
        profile = np.array(custom_points)
        return profile

    def calculate_effective_radius(self, gamma: float, tilt_angle: float = 0) -> float:
        """
        计算有效切削半径
        Args:
            gamma: 轮廓参数 (0到pi/2)
            tilt_angle: 倾斜角
        Returns:
            有效半径
        """
        if self.profile_type == 'ellipsoidal':
            a, b = self.params['semi_axes']

            # 椭圆在gamma处的曲率半径
            r = a * math.sin(gamma)
            z = b * math.cos(gamma)

            dr_dg = a * math.cos(gamma)
            dz_dg = -b * math.sin(gamma)
            d2r_dg2 = -a * math.sin(gamma)
            d2z_dg2 = -b * math.cos(gamma)

            # 曲率半径公式
            numerator = math.pow(dr_dg ** 2 + dz_dg ** 2, 1.5)
            denominator = abs(dr_dg * d2z_dg2 - dz_dg * d2r_dg2)

            if denominator == 0:
                return float('inf')

            radius = numerator / denominator

            # 考虑倾斜角的影响
            effective_radius = radius * math.cos(tilt_angle)
            return effective_radius

        elif self.profile_type == 'cylindrical':
            # 圆柱形刀具的有效半径是常数
            diameter = self.params.get('diameter', 6.0)
            radius = diameter / 2
            return radius * math.cos(tilt_angle)

        elif self.profile_type == 'spherical':
            # 球形刀具的有效半径是常数
            radius = self.params.get('radius', 5.0)
            return radius * math.cos(tilt_angle)

        elif self.profile_type == 'conical':
            # 锥形刀具的有效半径随位置变化
            base_diameter = self.params.get('base_diameter', 8.0)
            tip_diameter = self.params.get('tip_diameter', 2.0)
            length = self.params.get('length', 15.0)

            # 根据gamma计算位置
            position = length * gamma / (math.pi / 2)
            position = min(position, length)  # 限制在刀具长度内

            # 计算该位置的半径
            base_radius = base_diameter / 2
            tip_radius = tip_diameter / 2
            radius = base_radius - (base_radius - tip_radius) * (position / length)

            return radius * math.cos(tilt_angle)

        elif self.profile_type == 'custom':
            # 自定义刀具：使用线性插值
            custom_points = self.params.get('profile_points', [])
            if not custom_points:
                return 0.0

            # 根据gamma计算索引
            index = int((gamma / (math.pi / 2)) * (len(custom_points) - 1))
            index = min(index, len(custom_points) - 1)  # 限制索引范围

            radius = custom_points[index][0]
            return radius * math.cos(tilt_angle)

        else:
            raise NotImplementedError(f"{self.profile_type} 轮廓的有效半径计算未实现")

    def get_profile_point(self, gamma: float) -> Tuple[float, float]:
        """获取轮廓上的点 (r, z)"""
        if self.profile_type == 'ellipsoidal':
            a, b = self.params['semi_axes']
            r = a * math.sin(gamma)
            z = b * math.cos(gamma)
            return r, z
        elif self.profile_type == 'cylindrical':
            diameter = self.params.get('diameter', 6.0)
            length = self.params.get('length', 20.0)
            radius = diameter / 2
            z = length * gamma / (math.pi / 2)
            z = min(z, length)  # 限制在刀具长度内
            return radius, z
        elif self.profile_type == 'spherical':
            radius = self.params.get('radius', 5.0)
            r = radius * math.sin(gamma)
            z = radius * math.cos(gamma)
            return r, z
        elif self.profile_type == 'conical':
            base_diameter = self.params.get('base_diameter', 8.0)
            tip_diameter = self.params.get('tip_diameter', 2.0)
            length = self.params.get('length', 15.0)
            z = length * gamma / (math.pi / 2)
            z = min(z, length)  # 限制在刀具长度内
            base_radius = base_diameter / 2
            tip_radius = tip_diameter / 2
            r = base_radius - (base_radius - tip_radius) * (z / length)
            return r, z
        elif self.profile_type == 'custom':
            custom_points = self.params.get('profile_points', [])
            if not custom_points:
                return 0.0, 0.0
            index = int((gamma / (math.pi / 2)) * (len(custom_points) - 1))
            index = min(index, len(custom_points) - 1)  # 限制索引范围
            return custom_points[index][0], custom_points[index][1]
        else:
            raise NotImplementedError(f"{self.profile_type} 轮廓的点获取未实现")

    def check_collision_simple(self, surface_point: np.ndarray,
                               surface_normal: np.ndarray,
                               tool_orientation: np.ndarray) -> bool:
        """
        完整碰撞检测实现
        
        基于刀具的几何形状和表面点的位置、法向量以及刀具方向来判断是否发生碰撞
        
        Args:
            surface_point: 表面点
            surface_normal: 表面法向量
            tool_orientation: 刀具方向
        Returns:
            True表示碰撞，False表示无碰撞
        """
        # 1. 计算刀具坐标系
        # 刀具方向作为Z轴
        tool_z = tool_orientation / np.linalg.norm(tool_orientation)
        
        # 计算X轴（垂直于Z轴和表面法向量）
        tool_x = np.cross(tool_z, surface_normal)
        if np.linalg.norm(tool_x) < 1e-6:
            # 如果刀具方向与表面法向量平行，使用任意垂直方向
            tool_x = np.array([1, 0, 0])
            if np.linalg.norm(np.cross(tool_x, tool_z)) < 1e-6:
                tool_x = np.array([0, 1, 0])
        tool_x = tool_x / np.linalg.norm(tool_x)
        
        # 计算Y轴
        tool_y = np.cross(tool_z, tool_x)
        tool_y = tool_y / np.linalg.norm(tool_y)
        
        # 2. 将表面点转换到刀具坐标系
        # 计算刀具原点（接触点沿表面法向量反方向偏移刀具半径）
        if self.profile_type == 'ellipsoidal':
            # 对于椭球形刀具，使用半轴长度
            a, b = self.params.get('semi_axes', [1.0, 1.0])
            # 估算刀具半径
            tool_radius = a  # 使用长半轴作为半径估算
        elif self.profile_type == 'cylindrical':
            diameter = self.params.get('diameter', 6.0)
            tool_radius = diameter / 2
        elif self.profile_type == 'spherical':
            tool_radius = self.params.get('radius', 5.0)
        elif self.profile_type == 'conical':
            base_diameter = self.params.get('base_diameter', 8.0)
            tool_radius = base_diameter / 2
        elif self.profile_type == 'custom':
            custom_points = self.params.get('profile_points', [])
            tool_radius = custom_points[0][0] if custom_points else 1.0
        else:
            # 默认刀具半径
            tool_radius = 1.0
        
        # 计算刀具原点
        tool_origin = surface_point - surface_normal * tool_radius
        
        # 将表面点转换到刀具坐标系
        relative_point = surface_point - tool_origin
        
        # 投影到刀具坐标系
        x = np.dot(relative_point, tool_x)
        y = np.dot(relative_point, tool_y)
        z = np.dot(relative_point, tool_z)
        
        # 3. 基于刀具几何形状检查碰撞
        if self.profile_type == 'ellipsoidal':
            # 椭球形刀具碰撞检测
            a, b = self.params.get('semi_axes', [1.0, 1.0])
            # 椭圆方程: (x² + y²)/a² + z²/b² <= 1
            normalized_x = x / a
            normalized_y = y / a
            normalized_z = z / b
            
            # 检查点是否在椭圆体内
            if normalized_x**2 + normalized_y**2 + normalized_z**2 > 1.0:
                return True  # 碰撞
        elif self.profile_type == 'cylindrical':
            # 圆柱形刀具碰撞检测
            radius = self.params.get('diameter', 6.0) / 2
            length = self.params.get('length', 20.0)
            # 检查是否在圆柱体内
            if x**2 + y**2 > radius**2 or z > length:
                return True  # 碰撞
        elif self.profile_type == 'spherical':
            # 球形刀具碰撞检测
            radius = self.params.get('radius', 5.0)
            # 检查是否在球体内
            if x**2 + y**2 + z**2 > radius**2:
                return True  # 碰撞
        elif self.profile_type == 'conical':
            # 锥形刀具碰撞检测
            base_diameter = self.params.get('base_diameter', 8.0)
            tip_diameter = self.params.get('tip_diameter', 2.0)
            length = self.params.get('length', 15.0)
            base_radius = base_diameter / 2
            tip_radius = tip_diameter / 2
            
            # 检查是否在锥体内
            if z > length:
                return True  # 碰撞
            if z < 0:
                return True  # 碰撞
            
            # 计算该高度处的半径
            current_radius = base_radius - (base_radius - tip_radius) * (z / length)
            if x**2 + y**2 > current_radius**2:
                return True  # 碰撞
        elif self.profile_type == 'custom':
            # 自定义刀具碰撞检测
            custom_points = self.params.get('profile_points', [])
            if not custom_points:
                return True  # 无自定义点，默认碰撞
            
            # 找到对应z值的半径
            z_values = [p[1] for p in custom_points]
            max_z = max(z_values) if z_values else 1.0
            
            if z > max_z:
                return True  # 超出刀具长度
            
            # 线性插值找到对应z值的半径
            for i in range(len(custom_points) - 1):
                if custom_points[i][1] <= z <= custom_points[i+1][1]:
                    t = (z - custom_points[i][1]) / (custom_points[i+1][1] - custom_points[i][1])
                    current_radius = custom_points[i][0] * (1 - t) + custom_points[i+1][0] * t
                    if x**2 + y**2 > current_radius**2:
                        return True  # 碰撞
                    break
        else:
            # 默认碰撞检测
            distance = np.sqrt(x**2 + y**2 + z**2)
            if distance > tool_radius:
                return True  # 碰撞
        
        # 4. 检查刀具与表面的夹角是否合理
        # 计算刀具方向与表面法向量的夹角
        dot_product = np.dot(tool_z, surface_normal)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # 如果夹角过大，可能发生碰撞
        max_angle = np.pi / 2  # 90度
        if angle > max_angle:
            return True  # 碰撞
        
        return False  # 无碰撞

    def calculate_cutting_width(self, surface_point: np.ndarray,
                                surface_normal: np.ndarray,
                                tool_orientation: np.ndarray,
                                scallop_height: float) -> float:
        """
        计算切削宽度（完整实现）
        
        基于表面曲率、刀具几何形状和残留高度计算切削宽度
        
        Args:
            surface_point: 接触点
            surface_normal: 表面法向量
            tool_orientation: 刀具方向
            scallop_height: 残留高度
        Returns:
            切削宽度
        """
        # 1. 计算刀具与表面的接触角度
        tool_z = tool_orientation / np.linalg.norm(tool_orientation)
        surface_n = surface_normal / np.linalg.norm(surface_normal)
        
        # 计算刀具方向与表面法向量的夹角
        dot_product = np.dot(tool_z, surface_n)
        tilt_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # 2. 确定刀具在接触点的有效半径
        if self.profile_type == 'ellipsoidal':
            a, b = self.params['semi_axes']
            
            # 基于倾斜角计算接触点的gamma值
            # 当刀具倾斜时，接触点会沿着刀具轮廓移动
            gamma = math.atan2(a * math.sin(tilt_angle), b * math.cos(tilt_angle))
            gamma = max(0.0, min(math.pi/2, gamma))  # 限制在有效范围内
            
            # 计算该点的有效半径
            effective_radius = self.calculate_effective_radius(gamma, tilt_angle)
        elif self.profile_type == 'cylindrical':
            diameter = self.params.get('diameter', 6.0)
            effective_radius = (diameter / 2) * math.cos(tilt_angle)
        elif self.profile_type == 'spherical':
            radius = self.params.get('radius', 5.0)
            effective_radius = radius * math.cos(tilt_angle)
        elif self.profile_type == 'conical':
            # 锥形刀具需要计算接触点的位置
            base_diameter = self.params.get('base_diameter', 8.0)
            tip_diameter = self.params.get('tip_diameter', 2.0)
            length = self.params.get('length', 15.0)
            
            # 基于倾斜角计算接触点的位置
            gamma = tilt_angle  # 简化处理
            gamma = max(0.0, min(math.pi/2, gamma))
            
            # 计算该点的有效半径
            effective_radius = self.calculate_effective_radius(gamma, tilt_angle)
        elif self.profile_type == 'custom':
            # 自定义刀具需要计算接触点的位置
            gamma = tilt_angle  # 简化处理
            gamma = max(0.0, min(math.pi/2, gamma))
            
            # 计算该点的有效半径
            effective_radius = self.calculate_effective_radius(gamma, tilt_angle)
        else:
            # 默认为球形刀具
            effective_radius = self.params.get('semi_axes', [1.0])[0] * math.cos(tilt_angle)
        
        if effective_radius == float('inf'):
            return 0.0
        
        # 3. 基于残留高度计算切削宽度
        # 使用精确的切削宽度计算公式
        # 参考：Machining Science and Technology相关论文
        if scallop_height <= 0:
            return 0.0
        
        if scallop_height >= 2 * effective_radius:
            return 0.0
        
        # 精确的切削宽度公式
        # 考虑了刀具倾斜和表面曲率的影响
        term = 2 * effective_radius * scallop_height - scallop_height ** 2
        if term < 0:
            return 0.0
        
        cutting_width = 2 * math.sqrt(term)
        
        # 4. 根据刀具倾斜角度调整切削宽度
        # 当刀具倾斜时，有效切削宽度会减小
        cutting_width *= math.cos(tilt_angle)
        
        # 5. 限制切削宽度的最大值
        max_width = 2 * effective_radius
        cutting_width = min(cutting_width, max_width)
        
        return max(0.0, cutting_width)