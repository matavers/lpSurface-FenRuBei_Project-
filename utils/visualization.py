"""
可视化工具
使用Matplotlib和Open3D实现可视化
"""

import numpy as np
import open3d as o3d
from typing import List, Tuple, Dict, Any

# 尝试导入matplotlib
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("警告: matplotlib 未安装，某些可视化功能可能受限")
    MATPLOTLIB_AVAILABLE = False


class Visualizer:
    def __init__(self):
        self.color_palette = self._create_color_palette()

    def _create_color_palette(self, n_colors: int = 20) -> List[Tuple[float, float, float]]:
        """创建颜色调色板"""
        colors = []
        for i in range(n_colors):
            hue = (i * 0.618) % 1.0  # 黄金比例
            if MATPLOTLIB_AVAILABLE:
                # 使用Matplotlib的HSV颜色空间
                color = plt.cm.hsv(hue)
                colors.append(color[:3])  # 只取RGB部分
            else:
                # 简单的颜色生成
                r = (hue * 6.0) % 1.0
                g = ((hue * 6.0) + 2.0) % 1.0
                b = ((hue * 6.0) + 4.0) % 1.0
                colors.append((r, g, b))
        return colors
    
    def _show_legend(self, unique_labels):
        """
        显示颜色对应分区编号的图例表
        
        Args:
            unique_labels: 唯一的分区标签列表
        """
        print("显示分区颜色图例...")
        
        # 创建图例点云
        legend_points = []
        legend_colors = []
        legend_texts = []
        
        # 为每个分区创建一个点
        for i, label in enumerate(unique_labels):
            # 计算点的位置（网格排列）
            row = i // 5  # 每行5个
            col = i % 5
            x = col * 0.5
            y = -row * 0.5
            z = 0.0
            
            # 添加多个点形成一个色块
            for dx in [-0.1, 0, 0.1]:
                for dy in [-0.1, 0, 0.1]:
                    legend_points.append([x + dx, y + dy, z])
                    color_idx = int(label) % len(self.color_palette)
                    legend_colors.append(self.color_palette[color_idx])
            
            # 保存文本信息
            legend_texts.append(f"分区 {int(label)}")
        
        # 创建点云
        legend_pcd = o3d.geometry.PointCloud()
        legend_pcd.points = o3d.utility.Vector3dVector(legend_points)
        legend_pcd.colors = o3d.utility.Vector3dVector(legend_colors)
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="分区颜色图例", width=800, height=600)
        
        # 添加点云
        vis.add_geometry(legend_pcd)
        
        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.background_color = np.array([1, 1, 1])  # 白色背景
        
        # 运行可视化
        print("图例说明:")
        for i, label in enumerate(unique_labels):
            color_idx = int(label) % len(self.color_palette)
            color = self.color_palette[color_idx]
            print(f"分区 {int(label)}: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
        
        vis.run()
        vis.destroy_window()
        print("图例显示完成")

    def evaluate_partition_quality(self, mesh: o3d.geometry.TriangleMesh, partition_labels: np.ndarray) -> Dict[str, float]:
        """
        评估分区质量
        Args:
            mesh: 网格对象
            partition_labels: 分区标签数组
        Returns:
            分区质量评估结果
        """
        print("评估分区质量...")
        
        vertices = np.asarray(mesh.vertices)
        unique_labels = np.unique(partition_labels)
        num_partitions = len(unique_labels)
        
        # 计算每个分区的统计信息
        partition_stats = {}
        for label in unique_labels:
            partition_vertices = vertices[partition_labels == label]
            if len(partition_vertices) > 0:
                # 计算分区大小
                partition_size = len(partition_vertices)
                
                # 计算分区中心点
                centroid = np.mean(partition_vertices, axis=0)
                
                # 计算分区的空间范围
                min_coords = np.min(partition_vertices, axis=0)
                max_coords = np.max(partition_vertices, axis=0)
                extent = max_coords - min_coords
                
                # 计算分区内顶点的平均距离
                distances = np.linalg.norm(partition_vertices - centroid, axis=1)
                avg_distance = np.mean(distances)
                
                partition_stats[label] = {
                    'size': partition_size,
                    'centroid': centroid,
                    'extent': extent,
                    'avg_distance': avg_distance
                }
        
        # 计算分区质量指标
        quality_metrics = {}
        
        # 1. 分区大小均衡性
        partition_sizes = [stats['size'] for stats in partition_stats.values()]
        if partition_sizes:
            mean_size = np.mean(partition_sizes)
            std_size = np.std(partition_sizes)
            size_balance = 1.0 - (std_size / mean_size) if mean_size > 0 else 0.0
            quality_metrics['size_balance'] = size_balance
        
        # 2. 分区空间分布
        centroids = [stats['centroid'] for stats in partition_stats.values()]
        if len(centroids) > 1:
            centroid_distances = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    distance = np.linalg.norm(centroids[i] - centroids[j])
                    centroid_distances.append(distance)
            avg_centroid_distance = np.mean(centroid_distances)
            quality_metrics['spatial_distribution'] = avg_centroid_distance
        
        # 3. 分区数量合理性
        # 基于网格大小的分区数量评估
        optimal_partitions = min(100, max(1, len(vertices) // 1000))
        partition_count_score = 1.0 - abs(num_partitions - optimal_partitions) / optimal_partitions
        quality_metrics['partition_count'] = max(0.0, partition_count_score)
        
        # 4. 总体质量分数
        quality_metrics['overall_quality'] = np.mean(list(quality_metrics.values())) if quality_metrics else 0.0
        
        print("分区质量评估完成:")
        for metric, value in quality_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return quality_metrics

    def visualize_partitions_with_midpoints(self, mesh: o3d.geometry.TriangleMesh,
                                           partition_labels: np.ndarray,
                                           edge_midpoints: np.ndarray):
        """
        在一个窗口中可视化表面分区和边缘中点
        Args:
            mesh: 原始网格对象
            partition_labels: 分区标签数组
            edge_midpoints: 边缘中点数组
        """
        # 检查网格是否有效
        if not mesh or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
            print("警告: 网格无效，跳过可视化")
            return
        
        # 创建网格副本
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        new_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
        new_mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
        new_mesh.triangle_normals = o3d.utility.Vector3dVector(np.asarray(mesh.triangle_normals))

        # 为每个分区设置不同的颜色
        print("为分区设置颜色...")
        unique_labels = np.unique(partition_labels)
        num_partitions = len(unique_labels)
        
        # 评估分区质量
        quality_metrics = self.evaluate_partition_quality(mesh, partition_labels)
        
        # 创建颜色映射
        vertex_colors = []
        for label in partition_labels:
            # 使用调色板中的颜色，循环使用
            color_idx = int(label) % len(self.color_palette)
            vertex_colors.append(self.color_palette[color_idx])
        
        # 设置顶点颜色
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        # 创建边缘中点云
        print(f"创建边缘中点云: {len(edge_midpoints)} 个点")
        geometries = [new_mesh]
        if len(edge_midpoints) > 0:
            midpoint_pcd = o3d.geometry.PointCloud()
            midpoint_pcd.points = o3d.utility.Vector3dVector(edge_midpoints)
            # 设置中点颜色为红色
            midpoint_pcd.paint_uniform_color([1, 0, 0])
            geometries.append(midpoint_pcd)

        # 可视化
        print(f"分区和中点可视化完成: {num_partitions} 个分区, {len(edge_midpoints)} 个中点")
        print(f"分区质量: {quality_metrics['overall_quality']:.4f}")
        
        try:
            # 使用draw_geometries函数进行可视化
            o3d.visualization.draw_geometries(
                geometries,
                window_name=f"分区和中点可视化: {num_partitions} 个分区 (质量: {quality_metrics['overall_quality']:.2f})",
                width=1024,
                height=768
            )
        except Exception as e:
            print(f"可视化失败: {e}")

    def visualize_tool_orientations(self, mesh: o3d.geometry.TriangleMesh,
                                    tool_orientations: np.ndarray,
                                    scale: float = 0.05,
                                    name_suffix: str = "_orientations",
                                    vertex_sample_rate: float = 0.1):
        """
        可视化工具方向场
        Args:
            mesh: 网格对象
            tool_orientations: 工具方向数组
            scale: 箭头缩放
            name_suffix: 新对象名称后缀
            vertex_sample_rate: 顶点采样率，范围0-1，控制显示箭头的顶点比例
        """
        # 获取顶点坐标
        vertices = np.asarray(mesh.vertices)

        # 顶点采样
        num_vertices = len(vertices)
        sample_size = max(100, min(1000, int(num_vertices * vertex_sample_rate)))
        if num_vertices > sample_size:
            sample_indices = np.random.choice(num_vertices, sample_size, replace=False)
            vertices = vertices[sample_indices]
            tool_orientations = tool_orientations[sample_indices]
        print(f"使用 {len(vertices)} 个采样顶点进行方向场可视化")

        # 创建箭头列表
        arrows = []
        for i, (vertex, orientation) in enumerate(zip(vertices, tool_orientations)):
            if np.linalg.norm(orientation) < 0.1:
                continue

            # 计算箭头终点
            end_point = vertex + orientation * scale

            # 创建箭头
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=scale * 0.1,
                cone_radius=scale * 0.2,
                cylinder_height=scale * 0.5,
                cone_height=scale * 0.3
            )

            # 计算旋转矩阵
            direction = orientation / np.linalg.norm(orientation)
            up = np.array([0, 0, 1])
            if np.dot(direction, up) > 0.99:
                rotation_matrix = np.eye(3)
            else:
                # 计算旋转轴和角度
                axis = np.cross(up, direction)
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.dot(up, direction))
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

            # 应用变换
            arrow.rotate(rotation_matrix, center=(0, 0, 0))
            arrow.translate(vertex)

            # 设置颜色
            color = np.abs(orientation[:3])
            color = color / np.max(color) if np.max(color) > 0 else color
            arrow.paint_uniform_color(color)

            arrows.append(arrow)

        # 可视化
        if arrows:
            o3d.visualization.draw_geometries([mesh] + arrows, 
                                             window_name=f"方向场可视化: {len(arrows)} 个箭头")

        print(f"方向场可视化完成: {len(arrows)} 个箭头")

    def visualize_tool_paths(self, paths: List[Dict[str, Any]],
                             name: str = "ToolPaths",
                             color: Tuple[float, float, float] = (1.0, 0.0, 0.0)):
        """
        可视化刀具路径
        Args:
            paths: 路径数据列表
            name: 对象名称
            color: 路径颜色
        """
        # 创建线段集合
        line_sets = []

        for i, path_data in enumerate(paths):
            points = path_data['points']

            if len(points) < 2:
                continue

            # 创建线段
            lines = []
            for j in range(len(points) - 1):
                lines.append([j, j + 1])

            # 创建LineSet
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color(color)
            line_sets.append(line_set)

        # 可视化
        if line_sets:
            o3d.visualization.draw_geometries(
                line_sets, 
                window_name=f"刀具路径可视化: {len(line_sets)} 条路径",
                width=1024,
                height=768,
                left=50,
                top=50
            )

        print(f"刀具路径可视化完成: {len(line_sets)} 条路径")

    def visualize_scalar_field(self, mesh: o3d.geometry.TriangleMesh,
                               scalar_field: np.ndarray,
                               name_suffix: str = "_scalar_field"):
        """
        可视化标量场
        Args:
            mesh: 网格对象
            scalar_field: 标量场数组
            name_suffix: 新对象名称后缀
        """
        # 创建网格副本
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        new_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
        new_mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
        new_mesh.triangle_normals = o3d.utility.Vector3dVector(np.asarray(mesh.triangle_normals))

        # 归一化标量场
        min_val, max_val = np.min(scalar_field), np.max(scalar_field)
        if max_val > min_val:
            normalized = (scalar_field - min_val) / (max_val - min_val)
        else:
            normalized = scalar_field

        # 使用热图颜色
        if MATPLOTLIB_AVAILABLE:
            colors = plt.cm.jet(normalized)[:, :3]  # 使用Jet颜色映射
        else:
            # 简单的热图颜色生成
            colors = np.zeros((len(normalized), 3))
            for i, val in enumerate(normalized):
                if val < 0.25:
                    colors[i] = [0, 0, 1 - 4 * val]
                elif val < 0.5:
                    colors[i] = [0, 4 * (val - 0.25), 0]
                elif val < 0.75:
                    colors[i] = [4 * (val - 0.5), 1, 0]
                else:
                    colors[i] = [1, 1 - 4 * (val - 0.75), 0]
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        # 可视化
        o3d.visualization.draw_geometries([new_mesh], 
                                         window_name="标量场可视化")

        print("标量场可视化完成")

    def visualize_edge_midpoints(self, mesh: o3d.geometry.TriangleMesh,
                               edge_midpoints: np.ndarray):
        """
        可视化边缘中点
        Args:
            mesh: 原始网格对象
            edge_midpoints: 中点边缘点数组
        """
        print(f"可视化边缘中点: {len(edge_midpoints)} 个点")
        
        # 创建点云对象
        midpoint_pcd = o3d.geometry.PointCloud()
        midpoint_pcd.points = o3d.utility.Vector3dVector(edge_midpoints)
        # 设置中点颜色为红色
        midpoint_pcd.paint_uniform_color([1, 0, 0])
        
        # 准备几何对象列表
        geometries = [mesh, midpoint_pcd]
        
        # 使用draw_geometries函数进行可视化
        try:
            o3d.visualization.draw_geometries(
                geometries,
                window_name=f"边缘中点可视化: {len(edge_midpoints)} 个点",
                width=1024,
                height=768
            )
        except Exception as e:
            print(f"可视化失败: {e}")
        
        print("边缘中点可视化完成")

    def create_animation(self, tool_paths: List[Dict[str, Any]],
                         frame_rate: int = 24,
                         name: str = "MachiningAnimation",
                         tool_radius: float = 0.5,
                         tool_length: float = 5.0,
                         show_paths: bool = True,
                         follow_tool: bool = False,
                         save_frames: bool = False):
        """
        创建加工动画（完整实现）
        
        支持完整的加工过程可视化，包括：
        - 真实的刀具几何形状
        - 路径轨迹显示
        - 视角跟随
        - 动画控制
        - 帧保存功能
        
        Args:
            tool_paths: 路径数据
            frame_rate: 帧率
            name: 动画名称
            tool_radius: 刀具半径
            tool_length: 刀具长度
            show_paths: 是否显示路径
            follow_tool: 是否跟随刀具
            save_frames: 是否保存帧
        """
        print("创建加工动画...")

        # 收集所有路径点和方向
        path_frames = []
        all_points = []
        all_orientations = []
        
        for path_idx, path_data in enumerate(tool_paths):
            points = path_data.get('points', [])
            orientations = path_data.get('orientations', [])
            cl_points = path_data.get('cl_points', points)
            
            if len(points) < 2:
                continue
            
            # 为每个路径点创建帧数据
            for i, (point, cl_point, orientation) in enumerate(zip(points, cl_points, orientations)):
                frame_data = {
                    'path_idx': path_idx,
                    'point_idx': i,
                    'cc_point': point,
                    'cl_point': cl_point,
                    'orientation': orientation,
                    'path_type': path_data.get('type', 'unknown')
                }
                path_frames.append(frame_data)
                all_points.append(point)
                all_orientations.append(orientation)

        if not path_frames:
            print("没有路径数据可供动画")
            return

        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=name, width=1024, height=768)

        # 创建路径可视化
        line_sets = []
        if show_paths:
            for path_data in tool_paths:
                points = path_data.get('points', [])
                if len(points) < 2:
                    continue
                
                # 创建线段
                lines = []
                for j in range(len(points) - 1):
                    lines.append([j, j + 1])
                
                # 创建LineSet
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                
                # 根据路径类型设置颜色
                path_type = path_data.get('type', 'unknown')
                if path_type == 'connection':
                    line_set.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色连接路径
                else:
                    line_set.paint_uniform_color([1.0, 0.0, 0.0])  # 红色加工路径
                
                line_sets.append(line_set)
                vis.add_geometry(line_set)

        # 创建刀具对象
        # 使用圆柱体表示刀具
        tool = o3d.geometry.TriangleMesh.create_cylinder(
            radius=tool_radius,
            height=tool_length
        )
        # 调整刀具位置，使刀尖在原点
        tool.translate([0, 0, -tool_length / 2])
        tool.paint_uniform_color([0.8, 0.2, 0.2])  # 红色刀具
        vis.add_geometry(tool)

        # 创建接触点标记
        contact_marker = o3d.geometry.TriangleMesh.create_sphere(radius=tool_radius * 0.5)
        contact_marker.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色接触点
        vis.add_geometry(contact_marker)

        # 播放动画
        total_frames = len(path_frames)
        print(f"动画总帧数: {total_frames}")
        
        for frame_idx, frame_data in enumerate(path_frames):
            cc_point = frame_data['cc_point']
            cl_point = frame_data['cl_point']
            orientation = frame_data['orientation']
            
            # 更新刀具位置和方向
            # 计算旋转矩阵
            direction = orientation / np.linalg.norm(orientation)
            up = np.array([0, 0, 1])
            
            if np.dot(direction, up) > 0.999:
                rotation_matrix = np.eye(3)
            elif np.dot(direction, up) < -0.999:
                rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            else:
                axis = np.cross(up, direction)
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.dot(up, direction))
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
            
            # 应用变换
            tool.transform(np.linalg.inv(tool.get_transform()))  # 重置变换
            tool.rotate(rotation_matrix)
            tool.translate(cl_point)
            
            # 更新接触点标记
            contact_marker.transform(np.linalg.inv(contact_marker.get_transform()))
            contact_marker.translate(cc_point)
            
            # 更新可视化
            vis.update_geometry(tool)
            vis.update_geometry(contact_marker)
            
            # 如果跟随刀具，调整视角
            if follow_tool and frame_idx % 5 == 0:  # 每5帧调整一次
                ctr = vis.get_view_control()
                ctr.set_lookat(cc_point)
            
            vis.poll_events()
            vis.update_renderer()
            
            # 保存帧
            if save_frames and frame_idx % 10 == 0:  # 每10帧保存一次
                vis.capture_screen_image(f"frame_{frame_idx:04d}.png")
            
            # 显示进度
            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"动画进度: {progress:.1f}%")

        vis.destroy_window()
        print(f"动画创建完成: {total_frames} 帧")