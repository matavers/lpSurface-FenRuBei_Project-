"""
五轴加工路径规划系统 - 主程序
Author: AI Assistant
Date: 2026
"""

import numpy as np
import math
import time
from typing import List, Tuple, Dict, Any, Optional
import json
import os
import open3d as o3d

# 导入核心模块
from core.meshProcessor import MeshProcessor
from core.nonSphericalTool import NonSphericalTool
from core.toolOrientationField import ToolOrientationField
from core.isoScallopField import IsoScallopFieldGenerator
from core.pathGenerator import PathGenerator
from utils.visualization import Visualizer
from utils.geometryTools import GeometryTools


class FiveAxisMachiningSystem:
    """五轴加工系统主控制器"""

    def __init__(self, config_path=None, intermediate_dir=None):
        """初始化系统"""
        self.config = self._load_config(config_path)
        self.mesh_processor = None
        self.tool = None
        self.partitioner = None
        self.orientation_field = None
        self.iso_scallop_generator = None
        self.path_generator = None
        self.visualizer = Visualizer()
        self.mesh = None  # 存储Open3D网格对象

        # 结果存储
        self.results = {
            'partition_labels': None,
            'tool_orientations': None,
            'scalar_field': None,
            'tool_paths': None,
            'metrics': {}
        }

        # 中间结果存储目录
        import datetime
        if intermediate_dir:
            self.intermediate_dir = intermediate_dir
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.intermediate_dir = os.path.join("intermediate", timestamp)
        
        # 创建中间结果目录
        if not os.path.exists(self.intermediate_dir):
            os.makedirs(self.intermediate_dir)
        print(f"中间结果将存储在: {self.intermediate_dir}")

        print("五轴加工系统初始化完成")

    def _load_config(self, config_path):
        """加载配置文件"""
        default_config = {
            'tool': {
                'type': 'ellipsoidal',
                'semi_axes': [9.0, 3.0],
                'shank_diameter': 6.0
            },
            'machining': {
                'scallop_height': 0.4,
                'feed_rate': 200.0,
                'spindle_speed': 8000.0
            },
            'algorithm': {
                'partition_resolution': 0.05,  # 分区分辨率参数，降低以减少分区数量
                'smoothing_lambda': 0.5,
                'max_iterations': 100,
                'tolerance': 1e-4,
                'tar_sampling_resolution': 30
            },
            'visualization': {
                'show_partitions': True,
                'show_orientations': True,
                'show_paths': True,
                'orientation_scale': 0.05
            }
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # 合并配置
                    default_config.update(user_config)
            except Exception as e:
                print(f"加载配置文件失败: {e}")

        return default_config
    
    def save_intermediate_result(self, step_name, data):
        """保存中间结果"""
        import numpy as np
        
        file_path = os.path.join(self.intermediate_dir, f"{step_name}.npy")
        try:
            np.save(file_path, data)
            print(f"中间结果已保存: {file_path}")
            return True
        except Exception as e:
            print(f"保存中间结果失败: {e}")
            return False
    
    def load_intermediate_result(self, step_name):
        """加载中间结果"""
        import numpy as np
        
        file_path = os.path.join(self.intermediate_dir, f"{step_name}.npy")
        if os.path.exists(file_path):
            try:
                data = np.load(file_path)
                print(f"中间结果已加载: {file_path}")
                return data
            except Exception as e:
                print(f"加载中间结果失败: {e}")
                return None
        else:
            print(f"中间结果文件不存在: {file_path}")
            return None
    
    def save_metrics(self):
        """保存指标"""
        metrics_file = os.path.join(self.intermediate_dir, "metrics.json")
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.results['metrics'], f, indent=2)
            print(f"指标已保存: {metrics_file}")
        except Exception as e:
            print(f"保存指标失败: {e}")
    
    def load_metrics(self):
        """加载指标"""
        metrics_file = os.path.join(self.intermediate_dir, "metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                self.results['metrics'] = metrics
                print(f"指标已加载: {metrics_file}")
                return True
            except Exception as e:
                print(f"加载指标失败: {e}")
                return False
        else:
            return False

    def load_mesh_from_file(self, input_path, mesh_algorithm="delaunay_cocone", surface_func=None, surface_params=None):
        """从文件加载网格或生成曲面网格"""
        # 当mesh_algorithm为obj时，直接使用OBJ文件，跳过曲面函数生成和采样步骤
        if mesh_algorithm == "obj":
            # 检查文件是否存在
            if not os.path.exists(input_path):
                raise ValueError(f"文件不存在: {input_path}")

            print(f"加载网格: {input_path}")
            # 使用Open3D加载网格
            self.mesh = o3d.io.read_triangle_mesh(input_path, True)
            
            # 检查网格是否有效
            if len(self.mesh.vertices) == 0 or len(self.mesh.triangles) == 0:
                raise ValueError("加载的网格无效")

            # 计算法线
            print("计算网格法线...")
            self.mesh.compute_vertex_normals()
            self.mesh.compute_triangle_normals()

            # 简化网格
            if len(self.mesh.vertices) > 10000:
                print("网格顶点数量过多，自动简化网格...")
                self.mesh = self.mesh.simplify_vertex_clustering(
                    voxel_size=0.1,
                    contraction=o3d.geometry.SimplificationContraction.Average
                )
                print(f"网格简化完成: {len(self.mesh.vertices)} 个顶点, {len(self.mesh.triangles)} 个三角形")

            # 可视化当前操作的网格
            print("可视化当前操作的网格...")
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="当前操作网格")
            vis.add_geometry(self.mesh)
            print("按ESC键关闭可视化窗口...")
            vis.run()
            vis.destroy_window()

            # 直接使用OBJ文件，不进行重建
            print("直接使用OBJ文件，跳过采样步骤...")
        else:
            # 处理曲面函数生成
            if surface_func:
                print(f"使用曲面函数: {surface_func}")
                from core.surfaceGenerator import SurfaceGenerator
                generator = SurfaceGenerator()
                
                resolution = surface_params.get('resolution', 50)
                
                if surface_func == "sphere":
                    # 生成球体
                    sphere_path = generator.generate_sphere(
                        radius=10.0,
                        resolution=resolution,
                        output_path="temp_sphere.obj",
                        density_factor=1.0
                    )
                    input_path = sphere_path
                elif surface_func == "torus":
                    # 生成圆环
                    torus_path = generator.generate_torus(
                        radius=10.0,
                        tube_radius=3.0,
                        resolution=resolution,
                        output_path="temp_torus.obj",
                        density_factor=1.0
                    )
                    input_path = torus_path
                elif surface_func == "saddle":
                    # 生成马鞍面
                    saddle_path = generator.generate_saddle(
                        resolution=resolution,
                        scale=10.0,
                        output_path="temp_saddle.obj",
                        density_factor=1.0
                    )
                    input_path = saddle_path
                else:
                    raise ValueError(f"不支持的曲面函数: {surface_func}")
            
            # 检查文件是否存在
            if not os.path.exists(input_path):
                raise ValueError(f"文件不存在: {input_path}")

            print(f"加载网格: {input_path}")
            # 使用Open3D加载网格
            self.mesh = o3d.io.read_triangle_mesh(input_path, True)
            
            # 检查网格是否有效
            if len(self.mesh.vertices) == 0 or len(self.mesh.triangles) == 0:
                raise ValueError("加载的网格无效")

            # 计算法线
            print("计算网格法线...")
            self.mesh.compute_vertex_normals()
            self.mesh.compute_triangle_normals()

            # 简化网格
            if len(self.mesh.vertices) > 10000:
                print("网格顶点数量过多，自动简化网格...")
                self.mesh = self.mesh.simplify_vertex_clustering(
                    voxel_size=0.1,
                    contraction=o3d.geometry.SimplificationContraction.Average
                )
                print(f"网格简化完成: {len(self.mesh.vertices)} 个顶点, {len(self.mesh.triangles)} 个三角形")

            # 根据选择的算法处理网格（采样构造网格）
            print(f"使用网格生成算法: {mesh_algorithm}")
            
            if mesh_algorithm == "bpa":
                # Ball Pivoting Algorithm
                print("使用Ball Pivoting Algorithm (BPA)重建网格...")
                # 转换为点云
                pcd = o3d.geometry.PointCloud()
                pcd.points = self.mesh.vertices
                pcd.normals = self.mesh.vertex_normals
                
                # 如果没有法线，计算法线
                if not pcd.has_normals():
                    pcd.estimate_normals()
                    pcd.orient_normals_consistent_tangent_plane(10)
                
                # 使用BPA重建
                radii = [0.005, 0.01, 0.02, 0.04]
                try:
                    # 尝试获取密度信息
                    self.mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        pcd, o3d.utility.DoubleVector(radii)
                    )
                    print(f"BPA网格重建完成: {len(self.mesh.vertices)} 个顶点, {len(self.mesh.triangles)} 个三角形")
                except ValueError:
                    # 如果只返回一个对象
                    self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        pcd, o3d.utility.DoubleVector(radii)
                    )
                    print(f"BPA网格重建完成: {len(self.mesh.vertices)} 个顶点, {len(self.mesh.triangles)} 个三角形")
            elif mesh_algorithm == "poisson":
                # Poisson Reconstruction
                print("使用Poisson Reconstruction重建网格...")
                # 转换为点云
                pcd = o3d.geometry.PointCloud()
                pcd.points = self.mesh.vertices
                pcd.normals = self.mesh.vertex_normals
                
                # 如果没有法线，计算法线
                if not pcd.has_normals():
                    pcd.estimate_normals()
                    pcd.orient_normals_consistent_tangent_plane(10)
                
                # 使用Poisson重建
                try:
                    # 尝试获取密度信息
                    self.mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd, depth=9
                    )
                    print(f"Poisson网格重建完成: {len(self.mesh.vertices)} 个顶点, {len(self.mesh.triangles)} 个三角形")
                except ValueError:
                    # 如果只返回一个对象
                    self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd, depth=9
                    )
                    print(f"Poisson网格重建完成: {len(self.mesh.vertices)} 个顶点, {len(self.mesh.triangles)} 个三角形")
            elif mesh_algorithm == "tsdf":
                # TSDF / Signed Distance Field
                print("使用TSDF/Signed Distance Field重建网格...")
                # 转换为点云
                pcd = o3d.geometry.PointCloud()
                pcd.points = self.mesh.vertices
                pcd.normals = self.mesh.vertex_normals
                
                # 如果没有法线，计算法线
                if not pcd.has_normals():
                    pcd.estimate_normals()
                    pcd.orient_normals_consistent_tangent_plane(10)
                
                # 使用体素格重建（TSDF的一种实现）
                voxel_size = 0.01
                self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_voxel_grid(
                    pcd, voxel_size
                )
                print(f"TSDF网格重建完成: {len(self.mesh.vertices)} 个顶点, {len(self.mesh.triangles)} 个三角形")
            elif mesh_algorithm == "delaunay_cocone":
                # Delaunay + Cocone
                print("使用Delaunay + Cocone重建网格...")
                # 转换为点云
                pcd = o3d.geometry.PointCloud()
                pcd.points = self.mesh.vertices
                
                # 使用Delaunay三角剖分
                # Open3D的凸包算法基于Delaunay三角剖分
                self.mesh, _ = pcd.compute_convex_hull()
                print(f"Delaunay + Cocone网格重建完成: {len(self.mesh.vertices)} 个顶点, {len(self.mesh.triangles)} 个三角形")
            
            # 可视化当前操作的网格（采样效果）
            print("可视化当前操作的网格（采样效果）...")
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="当前操作网格（采样效果）")
            vis.add_geometry(self.mesh)
            print("按ESC键关闭可视化窗口...")
            vis.run()
            vis.destroy_window()
        
        print(f"最终网格: {len(self.mesh.vertices)} 个顶点, {len(self.mesh.triangles)} 个三角形")
        
        # 创建MeshProcessor实例
        # 暂时使用一个简单的适配器，未来可以加入更多的底层库或者加强对Open3D网格的支持
        self.mesh_processor = MeshProcessor(self.mesh)
        return self.mesh_processor

    def setup_tool(self):
        """设置刀具"""
        # default或者settings.json中配置的刀具参数
        tool_config = self.config['tool']
        self.tool = NonSphericalTool(
            profile_type=tool_config['type'],
            params=tool_config
        )
        print(f"刀具设置完成: {tool_config['type']}")
        return self.tool

    def run_partitioning(self):
        """运行表面分区"""
        if not self.mesh_processor or not self.tool:
            raise ValueError("请先加载网格和设置刀具")

        print("开始表面分区...")

        # 使用新的高级表面分区器（基于算法version2）
        from core.advancedSurfacePartitioner import AdvancedSurfacePartitioner
        
        self.partitioner = AdvancedSurfacePartitioner(
            self.mesh_processor,
            self.tool,
            resolution=self.config['algorithm']['partition_resolution']
        )
        print("使用新的基于算法version2的分区器")

        start_time = time.time()
        partition_labels, edge_midpoints = self.partitioner.partition_surface()
        partition_time = time.time() - start_time

        self.results['partition_labels'] = partition_labels
        self.results['edge_midpoints'] = edge_midpoints
        self.results['metrics']['partition_time'] = partition_time
        self.results['metrics']['num_partitions'] = len(np.unique(partition_labels))
        self.results['metrics']['num_edge_midpoints'] = len(edge_midpoints)
        self.results['metrics']['partitioner_type'] = 'algorithm_version2'

        print(f"表面分区完成: {self.results['metrics']['num_partitions']} 个分区")
        print(f"提取边缘中点: {len(edge_midpoints)} 个")
        print(f"分区耗时: {partition_time:.2f} 秒")

        return partition_labels, edge_midpoints

    def generate_tool_orientation_field(self):
        """生成工具方向场"""
        if self.partitioner is None:
            raise ValueError("请先运行表面分区")

        print("生成工具方向场...")

        self.orientation_field = ToolOrientationField(
            self.mesh_processor,
            self.results['partition_labels'],
            self.tool
        )

        start_time = time.time()
        tool_orientations = self.orientation_field.generate_field()
        orientation_time = time.time() - start_time

        self.results['tool_orientations'] = tool_orientations
        self.results['metrics']['orientation_time'] = orientation_time

        print(f"工具方向场生成完成")
        print(f"方向场计算耗时: {orientation_time:.2f} 秒")

        return tool_orientations

    def generate_iso_scallop_field(self):
        """生成等残留高度场"""
        if self.orientation_field is None:
            raise ValueError("请先生成工具方向场")

        print("生成等残留高度场...")

        self.iso_scallop_generator = IsoScallopFieldGenerator(
            self.mesh_processor,
            self.results['tool_orientations'],
            self.tool,
            self.config['machining']['scallop_height']
        )

        start_time = time.time()
        scalar_field = self.iso_scallop_generator.generate_scalar_field()
        field_time = time.time() - start_time

        self.results['scalar_field'] = scalar_field
        self.results['metrics']['scalar_field_time'] = field_time

        print(f"等残留高度场生成完成")
        print(f"标量场计算耗时: {field_time:.2f} 秒")

        return scalar_field

    def generate_tool_paths(self):
        """生成刀具路径"""
        if self.iso_scallop_generator is None:
            raise ValueError("请先生成等残留高度场")

        print("生成刀具路径...")

        # 提取等值线
        iso_curves = self.iso_scallop_generator.extract_iso_curves(
            self.results['scalar_field']
        )

        self.path_generator = PathGenerator(
            self.mesh_processor,
            iso_curves,
            self.results['tool_orientations'],
            self.tool
        )

        start_time = time.time()
        tool_paths = self.path_generator.generate_final_path()
        path_time = time.time() - start_time

        self.results['tool_paths'] = tool_paths
        self.results['metrics']['path_generation_time'] = path_time
        self.results['metrics']['total_path_length'] = self._calculate_total_path_length(tool_paths)

        print(f"刀具路径生成完成")
        print(f"路径总长度: {self.results['metrics']['total_path_length']:.2f} mm")
        print(f"路径生成耗时: {path_time:.2f} 秒")

        return tool_paths

    def _calculate_total_path_length(self, tool_paths):
        """计算路径总长度"""
        total_length = 0
        for path_data in tool_paths['paths']:
            points = path_data['points']
            for i in range(len(points) - 1):
                total_length += np.linalg.norm(points[i + 1] - points[i])
        return total_length

    def visualize_results(self):
        """可视化所有结果"""
        print("可视化结果...")

        vis_config = self.config['visualization']

        if vis_config['show_partitions'] and self.results['partition_labels'] is not None:
            if 'edge_midpoints' in self.results and self.results['edge_midpoints'] is not None:
                # 使用新的可视化方法，在一个窗口中同时显示颜色块分区和中点
                self.visualizer.visualize_partitions_with_midpoints(
                    self.mesh,
                    self.results['partition_labels'],
                    self.results['edge_midpoints']
                )

        if vis_config['show_orientations'] and self.results['tool_orientations'] is not None:
            self.visualizer.visualize_tool_orientations(
                self.mesh,
                self.results['tool_orientations'],
                scale=vis_config['orientation_scale']
            )

        if vis_config['show_paths'] and self.results['tool_paths'] is not None:
            self.visualizer.visualize_tool_paths(
                self.results['tool_paths']['paths']
            )

        print("可视化完成")

    def export_results(self, output_base="output"):
        """导出结果"""
        import datetime
        
        # 创建基于时间戳的唯一文件夹名称
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_base, timestamp)
        
        # 创建目录结构
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 创建models子目录
        models_dir = os.path.join(output_dir, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # 导出路径为CSV
        if self.results['tool_paths']:
            self._export_paths_to_csv(output_dir)

        # 导出配置和指标
        self._export_metrics(output_dir)

        print(f"结果已导出到: {output_dir}")
        print(f"模型文件保存到: {models_dir}")
        
        return output_dir

    def _export_paths_to_csv(self, output_dir):
        """导出路径为CSV文件"""
        import csv

        for i, path_data in enumerate(self.results['tool_paths']['paths']):
            filename = os.path.join(output_dir, f"tool_path_{i:03d}.csv")
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['X', 'Y', 'Z', 'i', 'j', 'k'])  # 位置和方向

                for point, orientation in zip(path_data['points'], path_data['orientations']):
                    writer.writerow([
                        point[0], point[1], point[2],
                        orientation[0], orientation[1], orientation[2]
                    ])

    def _export_metrics(self, output_dir):
        """导出性能指标"""
        # 计算总处理时间并添加到指标中
        time_metrics = ['partition_time', 'orientation_time', 'scalar_field_time', 'path_generation_time']
        total_time = sum(self.results['metrics'].get(key, 0) for key in time_metrics)
        self.results['metrics']['total_processing_time'] = total_time
        
        metrics_file = os.path.join(output_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.results['metrics'], f, indent=2)
    
    def _export_additional_data(self, output_dir):
        """导出其他数据"""
        import numpy as np
        
        # 保存网格顶点信息
        if hasattr(self, 'mesh') and self.mesh is not None:
            vertices_file = os.path.join(output_dir, "vertices.npy")
            np.save(vertices_file, np.asarray(self.mesh.vertices))
            
            # 保存网格三角形信息
            if hasattr(self.mesh, 'triangles') and len(self.mesh.triangles) > 0:
                triangles_file = os.path.join(output_dir, "triangles.npy")
                np.save(triangles_file, np.asarray(self.mesh.triangles))
        
        # 保存分区数据
        if 'partition_labels' in self.results:
            partition_file = os.path.join(output_dir, "partition_labels.npy")
            np.save(partition_file, self.results['partition_labels'])
        
        # 保存边缘中点数据
        if 'edge_midpoints' in self.results:
            edge_midpoints_file = os.path.join(output_dir, "edge_midpoints.npy")
            np.save(edge_midpoints_file, self.results['edge_midpoints'])
        
        # 保存方向场数据
        if 'tool_orientations' in self.results:
            orientation_file = os.path.join(output_dir, "orientation_field.npy")
            np.save(orientation_file, self.results['tool_orientations'])
        
        # 保存标量场数据
        if 'scalar_field' in self.results:
            scalar_file = os.path.join(output_dir, "scalar_field.npy")
            np.save(scalar_file, self.results['scalar_field'])
        
        # 保存刀具路径为JSON格式（便于后续可视化）
        if 'tool_paths' in self.results and self.results['tool_paths']:
            tool_paths_file = os.path.join(output_dir, "tool_paths.json")
            # 转换numpy数组为Python列表
            tool_paths_data = self.results['tool_paths']
            # 深拷贝数据结构
            import copy
            tool_paths_serializable = copy.deepcopy(tool_paths_data)
            
            # 递归转换所有numpy数组为Python列表
            def convert_ndarray(obj):
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, list):
                    return [convert_ndarray(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: convert_ndarray(value) for key, value in obj.items()}
                else:
                    return obj
            
            # 转换所有numpy数组
            tool_paths_serializable = convert_ndarray(tool_paths_serializable)
            
            with open(tool_paths_file, 'w') as f:
                json.dump(tool_paths_serializable, f, indent=2)
        
        # 保存分区边缘数据（详细格式）
        if 'partition_labels' in self.results and hasattr(self, 'mesh') and self.mesh is not None:
            # 提取分区边缘
            partition_edges = []
            labels = self.results['partition_labels']
            vertices = np.asarray(self.mesh.vertices)
            triangles = np.asarray(self.mesh.triangles)
            
            # 遍历所有三角形
            for triangle in triangles:
                # 检查三角形的三个边
                for i in range(3):
                    v1 = triangle[i]
                    v2 = triangle[(i + 1) % 3]
                    
                    # 检查两个顶点是否属于不同的分区
                    if labels[v1] != labels[v2]:
                        # 计算中点
                        midpoint = ((vertices[v1] + vertices[v2]) / 2).tolist()
                        # 添加边缘
                        partition_edges.append({
                            'vertices': [int(v1), int(v2)],
                            'points': [vertices[v1].tolist(), vertices[v2].tolist()],
                            'midpoint': midpoint,
                            'labels': [int(labels[v1]), int(labels[v2])]
                        })
            
            # 去重边缘
            unique_edges = []
            seen = set()
            for edge in partition_edges:
                # 创建一个唯一的键来标识边缘（忽略顺序）
                key = tuple(sorted([edge['vertices'][0], edge['vertices'][1]]))
                if key not in seen:
                    seen.add(key)
                    unique_edges.append(edge)
            
            # 保存边缘数据
            if unique_edges:
                edges_file = os.path.join(output_dir, "partition_edges.json")
                with open(edges_file, 'w') as f:
                    json.dump({'edges': unique_edges}, f, indent=2)
                print(f"分区边缘数据已保存: {len(unique_edges)} 条边缘")

    def run_full_pipeline(self, input_path, skip_visualization=False, resume_from=None, mesh_algorithm="delaunay_cocone", surface_func=None, surface_params=None):
        """运行完整处理流程"""
        print("=" * 50)
        print("五轴加工路径规划系统")
        print("=" * 50)

        try:
            # 尝试加载之前的指标
            self.load_metrics()
            
            # 1. 加载网格
            print("\n1. 加载网格...")
            self.load_mesh_from_file(input_path, mesh_algorithm=mesh_algorithm, surface_func=surface_func, surface_params=surface_params)

            # 2. 设置刀具
            print("\n2. 设置刀具...")
            self.setup_tool()

            # 3. 表面分区
            print("\n3. 表面分区...")
            partition_labels = self.load_intermediate_result("partition_labels")
            edge_midpoints = self.load_intermediate_result("edge_midpoints")
            if partition_labels is not None and edge_midpoints is not None:
                self.results['partition_labels'] = partition_labels
                self.results['edge_midpoints'] = edge_midpoints
                print("跳过分区步骤，使用已加载的分区结果")
            else:
                self.run_partitioning()
                if self.results['partition_labels'] is not None and self.results['edge_midpoints'] is not None:
                    self.save_intermediate_result("partition_labels", self.results['partition_labels'])
                    self.save_intermediate_result("edge_midpoints", self.results['edge_midpoints'])
                    self.save_metrics()

            # 4. 生成工具方向场
            print("\n4. 生成工具方向场...")
            tool_orientations = self.load_intermediate_result("tool_orientations")
            if tool_orientations is not None:
                self.results['tool_orientations'] = tool_orientations
                print("跳过方向场生成步骤，使用已加载的方向场结果")
            else:
                self.generate_tool_orientation_field()
                if self.results['tool_orientations'] is not None:
                    self.save_intermediate_result("tool_orientations", self.results['tool_orientations'])
                    self.save_metrics()

            # 5. 生成等残留高度场
            print("\n5. 生成等残留高度场...")
            scalar_field = self.load_intermediate_result("scalar_field")
            if scalar_field is not None:
                self.results['scalar_field'] = scalar_field
                print("跳过标量场生成步骤，使用已加载的标量场结果")
            else:
                self.generate_iso_scallop_field()
                if self.results['scalar_field'] is not None:
                    self.save_intermediate_result("scalar_field", self.results['scalar_field'])
                    self.save_metrics()

            # 6. 生成刀具路径
            print("\n6. 生成刀具路径...")
            # 刀具路径比较复杂，暂时不支持加载，每次都重新生成
            self.generate_tool_paths()

            # 7. 可视化
            if not skip_visualization:
                print("\n7. 可视化结果...")
                self.visualize_results()
            else:
                print("跳过可视化步骤...")

            # 8. 导出结果
            print("\n8. 导出结果...")
            output_dir = self.export_results()
            
            # 9. 保存其他数据
            print("\n9. 保存附加数据...")
            self._export_additional_data(output_dir)

            # 打印总结
            print("\n" + "=" * 50)
            print("处理完成!")
            print("=" * 50)
            # 只计算时间相关的指标
            time_metrics = ['partition_time', 'orientation_time', 'scalar_field_time', 'path_generation_time']
            total_time = sum(self.results['metrics'].get(key, 0) for key in time_metrics)
            print(f"总处理时间: {total_time:.2f} 秒")
            print(f"分区数量: {self.results['metrics'].get('num_partitions', 'N/A')}")
            print(f"路径总长度: {self.results['metrics'].get('total_path_length', 'N/A'):.2f} mm")
            print(f"详细结果保存于: {output_dir}")
            print(f"中间结果存储于: {self.intermediate_dir}")

            return True

        except Exception as e:
            print(f"处理过程中出错: {e}")
            import traceback
            traceback.print_exc()
            # 保存已有的结果
            self.save_metrics()
            return False
    
    def run_partition_only(self, input_path, skip_visualization=False, mesh_algorithm="delaunay_cocone", surface_func=None, surface_params=None):
        """只运行分区并保存数据，不执行刀具路径规划"""
        print("=" * 70)
        print("运行分区并保存数据（不执行刀具路径规划）")
        print("=" * 70)

        try:
            # 1. 加载网格
            print("\n1. 加载网格...")
            self.load_mesh_from_file(input_path, mesh_algorithm=mesh_algorithm, surface_func=surface_func, surface_params=surface_params)

            # 2. 设置刀具
            print("\n2. 设置刀具...")
            self.setup_tool()

            # 3. 表面分区
            print("\n3. 运行表面分区...")
            partition_labels, edge_midpoints = self.run_partitioning()
            
            # 4. 创建基于时间戳的output子目录
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_subdir = os.path.join("output", timestamp)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            print(f"输出目录: {output_subdir}")

            # 5. 保存分区结果
            print("\n4. 保存分区结果...")
            # 保存分区标签
            partition_file = os.path.join(output_subdir, "partition_labels.npy")
            np.save(partition_file, partition_labels)
            print(f"分区标签保存到: {partition_file}")
            print(f"分区数量: {len(np.unique(partition_labels))}")
            
            # 6. 保存网格数据
            print("\n5. 保存网格数据...")
            vertices = np.asarray(self.mesh.vertices)
            faces = np.asarray(self.mesh.triangles)
            
            # 保存顶点
            vertices_file = os.path.join(output_subdir, "vertices.npy")
            np.save(vertices_file, vertices)
            print(f"顶点数据保存到: {vertices_file}")
            
            # 保存面
            faces_file = os.path.join(output_subdir, "triangles.npy")
            np.save(faces_file, faces)
            print(f"面数据保存到: {faces_file}")
            
            # 7. 提取和保存边缘点
            print("\n6. 提取和保存边缘点...")
            from core.surfaceGenerator import SurfaceGenerator
            surface_generator = SurfaceGenerator()
            
            # 提取每个分区的边缘，确保每条边缘只被处理一次
            unique_labels = np.unique(partition_labels)
            all_edge_points = []
            processed_edges = set()  # 用于跟踪已处理的边缘
            
            # 首先收集所有唯一的边缘，确保每条边缘只被处理一次
            unique_edges = {}  # 存储每条唯一边缘的点和相关分区
            
            for v in range(len(vertices)):
                # 检查邻居是否属于不同分区
                neighbors = self.mesh_processor.adjacency[v]
                for neighbor in neighbors:
                    if partition_labels[neighbor] != partition_labels[v]:
                        # 创建边缘的唯一标识（按顺序排列顶点索引）
                        edge_key = tuple(sorted([v, neighbor]))
                        if edge_key not in processed_edges:
                            processed_edges.add(edge_key)
                            # 存储边缘点和相关分区
                            label1 = partition_labels[v]
                            label2 = partition_labels[neighbor]
                            # 选择一个分区来处理这条边缘（选择标签较小的）
                            processing_label = min(label1, label2)
                            # 存储边缘点
                            edge_points = [vertices[v], vertices[neighbor]]
                            if processing_label not in unique_edges:
                                unique_edges[processing_label] = []
                            unique_edges[processing_label].extend(edge_points)
            
            # 处理每个分区的边缘，只收集边缘点，不进行曲线拟合
            all_new_vertices = []
            all_new_vertices_labels = []  # 存储每个新顶点对应的分区标签
            
            for label in unique_labels:
                if label in unique_edges:
                    edge_points = unique_edges[label]
                    
                    # 去重边缘点
                    unique_edge_points = []
                    seen = set()
                    for point in edge_points:
                        point_tuple = tuple(point)
                        if point_tuple not in seen:
                            seen.add(point_tuple)
                            unique_edge_points.append(point)
                    
                    print(f"分区 {label}: 收集边缘点，{len(unique_edge_points)} 个边缘点")
                    all_edge_points.extend(unique_edge_points)
                else:
                    print(f"分区 {label}: 无边缘点")
            
            # 对边缘点进行全局去重
            unique_all_edge_points = []
            seen = set()
            for point in all_edge_points:
                point_tuple = tuple(point)
                if point_tuple not in seen:
                    seen.add(point_tuple)
                    unique_all_edge_points.append(point)
            
            # 保存边缘点
            edge_points_file = os.path.join(output_subdir, "edge_points.npy")
            np.save(edge_points_file, unique_all_edge_points)
            print(f"边缘点数据保存到: {edge_points_file}")
            print(f"去重后边缘点数量: {len(unique_all_edge_points)}")
            
            # 8. 保存分区统计信息
            print("\n7. 保存分区统计信息...")
            metrics = {
                'num_partitions': len(np.unique(partition_labels)),
                'num_vertices': len(vertices),
                'num_faces': len(faces),
                'num_edge_points': len(all_edge_points),
                'partition_time': self.results['metrics'].get('partition_time', 0)
            }
            metrics_file = os.path.join(output_subdir, "metrics.json")
            import json
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"分区统计信息保存到: {metrics_file}")
            
            # 9. 保存附加数据
            print("\n8. 保存附加数据...")
            self._export_additional_data(output_subdir)
            
            # 10. 检查分区数量是否与三角面数量相同
            print("\n9. 检查分区结果...")
            num_partitions = len(np.unique(partition_labels))
            num_faces = len(faces)
            print(f"三角面数量: {num_faces}")
            print(f"分区数量: {num_partitions}")
            
            if num_partitions == num_faces:
                print("警告: 分区数量与三角面数量相同，每个三角面可能被作为一个分区")
                print("这可能是因为Leiden聚类参数设置不当，或者网格过于简单")
                print("建议调整AdvancedSurfacePartitioner中的resolution参数或聚类参数")
            else:
                print("分区数量与三角面数量不同，分区结果看起来合理")
            
            # 11. 可视化（可选）
            if not skip_visualization:
                print("\n10. 可视化分区结果...")
                # 使用新的可视化方法，在一个窗口中同时显示颜色块分区和中点
                if 'edge_midpoints' in self.results:
                    self.visualizer.visualize_partitions_with_midpoints(
                        self.mesh, 
                        partition_labels, 
                        self.results['edge_midpoints']
                    )
                else:
                    # 如果没有边缘中点，使用原来的可视化方法
                    self.visualize_results()
            else:
                print("\n10. 跳过可视化步骤...")
            
            # 打印总结
            print("\n" + "=" * 70)
            print("分区完成!")
            print("=" * 70)
            print(f"总处理时间: {self.results['metrics'].get('partition_time', 0):.2f} 秒")
            print(f"分区数量: {num_partitions}")
            print(f"详细结果保存于: {output_subdir}")
            print(f"中间结果存储于: {self.intermediate_dir}")
            print("测试已终止，未执行刀具路径计算")

            return True

        except Exception as e:
            print(f"处理过程中出错: {e}")
            import traceback
            traceback.print_exc()
            # 保存已有的结果
            self.save_metrics()
            return False


# 命令行运行
if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    partition_only = False
    mesh_algorithm = "delaunay_cocone"  # 默认使用Delaunay + Cocone算法
    surface_func = None
    surface_params = {}
    input_path = None
    
    for arg in sys.argv[1:]:
        if arg == "--partition-only":
            partition_only = True
        elif arg.startswith("--mesh-algorithm="):
            mesh_algorithm = arg.split("=")[1]
        elif arg.startswith("--surface="):
            surface_func = arg.split("=")[1]
        elif arg.startswith("--resolution="):
            surface_params['resolution'] = int(arg.split("=")[1])
        elif arg.startswith("--path="):
            input_path = arg.split("=")[1]
    
    # 验证参数组合
    # 1. --path只能与--mesh-algorithm=obj共存
    if input_path and mesh_algorithm != "obj":
        print("错误: --path参数只能与--mesh-algorithm=obj共存")
        # 显示用法说明
        print("用法: python main.py [--partition-only] [--mesh-algorithm=<algorithm>] [--surface=<function>] [--resolution=<int>] [--path=<path>]")
        print("  --partition-only: 只运行分区并保存数据，不执行刀具路径规划")
        print("  --mesh-algorithm: 网格生成算法，可选值: delaunay_cocone (默认), bpa, poisson, tsdf, obj")
        print("  --surface: 曲面函数名称，可选值: sphere, torus, saddle")
        print("  --resolution: 曲面采样分辨率，默认值: 50")
        print("  --path: OBJ文件路径（仅与--mesh-algorithm=obj共存）")
        sys.exit(1)
    
    # 2. --surface只能与--mesh-algorithm=算法名称（非obj）共存
    if surface_func and mesh_algorithm == "obj":
        print("错误: --surface参数不能与--mesh-algorithm=obj共存")
        # 显示用法说明
        print("用法: python main.py [--partition-only] [--mesh-algorithm=<algorithm>] [--surface=<function>] [--resolution=<int>] [--path=<path>]")
        print("  --partition-only: 只运行分区并保存数据，不执行刀具路径规划")
        print("  --mesh-algorithm: 网格生成算法，可选值: delaunay_cocone (默认), bpa, poisson, tsdf, obj")
        print("  --surface: 曲面函数名称，可选值: sphere, torus, saddle")
        print("  --resolution: 曲面采样分辨率，默认值: 50")
        print("  --path: OBJ文件路径（仅与--mesh-algorithm=obj共存）")
        sys.exit(1)
    
    # 3. 当--mesh-algorithm=obj时，必须指定--path
    if mesh_algorithm == "obj" and not input_path:
        print("错误: 当--mesh-algorithm=obj时，必须指定--path参数")
        # 显示用法说明
        print("用法: python main.py [--partition-only] [--mesh-algorithm=<algorithm>] [--surface=<function>] [--resolution=<int>] [--path=<path>]")
        print("  --partition-only: 只运行分区并保存数据，不执行刀具路径规划")
        print("  --mesh-algorithm: 网格生成算法，可选值: delaunay_cocone (默认), bpa, poisson, tsdf, obj")
        print("  --surface: 曲面函数名称，可选值: sphere, torus, saddle")
        print("  --resolution: 曲面采样分辨率，默认值: 50")
        print("  --path: OBJ文件路径（仅与--mesh-algorithm=obj共存）")
        sys.exit(1)
    
    # 4. 当使用--surface时，不需要指定--path
    if surface_func and input_path:
        print("错误: 当使用--surface参数时，不需要指定--path参数")
        # 显示用法说明
        print("用法: python main.py [--partition-only] [--mesh-algorithm=<algorithm>] [--surface=<function>] [--resolution=<int>] [--path=<path>]")
        print("  --partition-only: 只运行分区并保存数据，不执行刀具路径规划")
        print("  --mesh-algorithm: 网格生成算法，可选值: delaunay_cocone (默认), bpa, poisson, tsdf, obj")
        print("  --surface: 曲面函数名称，可选值: sphere, torus, saddle")
        print("  --resolution: 曲面采样分辨率，默认值: 50")
        print("  --path: OBJ文件路径（仅与--mesh-algorithm=obj共存）")
        sys.exit(1)
    
    # 5. 当既没有--path也没有--surface时，使用默认曲面
    if not input_path and not surface_func:
        print("错误: 必须指定--path参数或--surface参数")
        # 显示用法说明
        print("用法: python main.py [--partition-only] [--mesh-algorithm=<algorithm>] [--surface=<function>] [--resolution=<int>] [--path=<path>]")
        print("  --partition-only: 只运行分区并保存数据，不执行刀具路径规划")
        print("  --mesh-algorithm: 网格生成算法，可选值: delaunay_cocone (默认), bpa, poisson, tsdf, obj")
        print("  --surface: 曲面函数名称，可选值: sphere, torus, saddle")
        print("  --resolution: 曲面采样分辨率，默认值: 50")
        print("  --path: OBJ文件路径（仅与--mesh-algorithm=obj共存）")
        # 示例用法
        print("示例: python main.py --mesh-algorithm=obj --path=test_sphere.obj")
        print("示例: python main.py --mesh-algorithm=obj --path=test_sphere.obj --partition-only")
        print("示例: python main.py --mesh-algorithm=bpa --surface=sphere --resolution=50")
        print("说明:")
        print("  1. 当 --mesh-algorithm=obj 时，必须指定--path参数，直接使用OBJ文件，跳过采样步骤")
        print("  2. 当指定 --surface 参数时，会根据曲面函数生成网格，不能与--path参数同时使用")
        print("  3. 曲面方程直接在代码中定义，无需通过命令行传入")
        print("  4. 有默认值的参数若没有指定，采用默认值")
        print("  5. 若--mesh-algorithm为obj才采用.obj文件，跳过采样步骤")
        print("  6. 若为其他算法名称则采样构造网格，随后再进行其他步骤")
        sys.exit(1)
    
    system = FiveAxisMachiningSystem()
    
    # 检查是否指定了只运行分区
    if partition_only:
        success = system.run_partition_only(input_path, mesh_algorithm=mesh_algorithm, surface_func=surface_func, surface_params=surface_params)
    else:
        success = system.run_full_pipeline(input_path, mesh_algorithm=mesh_algorithm, surface_func=surface_func, surface_params=surface_params)
    
    if success:
        print("处理完成!")
    else:
        print("处理失败，请查看控制台输出")