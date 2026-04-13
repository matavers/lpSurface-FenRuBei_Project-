#!/usr/bin/env python3
"""
可视化脚本 - 选择output子目录并自动可视化数据

这个脚本提供了一个类似软件的操作界面，允许用户：
1. 浏览output目录下的所有计算结果文件夹
2. 选择特定的结果文件夹进行可视化
3. 自动加载并显示分区、方向场、刀具路径等数据
"""

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import open3d as o3d
from tkinter import Tk, filedialog, Label, Button, Listbox, Scrollbar, Frame

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class ResultVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("Five-Axis Machining Path Planning - Result Visualisation")
        self.master.geometry("900x600")
        
        # 文件夹类型
        self.folder_type = "output"  # 默认选择output文件夹
        self.folder_bases = {
            "output": "output",
            "intermediate": "intermediate"
        }
        
        # 控制是否跳过点云可视化，默认为False
        self.skip_point_cloud = True
        
        # 创建GUI组件
        self.create_widgets()
        
        # 加载结果文件夹列表
        self.load_result_folders()
    
    def create_widgets(self):
        """创建GUI组件"""
        # 标题
        title_label = Label(self.master, text="五轴加工路径规划结果可视化", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # 文件夹类型选择框架
        type_frame = Frame(self.master)
        type_frame.pack(pady=5)
        
        type_label = Label(type_frame, text="选择文件夹类型:", font=("Arial", 12))
        type_label.pack(side="left", padx=10)
        
        # 输出文件夹按钮
        output_button = Button(type_frame, text="输出文件夹 (output)", 
                              command=lambda: self.switch_folder_type("output"),
                              width=20, relief="groove", borderwidth=2)
        output_button.pack(side="left", padx=5)
        
        # 中间文件夹按钮
        intermediate_button = Button(type_frame, text="中间文件夹 (intermediate)", 
                                    command=lambda: self.switch_folder_type("intermediate"),
                                    width=20, relief="flat", borderwidth=2)
        intermediate_button.pack(side="left", padx=5)
        
        # 存储按钮引用
        self.type_buttons = {
            "output": output_button,
            "intermediate": intermediate_button
        }
        
        # 更新按钮状态
        self.update_type_buttons()
        
        # 文件夹列表框架
        list_frame = Frame(self.master)
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # 滚动条
        scrollbar = Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        # 文件夹列表
        self.folder_list = Listbox(list_frame, yscrollcommand=scrollbar.set, width=90, height=20)
        self.folder_list.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.folder_list.yview)
        
        # 按钮框架
        button_frame = Frame(self.master)
        button_frame.pack(pady=10)
        
        # 可视化按钮
        visualize_button = Button(button_frame, text="可视化选中结果", command=self.visualize_selected, width=20)
        visualize_button.pack(side="left", padx=10)
        
        # 刷新按钮
        refresh_button = Button(button_frame, text="刷新列表", command=self.load_result_folders, width=15)
        refresh_button.pack(side="left", padx=10)
        
        # 跳过点云可视化复选框
        from tkinter import Checkbutton, IntVar
        self.skip_point_cloud_var = IntVar()
        self.skip_point_cloud_var.set(0)  # 默认不跳过
        skip_point_cloud_check = Checkbutton(button_frame, text="跳过点云可视化", variable=self.skip_point_cloud_var, 
                                           command=lambda: setattr(self, 'skip_point_cloud', bool(self.skip_point_cloud_var.get())))
        skip_point_cloud_check.pack(side="left", padx=10)
        
        # 退出按钮
        exit_button = Button(button_frame, text="退出", command=self.master.quit, width=10)
        exit_button.pack(side="left", padx=10)
        
        # 状态标签
        self.status_label = Label(self.master, text="就绪", font=("Arial", 10))
        self.status_label.pack(pady=5)
    
    def switch_folder_type(self, folder_type):
        """切换文件夹类型"""
        if folder_type in self.folder_bases:
            self.folder_type = folder_type
            self.update_type_buttons()
            self.load_result_folders()
            self.status_label.config(text=f"已切换到 {folder_type} 文件夹类型")
    
    def update_type_buttons(self):
        """更新文件夹类型按钮的状态"""
        for type_name, button in self.type_buttons.items():
            if type_name == self.folder_type:
                button.config(relief="groove", borderwidth=2, state="disabled")
            else:
                button.config(relief="flat", borderwidth=2, state="normal")
    
    def load_result_folders(self):
        """加载结果文件夹列表"""
        self.folder_list.delete(0, "end")
        
        current_base = self.folder_bases[self.folder_type]
        
        if not os.path.exists(current_base):
            self.status_label.config(text=f"错误: {current_base} 目录不存在")
            return
        
        # 获取所有子目录
        try:
            folders = [f for f in os.listdir(current_base) 
                      if os.path.isdir(os.path.join(current_base, f))]
            
            if not folders:
                self.status_label.config(text=f"没有找到 {self.folder_type} 文件夹")
                return
            
            # 按时间戳排序
            folders.sort(reverse=True)
            
            # 添加到列表
            for folder in folders:
                # 添加文件夹类型标识
                self.folder_list.insert("end", f"[{self.folder_type}] {folder}")
            
            self.status_label.config(text=f"找到 {len(folders)} 个 {self.folder_type} 文件夹")
        except Exception as e:
            self.status_label.config(text=f"加载文件夹时出错: {e}")
    
    def visualize_selected(self):
        """可视化选中的结果文件夹"""
        selected_indices = self.folder_list.curselection()
        if not selected_indices:
            self.status_label.config(text="请先选择一个结果文件夹")
            return
        
        selected_item = self.folder_list.get(selected_indices[0])
        
        # 解析文件夹类型和实际文件夹名
        if selected_item.startswith("[output]"):
            folder_type = "output"
            actual_folder = selected_item[8:].strip()
        elif selected_item.startswith("[intermediate]"):
            folder_type = "intermediate"
            actual_folder = selected_item[14:].strip()
        else:
            # 兼容旧格式
            folder_type = self.folder_type
            actual_folder = selected_item
        
        base_path = self.folder_bases[folder_type]
        result_path = os.path.join(base_path, actual_folder)
        
        try:
            self.status_label.config(text=f"正在可视化: {actual_folder}")
            
            # 根据文件夹类型加载并可视化数据
            if folder_type == "output":
                # 从output文件夹加载完整结果
                self.load_and_visualize(result_path)
            elif folder_type == "intermediate":
                # 从intermediate文件夹加载中间结果
                self.load_and_visualize_intermediate(result_path)
            
            self.status_label.config(text=f"可视化完成: {actual_folder}")
        except Exception as e:
            self.status_label.config(text=f"可视化时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def load_and_visualize(self, result_path):
        """加载并可视化数据"""
        print(f"加载结果路径: {result_path}")
        
        # 加载网格数据
        vertices, triangles = self.load_mesh_data(result_path)
        
        # 1. 加载指标
        metrics_file = os.path.join(result_path, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            print("\n=== 加工指标 ===")
            for key, value in metrics.items():
                print(f"{key}: {value}")
        
        # 2. 可视化刀具路径
        tool_paths_file = os.path.join(result_path, "tool_paths.json")
        if os.path.exists(tool_paths_file):
            try:
                with open(tool_paths_file, 'r') as f:
                    tool_paths = json.load(f)
                self.visualize_tool_paths(tool_paths)
            except json.JSONDecodeError as e:
                print(f"解析tool_paths.json时出错: {e}")
                print("尝试使用CSV文件可视化刀具路径...")
                # 尝试从CSV文件加载路径
                self._visualize_tool_paths_from_csv(result_path)
        
        # 3. 可视化分区数据
        partition_file = os.path.join(result_path, "partition_labels.npy")
        if os.path.exists(partition_file):
            partition_labels = np.load(partition_file)
            print(f"\n分区数量: {len(np.unique(partition_labels))}")
            self.visualize_partitions(partition_labels, vertices, triangles)
        
        # 4. 可视化边缘拟合点
        self.visualize_edge_fitting_points(result_path)
        
        # 5. 可视化分区边缘数据
        self.visualize_partition_edges(result_path)
        
        # 6. 可视化方向场
        orientation_file = os.path.join(result_path, "orientation_field.npy")
        if os.path.exists(orientation_file):
            try:
                orientation_field = np.load(orientation_file, allow_pickle=True)
                print(f"方向场形状: {orientation_field.shape}")
                self.visualize_tool_orientations(orientation_field, vertices)
            except Exception as e:
                print(f"加载方向场时出错: {e}")
                print("跳过方向场可视化")
        
        # 7. 可视化标量场
        scalar_file = os.path.join(result_path, "scalar_field.npy")
        if os.path.exists(scalar_file):
            try:
                scalar_field = np.load(scalar_file, allow_pickle=True)
                print(f"标量场形状: {scalar_field.shape}")
                self.visualize_scalar_field(scalar_field)
            except Exception as e:
                print(f"加载标量场时出错: {e}")
                print("跳过标量场可视化")
    
    def load_and_visualize_intermediate(self, intermediate_path):
        """加载并可视化中间结果"""
        print(f"加载中间结果路径: {intermediate_path}")
        
        # 加载网格数据（如果存在）
        vertices, triangles = self.load_mesh_data(intermediate_path)
        
        # 加载指标
        metrics_file = os.path.join(intermediate_path, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            print("\n=== 加工指标 ===")
            for key, value in metrics.items():
                print(f"{key}: {value}")
        
        # 加载并可视化分区数据
        partition_file = os.path.join(intermediate_path, "partition_labels.npy")
        if os.path.exists(partition_file):
            partition_labels = np.load(partition_file)
            print(f"\n分区数量: {len(np.unique(partition_labels))}")
            self.visualize_partitions(partition_labels, vertices, triangles)
        
        # 加载并可视化方向场数据
        orientation_file = os.path.join(intermediate_path, "tool_orientations.npy")
        if os.path.exists(orientation_file):
            orientation_field = np.load(orientation_file)
            print(f"方向场形状: {orientation_field.shape}")
            self.visualize_tool_orientations(orientation_field, vertices)
        
        # 加载并可视化标量场数据
        scalar_file = os.path.join(intermediate_path, "scalar_field.npy")
        if os.path.exists(scalar_file):
            scalar_field = np.load(scalar_file)
            print(f"标量场形状: {scalar_field.shape}")
            self.visualize_scalar_field(scalar_field)
    
    def visualize_tool_paths(self, tool_paths):
        """可视化刀具路径"""
        print("\n=== 可视化刀具路径 ===")
        print(f"路径数量: {len(tool_paths['paths'])}")
        
        # 创建Open3D可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="刀具路径可视化")
        
        # 添加坐标系
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(coord_frame)
        
        # 加载所有路径
        for i, path_data in enumerate(tool_paths['paths']):
            points = path_data['points']
            
            # 创建点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 为不同路径设置不同颜色
            color = [(i % 10) / 10.0, (i // 10) / 10.0, 0.5]
            pcd.paint_uniform_color(color)
            vis.add_geometry(pcd)
            
            # 创建线条
            if len(points) > 1:
                lines = []
                for j in range(len(points) - 1):
                    lines.append([j, j + 1])
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.paint_uniform_color(color)
                vis.addwwa_geometry(line_set)
        
        # 设置视角
        vis.run()
        vis.destroy_window()
    
    def visualize_scalar_field(self, scalar_field):
        """可视化标量场"""
        print("\n=== 可视化标量场 ===")
        print(f"标量场形状: {scalar_field.shape}")
        
        # 检查标量场是否为空
        if not scalar_field.shape:
            print("标量场为空，跳过可视化")
            return
        
        try:
            # 打印标量场的统计信息
            print(f"标量场最小值: {np.min(scalar_field)}")
            print(f"标量场最大值: {np.max(scalar_field)}")
            print(f"标量场平均值: {np.mean(scalar_field)}")
            print(f"标量场标准差: {np.std(scalar_field)}")
            
            # 根据标量场形状选择可视化方式
            if len(scalar_field.shape) == 1:
                # 一维标量场 - 绘制折线图和直方图
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                
                # 折线图
                ax1.plot(scalar_field, 'b-', linewidth=0.5)
                ax1.set_title('等残留高度场 (折线图)')
                ax1.set_xlabel('顶点索引')
                ax1.set_ylabel('标量值')
                ax1.grid(True)
                
                # 设置y轴范围，确保数据可见
                y_min = np.min(scalar_field)
                y_max = np.max(scalar_field)
                if y_max > y_min:
                    ax1.set_ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)
                
                # 直方图
                # 计算合适的bins数量
                scalar_range = np.max(scalar_field) - np.min(scalar_field)
                if scalar_range > 0:
                    max_bins = min(50, int(scalar_range * 1000))
                    max_bins = max(5, max_bins)  # 确保至少有5个bins
                    ax2.hist(scalar_field, bins=max_bins, alpha=0.7, color='blue')
                else:
                    # 所有值都相同，使用单个bin
                    ax2.hist(scalar_field, bins=5, alpha=0.7, color='blue')
                
                ax2.set_title('等残留高度场 (直方图)')
                ax2.set_xlabel('标量值')
                ax2.set_ylabel('频率')
                ax2.grid(True)
            else:
                # 多维标量场 - 使用imshow
                plt.figure(figsize=(10, 8))
                plt.imshow(scalar_field, cmap='viridis')
                plt.colorbar(label='标量值')
                plt.title('等残留高度场')
                plt.xlabel('X')
                plt.ylabel('Y')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"可视化标量场时出错: {e}")
            print("跳过标量场可视化")
    
    def load_mesh_data(self, result_path):
        """加载网格数据"""
        vertices_file = os.path.join(result_path, "vertices.npy")
        triangles_file = os.path.join(result_path, "triangles.npy")
        
        vertices = None
        triangles = None
        
        if os.path.exists(vertices_file):
            vertices = np.load(vertices_file)
            print(f"加载网格顶点: {vertices.shape[0]} 个顶点")
        
        if os.path.exists(triangles_file):
            triangles = np.load(triangles_file)
            print(f"加载网格三角形: {triangles.shape[0]} 个三角形")
        
        return vertices, triangles
    
    def load_intermediate_data(self, intermediate_path):
        """从中间数据目录加载数据"""
        print(f"从中间数据目录加载数据: {intermediate_path}")
        
        # 加载网格数据（如果存在）
        vertices = None
        triangles = None
        
        # 加载分区数据
        partition_file = os.path.join(intermediate_path, "partition_labels.npy")
        partition_labels = None
        if os.path.exists(partition_file):
            partition_labels = np.load(partition_file)
            print(f"加载分区数据: {partition_labels.shape[0]} 个顶点")
        
        # 加载方向场数据
        orientation_file = os.path.join(intermediate_path, "tool_orientations.npy")
        orientation_field = None
        if os.path.exists(orientation_file):
            orientation_field = np.load(orientation_file)
            print(f"加载方向场数据: {orientation_field.shape}")
        
        # 加载标量场数据
        scalar_file = os.path.join(intermediate_path, "scalar_field.npy")
        scalar_field = None
        if os.path.exists(scalar_file):
            scalar_field = np.load(scalar_file)
            print(f"加载标量场数据: {scalar_field.shape}")
        
        return partition_labels, orientation_field, scalar_field
    
    def visualize_partitions(self, partition_labels, vertices=None, triangles=None):
        """可视化分区数据"""
        print("\n=== 可视化分区信息 ===")
        print(f"分区数量: {len(np.unique(partition_labels))}")
        
        # 3D可视化分区
        if vertices is not None:
            # 传统的填充颜色方式
            self.visualize_partitions_3d(partition_labels, vertices, triangles)
            # 点云方式，根据skip_point_cloud变量决定是否跳过
            if not self.skip_point_cloud:
                self.visualize_partitions_point_cloud(partition_labels, vertices)
            else:
                print("跳过点云可视化")
        else:
            print("警告: 缺少网格顶点数据，无法进行3D分区可视化")
    
    def visualize_partitions_3d(self, partition_labels, vertices, triangles=None):
        """3D可视化分区"""
        print("\n=== 3D可视化分区 ===")
        
        try:
            # 创建Open3D网格对象
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            
            if triangles is not None:
                # 确保三角形索引是整数类型
                triangles = triangles.astype(int)
                mesh.triangles = o3d.utility.Vector3iVector(triangles)
                print(f"添加 {triangles.shape[0]} 个三角形到网格")
            
            # 为每个顶点分配颜色
            unique_labels = np.unique(partition_labels)
            num_labels = len(unique_labels)
            
            print(f"创建 {num_labels} 个分区的颜色映射") 
            
            # 创建颜色映射
            colors = plt.cm.jet(np.linspace(0, 1, num_labels))[:, :3]
            label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
            
            # 为每个顶点设置颜色
            vertex_colors = np.array([label_to_color[label] for label in partition_labels])
            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            print(f"为 {vertex_colors.shape[0]} 个顶点分配颜色")
            
            # 计算法线
            mesh.compute_vertex_normals()
            print("计算网格法线完成")
            
            # 可视化
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="3D分区可视化")
            vis.add_geometry(mesh)
            
            # 添加坐标系
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
            vis.add_geometry(coord_frame)
            
            # 计算并添加中间边缘点
            print("计算中间边缘点...")
            edge_midpoints = []
            # 构建邻接表
            adjacency = [[] for _ in range(len(vertices))]
            if triangles is not None:
                for triangle in triangles:
                    v0, v1, v2 = triangle
                    adjacency[v0].append(v1)
                    adjacency[v1].append(v0)
                    adjacency[v1].append(v2)
                    adjacency[v2].append(v1)
                    adjacency[v2].append(v0)
                    adjacency[v0].append(v2)
            
            # 去重邻接表
            for i in range(len(adjacency)):
                adjacency[i] = list(set(adjacency[i]))
            
            # 计算中间边缘点
            processed_edges = set()
            for v in range(len(vertices)):
                for neighbor in adjacency[v]:
                    if partition_labels[neighbor] != partition_labels[v]:
                        edge_key = tuple(sorted([v, neighbor]))
                        if edge_key not in processed_edges:
                            processed_edges.add(edge_key)
                            # 计算中点
                            midpoint = (vertices[v] + vertices[neighbor]) / 2
                            edge_midpoints.append(midpoint)
            
            # 添加中间边缘点
            if edge_midpoints:
                print(f"添加 {len(edge_midpoints)} 个中间边缘点")
                midpoint_pcd = o3d.geometry.PointCloud()
                midpoint_pcd.points = o3d.utility.Vector3dVector(edge_midpoints)
                midpoint_pcd.paint_uniform_color([1, 0, 0])  # 红色
                vis.add_geometry(midpoint_pcd)
            
            # 设置视角
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            
            print("启动3D分区可视化窗口...")
            vis.run()
            vis.destroy_window()
            print("3D分区可视化完成")
        except Exception as e:
            print(f"3D分区可视化出错: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize_partitions_point_cloud(self, partition_labels, vertices):
        """以点云方式可视化分区"""
        print("\n=== 点云方式可视化分区 ===")
        
        try:
            # 创建Open3D点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            
            # 为每个顶点分配颜色
            unique_labels = np.unique(partition_labels)
            num_labels = len(unique_labels)
            
            print(f"创建 {num_labels} 个分区的颜色映射")
            
            # 创建颜色映射
            colors = plt.cm.jet(np.linspace(0, 1, num_labels))[:, :3]
            label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
            
            # 为每个顶点设置颜色
            vertex_colors = np.array([label_to_color[label] for label in partition_labels])
            pcd.colors = o3d.utility.Vector3dVector(vertex_colors)
            print(f"为 {vertex_colors.shape[0]} 个顶点分配颜色")
            
            # 可视化
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="点云分区可视化")
            vis.add_geometry(pcd)
            
            # 添加坐标系
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
            vis.add_geometry(coord_frame)
            
            # 计算并添加中间边缘点（跳过耗时的邻接表计算）
            print("跳过中间边缘点计算，以提高性能...")
            
            # 设置视角
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            
            print("启动点云分区可视化窗口...")
            vis.run()
            vis.destroy_window()
            print("点云分区可视化完成")
        except Exception as e:
            print(f"点云分区可视化出错: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize_tool_orientations(self, orientation_field, vertices=None):
        """可视化刀具方向场"""
        print("\n=== 可视化刀具方向场 ===")
        print(f"方向场形状: {orientation_field.shape}")
        
        # 检查方向场是否为空
        if not orientation_field.shape:
            print("方向场为空，跳过可视化")
            return
        
        # 顶点采样，避免数据量过大
        num_vertices = orientation_field.shape[0]
        sample_size = min(1000, num_vertices)
        if num_vertices > sample_size:
            sample_indices = np.random.choice(num_vertices, sample_size, replace=False)
            sampled_orientations = orientation_field[sample_indices]
        else:
            sampled_orientations = orientation_field
        
        try:
            # 计算方向向量的长度和角度
            lengths = np.linalg.norm(sampled_orientations, axis=1)
            angles_xy = np.arctan2(sampled_orientations[:, 1], sampled_orientations[:, 0])
            angles_xz = np.arctan2(sampled_orientations[:, 2], sampled_orientations[:, 0])
            
            # 绘制方向向量长度分布
            plt.figure(figsize=(10, 6))
            
            # 计算合适的bins数量
            lengths_range = np.max(lengths) - np.min(lengths)
            if lengths_range > 0:
                # 根据数据范围自适应计算bins数量
                max_bins = min(50, int(lengths_range * 100))
                max_bins = max(5, max_bins)  # 确保至少有5个bins
                plt.hist(lengths, bins=max_bins, alpha=0.7, color='green')
            else:
                # 所有值都相同，使用单个bin
                plt.hist(lengths, bins=5, alpha=0.7, color='green')
            
            plt.title('方向向量长度分布')
            plt.xlabel('向量长度')
            plt.ylabel('频率')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # 绘制方向向量角度分布
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 角度分布使用固定的bins数量
            ax1.hist(angles_xy, bins=30, alpha=0.7, color='purple')
            ax1.set_title('XY平面方向角分布')
            ax1.set_xlabel('角度 (弧度)')
            ax1.set_ylabel('频率')
            ax1.grid(True, alpha=0.3)
            
            ax2.hist(angles_xz, bins=30, alpha=0.7, color='orange')
            ax2.set_title('XZ平面方向角分布')
            ax2.set_xlabel('角度 (弧度)')
            ax2.set_ylabel('频率')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"绘制方向场统计图表时出错: {e}")
            print("跳过统计图表绘制，直接进行3D可视化...")
        
        # 3D可视化方向场
        if vertices is not None:
            self.visualize_tool_orientations_3d(orientation_field, vertices)
    
    def visualize_tool_orientations_3d(self, orientation_field, vertices):
        """3D可视化刀具方向场"""
        print("\n=== 3D可视化刀具方向场 ===")
        
        try:
            # 顶点采样，避免数据量过大
            num_vertices = vertices.shape[0]
            sample_size = min(1000, num_vertices)
            if num_vertices > sample_size:
                sample_indices = np.random.choice(num_vertices, sample_size, replace=False)
                sampled_vertices = vertices[sample_indices]
                sampled_orientations = orientation_field[sample_indices]
            else:
                sampled_vertices = vertices
                sampled_orientations = orientation_field
            
            print(f"使用 {sample_size} 个采样顶点进行方向场可视化")
            
            # 创建Open3D可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="3D刀具方向场可视化")
            
            # 添加坐标系
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            vis.add_geometry(coord_frame)
            
            # 创建箭头列表
            arrows = []
            scale = 0.1  # 箭头缩放
            
            print("创建箭头...")
            for i, (vertex, orientation) in enumerate(zip(sampled_vertices, sampled_orientations)):
                if np.linalg.norm(orientation) < 0.1:
                    continue
                
                # 计算箭头终点
                end_point = vertex + orientation * scale
                
                # 创建箭头
                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=scale * 0.2,
                    cone_radius=scale * 0.4,
                    cylinder_height=scale * 0.6,
                    cone_height=scale * 0.4
                )
                
                # 计算旋转矩阵
                direction = orientation / np.linalg.norm(orientation)
                up = np.array([0, 0, 1])
                
                if np.dot(direction, up) > 0.99:
                    rotation_matrix = np.eye(3)
                elif np.dot(direction, up) < -0.99:
                    rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                else:
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
            
            print(f"创建完成: {len(arrows)} 个箭头")
            
            # 添加所有箭头
            for arrow in arrows:
                vis.add_geometry(arrow)
            
            print("添加箭头到可视化窗口...")
            
            # 设置视角
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            
            print("启动方向场可视化窗口...")
            vis.run()
            vis.destroy_window()
            print(f"方向场可视化完成: {len(arrows)} 个箭头")
        except Exception as e:
            print(f"方向场可视化出错: {e}")
            import traceback
            traceback.print_exc()
    
    def visualize_edge_fitting_points(self, result_path):
        """可视化边缘拟合点"""
        print("\n=== 可视化边缘拟合点 ===")
        
        # 加载边缘点
        edge_points_file = os.path.join(result_path, "edge_points.npy")
        projected_points_file = os.path.join(result_path, "projected_points.npy")
        partition_file = os.path.join(result_path, "partition_labels.npy")
        
        edge_points = None
        projected_points = None
        partition_labels = None
        
        if os.path.exists(edge_points_file):
            edge_points = np.load(edge_points_file)
            print(f"加载边缘点: {edge_points.shape[0]} 个点")
        
        if os.path.exists(projected_points_file):
            projected_points = np.load(projected_points_file)
            print(f"加载投影点: {projected_points.shape[0]} 个点")
        
        if os.path.exists(partition_file):
            partition_labels = np.load(partition_file)
            print(f"加载分区标签: {partition_labels.shape[0]} 个标签")
        
        if edge_points is None and projected_points is None:
            print("未找到边缘拟合点数据")
            return
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="边缘拟合点可视化")
        
        # 添加坐标系
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(coord_frame)
        
        # 不显示原始边缘点，只显示中间边缘点
        # if edge_points is not None:
        #     edge_pcd = o3d.geometry.PointCloud()
        #     edge_pcd.points = o3d.utility.Vector3dVector(edge_points)
        #     edge_pcd.paint_uniform_color([0, 0, 1])  # 蓝色
        #     vis.add_geometry(edge_pcd)
        
        # 显示投影点（拟合后的点，根据分区显示不同颜色）
        if projected_points is not None:
            # 尝试加载新顶点标签文件
            new_vertices_labels_file = os.path.join(result_path, "new_vertices_labels.npy")
            new_vertices_labels = None
            
            if os.path.exists(new_vertices_labels_file):
                new_vertices_labels = np.load(new_vertices_labels_file)
                print(f"加载新顶点标签: {new_vertices_labels.shape[0]} 个标签")
            
            if new_vertices_labels is not None and len(new_vertices_labels) == len(projected_points):
                # 使用新顶点标签为拟合点设置不同颜色
                unique_labels = np.unique(new_vertices_labels)
                num_labels = len(unique_labels)
                print(f"为 {num_labels} 个分区创建颜色映射")
                
                # 创建颜色映射
                colors = plt.cm.jet(np.linspace(0, 1, num_labels))[:, :3]
                label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
                
                # 为每个投影点设置颜色
                vertex_colors = np.array([label_to_color[label] for label in new_vertices_labels])
                projected_pcd = o3d.geometry.PointCloud()
                projected_pcd.points = o3d.utility.Vector3dVector(projected_points)
                projected_pcd.colors = o3d.utility.Vector3dVector(vertex_colors)
                vis.add_geometry(projected_pcd)
            elif partition_labels is not None:
                # 尝试使用分区标签（如果新顶点标签不存在）
                # 为不同分区的边缘设置不同颜色
                unique_labels = np.unique(partition_labels)
                num_labels = len(unique_labels)
                print(f"为 {num_labels} 个分区创建颜色映射")
                
                # 创建颜色映射
                colors = plt.cm.jet(np.linspace(0, 1, num_labels))[:, :3]
                label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
                
                # 为每个投影点设置颜色
                # 注意：这里假设投影点的顺序与分区标签对应
                # 实际应用中可能需要更复杂的映射逻辑
                if len(projected_points) == len(partition_labels):
                    vertex_colors = np.array([label_to_color[label] for label in partition_labels])
                    projected_pcd = o3d.geometry.PointCloud()
                    projected_pcd.points = o3d.utility.Vector3dVector(projected_points)
                    projected_pcd.colors = o3d.utility.Vector3dVector(vertex_colors)
                    vis.add_geometry(projected_pcd)
                else:
                    # 如果长度不匹配，使用默认红色
                    projected_pcd = o3d.geometry.PointCloud()
                    projected_pcd.points = o3d.utility.Vector3dVector(projected_points)
                    projected_pcd.paint_uniform_color([1, 0, 0])  # 红色
                    vis.add_geometry(projected_pcd)
            else:
                # 如果没有分区标签，使用默认红色
                projected_pcd = o3d.geometry.PointCloud()
                projected_pcd.points = o3d.utility.Vector3dVector(projected_points)
                projected_pcd.paint_uniform_color([1, 0, 0])  # 红色
                vis.add_geometry(projected_pcd)
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        
        print("启动边缘拟合点可视化窗口...")
        vis.run()
        vis.destroy_window()
        print("边缘拟合点可视化完成")
    
    def _visualize_tool_paths_from_csv(self, result_path):
        """从CSV文件加载刀具路径并可视化"""
        import glob
        import csv
        
        # 查找所有tool_path_*.csv文件
        csv_files = glob.glob(os.path.join(result_path, "tool_path_*.csv"))
        if not csv_files:
            print("未找到刀具路径CSV文件")
            return
        
        print(f"找到 {len(csv_files)} 个刀具路径CSV文件")
        
        # 创建Open3D可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="刀具路径可视化 (从CSV加载)")
        
        # 添加坐标系
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(coord_frame)
        
        # 加载所有路径
        for i, csv_file in enumerate(csv_files):
            points = []
            try:
                with open(csv_file, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # 跳过表头
                    for row in reader:
                        if len(row) >= 3:
                            x, y, z = float(row[0]), float(row[1]), float(row[2])
                            points.append([x, y, z])
            except Exception as e:
                print(f"读取 {csv_file} 时出错: {e}")
                continue
            
            if not points:
                continue
            
            # 创建点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 为不同路径设置不同颜色
            color = [(i % 10) / 10.0, (i // 10) / 10.0, 0.5]
            pcd.paint_uniform_color(color)
            vis.add_geometry(pcd)
            
            # 创建线条
            if len(points) > 1:
                lines = []
                for j in range(len(points) - 1):
                    lines.append([j, j + 1])
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.paint_uniform_color(color)
                vis.add_geometry(line_set)
        
        # 设置视角
        vis.run()
        vis.destroy_window()
    
    def visualize_partition_edges(self, result_path):
        """可视化分区边缘数据"""
        print("\n=== 可视化分区边缘数据 ===")
        
        # 加载分区边缘数据
        edges_file = os.path.join(result_path, "partition_edges.json")
        if not os.path.exists(edges_file):
            print("未找到分区边缘数据文件 partition_edges.json")
            return
        
        try:
            with open(edges_file, 'r') as f:
                edges_data = json.load(f)
            
            print(f"加载分区边缘数据: {len(edges_data['edges'])} 条边缘")
            
            # 创建Open3D可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="分区边缘数据可视化")
            
            # 添加坐标系
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
            vis.add_geometry(coord_frame)
            
            # 收集所有边缘中点
            midpoints = []
            for edge in edges_data['edges']:
                midpoints.append(edge['midpoint'])
            
            # 创建边缘中点云
            if midpoints:
                print(f"添加 {len(midpoints)} 个边缘中点")
                midpoint_pcd = o3d.geometry.PointCloud()
                midpoint_pcd.points = o3d.utility.Vector3dVector(midpoints)
                midpoint_pcd.paint_uniform_color([1, 0, 0])  # 红色
                vis.add_geometry(midpoint_pcd)
            
            # 可选：可视化边缘连线（跳过，因为会导致类型转换错误）
            # edge_lines = []
            # line_points = []
            # line_indices = []
            # 
            # for i, edge in enumerate(edges_data['edges']):
            #     # 添加两个顶点
            #     v0 = edge['vertices'][0]
            #     v1 = edge['vertices'][1]
            #     line_points.extend([v0, v1])
            #     # 添加线索引
            #     line_indices.append([2*i, 2*i + 1])
            # 
            # if line_points:
            #     print(f"添加 {len(line_indices)} 条边缘连线")
            #     line_set = o3d.geometry.LineSet()
            #     line_set.points = o3d.utility.Vector3dVector(line_points)
            #     line_set.lines = o3d.utility.Vector2iVector(line_indices)
            #     line_set.paint_uniform_color([0, 1, 0])  # 绿色
            #     vis.add_geometry(line_set)
            print("跳过边缘连线可视化，以避免类型转换错误")
            
            # 设置视角
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            
            print("启动分区边缘数据可视化窗口...")
            vis.run()
            vis.destroy_window()
            print("分区边缘数据可视化完成")
        except Exception as e:
            print(f"可视化分区边缘数据时出错: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    root = Tk()
    app = ResultVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
