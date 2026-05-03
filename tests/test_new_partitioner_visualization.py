"""
测试NewPartitioner类的功能并可视化结果
支持选择显示特定分区，重叠区域用混色
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import open3d as o3d
from core.meshProcessor import MeshProcessor
from new.newPartitoner import NewPartitioner

class PartitionerVisualizer:
    def __init__(self, mesh, partitions):
        self.mesh = mesh
        self.partitions = partitions
        self.num_partitions = len(partitions)
        
        self.vertex_partitions = [[] for _ in range(len(mesh.vertices))]
        for i, region in enumerate(partitions):
            for vertex in region:
                self.vertex_partitions[vertex].append(i)
        
        self.partition_sizes = [0] * self.num_partitions
        for vp in self.vertex_partitions:
            for p in vp:
                self.partition_sizes[p] += 1
        
        np.random.seed(42)
        self.colors = [np.random.rand(3) for _ in range(self.num_partitions)]
        
        self.selected_partitions = set()
        
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = mesh.vertices
        
        self.update_colors()
        
        self.print_summary()
        
    def update_colors(self):
        colors = np.zeros((len(self.mesh.vertices), 3))
        
        for i in range(len(self.mesh.vertices)):
            partition_indices = self.vertex_partitions[i]
            
            if len(partition_indices) == 0:
                colors[i] = [0.3, 0.3, 0.3]
            elif not self.selected_partitions:
                if len(partition_indices) == 1:
                    colors[i] = self.colors[partition_indices[0]]
                else:
                    mixed = np.zeros(3)
                    for idx in partition_indices:
                        mixed += np.array(self.colors[idx])
                    colors[i] = mixed / len(partition_indices)
            else:
                selected_in_point = [p for p in partition_indices if p in self.selected_partitions]
                if len(selected_in_point) == 0:
                    colors[i] = [0.12, 0.12, 0.12]
                elif len(selected_in_point) == 1:
                    colors[i] = self.colors[selected_in_point[0]]
                else:
                    mixed = np.zeros(3)
                    for idx in selected_in_point:
                        mixed += np.array(self.colors[idx])
                    colors[i] = mixed / len(selected_in_point)
        
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
        
    def print_summary(self):
        sorted_partitions = sorted(range(len(self.partition_sizes)), 
                                   key=lambda i: self.partition_sizes[i], reverse=True)
        
        multi_count = sum(1 for vp in self.vertex_partitions if len(vp) > 1)
        
        print("\n" + "="*60)
        print(f"分区统计:")
        print(f"  总分区数: {self.num_partitions}")
        print(f"  网格顶点数: {len(self.mesh.vertices)}")
        print(f"  属于多个分区的点: {multi_count}/{len(self.mesh.vertices)}")
        print(f"\n前20个最大分区:")
        for i, p_idx in enumerate(sorted_partitions[:20]):
            marker = "*" if p_idx in self.selected_partitions else " "
            print(f"  {marker}分区 {p_idx}: {self.partition_sizes[p_idx]} 个点")
        print("\n使用说明:")
        print("  - 输入分区编号 (如 0-9) 来选中/取消选中")
        print("  - 同时选中多个分区时，重叠区域会显示混色")
        print("  - 关闭可视化窗口退出")
        print("="*60)
        
    def run(self):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0.2, 0.2, 0.2])
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="NewPartitioner 分区显示", width=1280, height=960)
        vis.add_geometry(sphere)
        vis.add_geometry(self.point_cloud)
        
        render_option = vis.get_render_option()
        render_option.point_size = 6.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        
        print("\n开始可视化...")
        print("请在控制台输入分区编号 (如 0-9)")
        
        # 使用简单的循环让用户输入命令
        import threading
        import queue
        cmd_queue = queue.Queue()
        running = [True]
        
        def input_thread():
            while running[0]:
                try:
                    cmd = input("\n请输入分区编号 (0-9) 或 'a'全选前20个，'c'清空：").strip().lower()
                    cmd_queue.put(cmd)
                except:
                    pass
        
        t = threading.Thread(target=input_thread, daemon=True)
        t.start()
        
        while True:
            try:
                vis.poll_events()
                vis.update_renderer()
                
                try:
                    while True:
                        cmd = cmd_queue.get_nowait()
                        if cmd == 'a':
                            sorted_parts = sorted(range(len(self.partition_sizes)), 
                                               key=lambda i: self.partition_sizes[i], reverse=True)[:20]
                            self.selected_partitions = set(sorted_parts)
                            self.update_colors()
                            vis.update_geometry(self.point_cloud)
                            print(f"已选中前20个最大分区: {sorted(sorted_parts)}")
                        elif cmd == 'c':
                            self.selected_partitions.clear()
                            self.update_colors()
                            vis.update_geometry(self.point_cloud)
                            print("已清空选择")
                        else:
                            try:
                                idx = int(cmd)
                                if idx < 0 or idx >= self.num_partitions:
                                    print(f"分区索引无效，有效范围: 0-{self.num_partitions-1}")
                                else:
                                    if idx in self.selected_partitions:
                                        self.selected_partitions.remove(idx)
                                        print(f"取消选中分区 {idx}")
                                    else:
                                        self.selected_partitions.add(idx)
                                        print(f"选中分区 {idx}")
                                    
                                    print(f"当前选中分区: {sorted(self.selected_partitions) if self.selected_partitions else '无'}")
                                    
                                    self.update_colors()
                                    vis.update_geometry(self.point_cloud)
                            except:
                                print(f"无效命令: {cmd}")
                except queue.Empty:
                    pass
                
            except:
                break
        
        running[0] = False
        vis.destroy_window()
        print("\n可视化已关闭")

def test_new_partitioner_visualization():
    print("开始测试NewPartitioner并可视化结果...")
    
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    mesh.compute_vertex_normals()
    
    print(f"生成的网格: {len(mesh.vertices)} 个顶点, {len(mesh.triangles)} 个面")
    
    mesh_processor = MeshProcessor(mesh)
    partitioner = NewPartitioner(mesh_processor)
    
    partitions, _ = partitioner.partition_surface()
    print(f"分区完成: {len(partitions)} 个分区")
    
    visualizer = PartitionerVisualizer(mesh, partitions)
    visualizer.run()
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_new_partitioner_visualization()
