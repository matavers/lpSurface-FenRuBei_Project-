import open3d as o3d
import os
import numpy as np

def visualize_point_cloud_from_obj(obj_path):
    """
    使用Open3D从OBJ文件可视化点云
    """
    # 检查文件是否存在
    if not os.path.exists(obj_path):
        print(f"错误：文件 {obj_path} 不存在")
        return
    
    # 手动解析OBJ文件提取顶点数据
    print(f"正在解析OBJ文件: {obj_path}")
    vertices = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                # 提取顶点数据
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
    
    # 检查是否提取到顶点
    if not vertices:
        print("错误：未提取到顶点数据")
        return
    
    print(f"成功提取 {len(vertices)} 个顶点")
    
    # 创建点云
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(vertices))
    
    # 计算法线（用于更好的可视化效果）
    point_cloud.estimate_normals()
    
    # 创建可视化窗口
    print("正在创建可视化窗口...")
    print("按ESC键退出可视化")
    
    # 可视化点云
    o3d.visualization.draw_geometries([point_cloud])

def visualize_point_cloud_from_ply(ply_path):
    """
    使用Open3D从PLY文件可视化点云
    """
    # 检查文件是否存在
    if not os.path.exists(ply_path):
        print(f"错误：文件 {ply_path} 不存在")
        return
    
    # 加载PLY点云
    print(f"正在加载点云: {ply_path}")
    point_cloud = o3d.io.read_point_cloud(ply_path)
    
    # 检查点云是否加载成功
    if not point_cloud.has_points():
        print("错误：点云加载失败，没有点数据")
        return
    
    print(f"成功加载 {len(point_cloud.points)} 个点")
    
    # 计算法线（用于更好的可视化效果）
    point_cloud.estimate_normals()
    
    # 创建可视化窗口
    print("正在创建可视化窗口...")
    print("按ESC键退出可视化")
    
    # 可视化点云
    o3d.visualization.draw_geometries([point_cloud])

if __name__ == "__main__":
    # 点云文件路径
    ply_file = r"D:\Projects\lpSurface\GM\data\models\FloorChair(1).ply"
    
    # 调用点云可视化函数
    visualize_point_cloud_from_ply(ply_file)
