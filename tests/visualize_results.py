import open3d as o3d
import os

def visualize_point_cloud(ply_path):
    """
    可视化点云
    """
    # 检查文件是否存在
    if not os.path.exists(ply_path):
        print(f"错误：文件 {ply_path} 不存在")
        return
    
    # 加载点云
    print(f"正在加载点云: {ply_path}")
    point_cloud = o3d.io.read_point_cloud(ply_path)
    
    # 检查点云是否加载成功
    if not point_cloud.has_points():
        print("错误：点云加载失败，没有点数据")
        return
    
    print(f"成功加载 {len(point_cloud.points)} 个点")
    
    # 计算法线（用于更好的可视化效果）
    point_cloud.estimate_normals()
    
    # 可视化点云
    print("正在显示点云...")
    print("按ESC键退出可视化")
    o3d.visualization.draw_geometries([point_cloud], window_name="Point Cloud Visualization")

def visualize_mesh(obj_path):
    """
    可视化网格
    """
    # 检查文件是否存在
    if not os.path.exists(obj_path):
        print(f"错误：文件 {obj_path} 不存在")
        return
    
    # 加载网格
    print(f"正在加载网格: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)
    
    # 检查网格是否加载成功
    if not mesh.has_vertices():
        print("错误：网格加载失败，没有顶点数据")
        return
    
    print(f"成功加载网格：{len(mesh.vertices)} 个顶点，{len(mesh.triangles)} 个三角形")
    
    # 计算法线（用于更好的可视化效果）
    mesh.compute_vertex_normals()
    
    # 可视化网格
    print("正在显示网格...")
    print("按ESC键退出可视化")
    o3d.visualization.draw_geometries([mesh], window_name="Mesh Visualization")

if __name__ == "__main__":
    # 文件路径
    ply_file = r"D:\Projects\lpSurface\GM\data\models\FloorChair(1).ply"
    mesh_file = r"D:\Projects\lpSurface\GM\data\models\FloorChair(1)_mesh.obj"
    
    # 可视化点云
    print("=== 可视化点云 ===")
    visualize_point_cloud(ply_file)
    
    # 可视化网格
    print("\n=== 可视化网格 ===")
    visualize_mesh(mesh_file)
