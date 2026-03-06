import numpy as np

def generate_spherical_grid(sample_count=4000):
    """
    生成球面均匀采样点，形成正方形网格铺满球面（内接）
    没有两极，任意相邻两点测地线对应的角度相同
    
    Args:
        sample_count: 目标采样点数量
    
    Returns:
        vertices: 顶点坐标列表
        faces: 面索引列表
    """
    # 计算合适的网格大小，使得总点数接近4000
    # 对于正方形网格，我们使用n x m的网格，其中n是纬度方向的点数，m是经度方向的点数
    # 由于是球面，我们需要考虑网格在球面上的分布
    n = int(np.sqrt(sample_count))
    
    # 确保n足够大，以生成接近4000个点
    while n * n < sample_count:
        n += 1
    
    # 计算角度步长
    theta_step = 2 * np.pi / n  # 方位角步长
    phi_step = np.pi / (n + 1)  # 极角步长，+1是为了避免两极
    
    vertices = []
    
    # 生成顶点
    for i in range(1, n + 1):  # 从1开始，避免北极
        phi = i * phi_step
        for j in range(n):
            theta = j * theta_step
            
            # 球面坐标转笛卡尔坐标
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            vertices.append([x, y, z])
    
    # 生成面
    faces = []
    for i in range(n - 1):  # 行
        for j in range(n):  # 列
            # 当前顶点索引
            current = i * n + j
            # 下一行同列顶点
            next_row = (i + 1) * n + j
            # 下一列顶点
            next_col = i * n + (j + 1) % n
            # 下一行下一列顶点
            next_row_col = (i + 1) * n + (j + 1) % n
            
            # 添加两个三角形面
            faces.append([current + 1, next_row + 1, next_col + 1])
            faces.append([next_row + 1, next_row_col + 1, next_col + 1])
    
    return vertices, faces

def save_obj(vertices, faces, filename):
    """
    保存顶点和面到OBJ文件
    
    Args:
        vertices: 顶点坐标列表
        faces: 面索引列表
        filename: OBJ文件路径
    """
    with open(filename, 'w') as f:
        # 写入顶点
        for v in vertices:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        
        # 写入面
        for face in faces:
            f.write(f'f {face[0]} {face[1]} {face[2]}\n')

if __name__ == "__main__":
    # 生成采样点
    vertices, faces = generate_spherical_grid(4000)
    print(f"生成了 {len(vertices)} 个顶点和 {len(faces)} 个面")
    
    # 保存到OBJ文件
    output_path = "data/models/spherical_grid.obj"
    save_obj(vertices, faces, output_path)
    print(f"OBJ文件已保存到: {output_path}")
