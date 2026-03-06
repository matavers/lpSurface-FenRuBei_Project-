"""
曲面生成器模块
用于生成自定义曲面的OBJ文件
"""

import numpy as np
import os
from scipy.interpolate import CubicSpline


class SurfaceGenerator:
    """
    曲面生成器类
    用于生成自定义曲面的OBJ文件
    """
    
    def __init__(self):
        """
        初始化曲面生成器
        """
        pass
    
    def generate_surface(self, func, resolution=(50, 50), bounds=((-1, 1), (-1, 1)), output_path="custom_surface.obj", parametric=False, density_factor=2.0, visualize_as_point_cloud=False, point_cloud_downsample=0.0):
        """
        生成自定义曲面的OBJ文件
        
        Args:
            func: 自定义函数，接受(u, v)作为输入，返回(x, y, z)值
            resolution: 分辨率，(u_res, v_res)
            bounds: 边界，((u_min, u_max), (v_min, v_max))
            output_path: 输出OBJ文件路径
            parametric: 是否使用参数化模式，True时func返回(x, y, z)，False时func返回z值
            density_factor: 密度调节系数，用于加密采样
            visualize_as_point_cloud: 是否以点云形式可视化
            point_cloud_downsample: 点云下采样因子，0.0表示不采样
            
        Returns:
            str: 生成的OBJ文件路径
        """
        u_res, v_res = resolution
        (u_min, u_max), (v_min, v_max) = bounds
        
        # 按照分辨率均匀采样
        print(f"按照分辨率 {resolution} 均匀采样...")
        
        # 生成顶点
        vertices = []
        uv_coords = []  # 存储参数坐标
        
        for i in range(u_res + 1):
            for j in range(v_res + 1):
                # 参数化坐标
                u = i / u_res
                v = j / v_res
                
                # 映射到实际参数范围
                u_param = u_min + u * (u_max - u_min)
                v_param = v_min + v * (v_max - v_min)
                
                if parametric:
                    # 参数化模式：func返回(x, y, z)
                    x, y, z = func(u_param, v_param)
                else:
                    # 非参数化模式：func返回z值
                    x = u_param
                    y = v_param
                    z = func(x, y)
                
                vertices.append((x, y, z))
                uv_coords.append((u_param, v_param))
        
        # 生成面
        faces = []
        for i in range(u_res):
            for j in range(v_res):
                # 四个顶点索引（OBJ文件从1开始计数）
                v0 = i * (v_res + 1) + j + 1
                v1 = (i + 1) * (v_res + 1) + j + 1
                v2 = (i + 1) * (v_res + 1) + (j + 1) + 1
                v3 = i * (v_res + 1) + (j + 1) + 1
                
                # 添加两个三角形
                faces.append((v0, v1, v2))
                faces.append((v0, v2, v3))
        
        # 执行密度增加操作
        if density_factor > 1.0:
            vertices, faces = self._refine_mesh(vertices, faces, func, bounds, parametric, density_factor, uv_coords)
        
        # 写入OBJ文件
        with open(output_path, 'w') as f:
            # 写入顶点
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
            # 写入面，确保索引有效
            num_vertices = len(vertices)
            for face in faces:
                # 确保面索引在有效范围内
                v0, v1, v2 = face
                if v0 <= num_vertices and v1 <= num_vertices and v2 <= num_vertices:
                    f.write(f"f {v0} {v1} {v2}\n")
                else:
                    # 跳过无效的面
                    print(f"跳过无效面: {face} (顶点数: {num_vertices})")
        
        print(f"曲面生成完成，保存到: {output_path}")
        
        # 可视化生成的网格
        self.visualize_mesh(vertices, faces, title=f"生成的曲面: {os.path.basename(output_path)}", as_point_cloud=visualize_as_point_cloud, point_cloud_downsample=point_cloud_downsample)
        
        return output_path
    
    def _refine_mesh(self, vertices, faces, func, bounds, parametric, density_factor, uv_coords=None):
        """
        根据曲率加密网格
        
        Args:
            vertices: 初始顶点列表
            faces: 初始面列表
            func: 曲面函数
            bounds: 边界
            parametric: 是否参数化
            density_factor: 密度调节系数
            uv_coords: 参数坐标列表
            
        Returns:
            加密后的顶点和面
        """
        print("执行密度增加操作...")
        
        # 转换为numpy数组
        vertices_np = np.array(vertices)
        faces_np = np.array(faces) - 1  # 转换为0-based索引
        
        # 计算每个顶点的曲率
        curvatures = self._calculate_curvatures(vertices_np, faces_np, func, bounds, parametric)
        
        # 加密准则：根据局部曲率大小调整采样密度
        # ρ(u,v) = 1 + α·max(|k₁(u,v)|,|k₂(u,v)|)
        # 其中α为调节系数，这里使用density_factor作为α
        alpha = density_factor
        
        # 计算每个顶点的加密权重
        encryption_weights = 1 + alpha * curvatures
        
        # 迭代细分
        max_iterations = 3  # 减少迭代次数，从5次减少到3次
        for iteration in range(max_iterations):
            print(f"迭代 {iteration+1}/{max_iterations}...")
            new_vertices = vertices_np.tolist()
            new_uv_coords = uv_coords.copy() if uv_coords else []
            new_faces = []
            vertex_offset = len(new_vertices)
            
            # 统计需要加密的三角形数量
            refined_count = 0
            
            for face in faces_np:
                # 确保顶点索引是整数
                v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
                
                # 确保顶点索引在有效范围内
                if v0 >= len(vertices_np) or v1 >= len(vertices_np) or v2 >= len(vertices_np):
                    print(f"跳过无效面: {face} (顶点数: {len(vertices_np)})")
                    continue
                
                # 计算三角形顶点
                p0 = vertices_np[v0]
                p1 = vertices_np[v1]
                p2 = vertices_np[v2]
                
                # 计算边长
                edge1 = np.linalg.norm(p1 - p0)
                edge2 = np.linalg.norm(p2 - p1)
                edge3 = np.linalg.norm(p0 - p2)
                max_edge = max(edge1, edge2, edge3)
                
                # 计算曲率变化
                curv_change = abs(curvatures[v0] - curvatures[v1]) + \
                             abs(curvatures[v1] - curvatures[v2]) + \
                             abs(curvatures[v2] - curvatures[v0])
                
                # 计算三角形重心
                centroid = (p0 + p1 + p2) / 3
                
                # 计算重心到原点的距离（用于判断是否在两极区域）
                distance_to_center = np.linalg.norm(centroid)
                
                # 两极区域检测（避免在两极过度加密）
                is_polar_region = False
                if distance_to_center > 0:
                    # 计算向量与z轴的夹角
                    z_component = abs(centroid[2]) / distance_to_center
                    # 如果夹角小于15度，认为是两极区域
                    if z_component > np.cos(np.pi / 12):  # 15度
                        is_polar_region = True
                
                # 调整加密阈值，在两极区域使用更大的阈值
                edge_threshold = 0.2  # 增加边长阈值，从0.1增加到0.2
                curv_threshold = 0.1  # 曲率变化阈值
                
                if is_polar_region:
                    # 在两极区域增加阈值，减少加密
                    edge_threshold *= 1.5
                    curv_threshold *= 1.5
                
                if max_edge > edge_threshold or curv_change > curv_threshold:
                    # 计算重心处的参数坐标
                    new_u, new_v = 0.0, 0.0
                    if parametric and uv_coords:
                        # 对于参数化曲面，使用顶点的参数坐标来计算重心处的参数
                        # 获取三角形三个顶点的参数坐标
                        v0_idx = v0
                        v1_idx = v1
                        v2_idx = v2
                        
                        # 确保索引在有效范围内
                        if v0_idx < len(uv_coords) and v1_idx < len(uv_coords) and v2_idx < len(uv_coords):
                            u0, v0_uv = uv_coords[v0_idx]
                            u1, v1_uv = uv_coords[v1_idx]
                            u2, v2_uv = uv_coords[v2_idx]
                            
                            # 计算参数坐标的重心
                            new_u = (u0 + u1 + u2) / 3
                            new_v = (v0_uv + v1_uv + v2_uv) / 3
                            
                            # 确保参数在有效范围内
                            (u_min, u_max), (v_min, v_max) = bounds
                            new_u = max(u_min, min(u_max, new_u))
                            new_v = max(v_min, min(v_max, new_v))
                            
                            x, y, z = func(new_u, new_v)
                        else:
                            # 如果索引无效，使用空间重心
                            x, y, z = centroid
                    else:
                        x, y = centroid[0], centroid[1]
                        z = func(x, y)
                    
                    new_vertex = (x, y, z)
                    new_vertices.append(new_vertex)
                    if uv_coords:
                        new_uv_coords.append((new_u, new_v))
                    
                    # 分割三角形为三个小三角形
                    new_faces.append([v0, v1, vertex_offset])
                    new_faces.append([v1, v2, vertex_offset])
                    new_faces.append([v2, v0, vertex_offset])
                    vertex_offset += 1
                    refined_count += 1
                else:
                    # 不需要加密，保留原三角形
                    new_faces.append([v0, v1, v2])
            
            # 更新顶点和面
            vertices_np = np.array(new_vertices)
            faces_np = np.array(new_faces, dtype=int)
            if uv_coords:
                uv_coords = new_uv_coords
            
            print(f"  迭代完成，加密了 {refined_count} 个三角形")
            print(f"  当前顶点数: {len(vertices_np)}, 面数: {len(faces_np)}")
            
            # 重新计算曲率
            if iteration < max_iterations - 1:
                curvatures = self._calculate_curvatures(vertices_np, faces_np, func, bounds, parametric)
                encryption_weights = 1 + alpha * curvatures
        
        # 转换回1-based索引
        faces_np = faces_np + 1
        
        print(f"密度增加操作完成，顶点数: {len(vertices_np)}, 面数: {len(faces_np)}")
        return vertices_np.tolist(), faces_np.tolist()
    
    def _calculate_curvatures(self, vertices, faces, func, bounds, parametric):
        """
        计算每个顶点的曲率
        
        Args:
            vertices: 顶点数组
            faces: 面数组（0-based）
            func: 曲面函数
            bounds: 边界
            parametric: 是否参数化
            
        Returns:
            曲率数组
        """
        n_vertices = len(vertices)
        curvatures = np.zeros(n_vertices)
        
        print(f"计算 {n_vertices} 个顶点的曲率...")
        
        # 预处理：构建顶点到面的映射
        vertex_to_faces = [[] for _ in range(n_vertices)]
        for i, face in enumerate(faces):
            # 确保面索引是整数
            v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
            vertex_to_faces[v0].append(i)
            vertex_to_faces[v1].append(i)
            vertex_to_faces[v2].append(i)
        
        # 构建顶点邻接关系
        adjacency = [[] for _ in range(n_vertices)]
        for face in faces:
            v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
            adjacency[v0].append(v1)
            adjacency[v0].append(v2)
            adjacency[v1].append(v0)
            adjacency[v1].append(v2)
            adjacency[v2].append(v0)
            adjacency[v2].append(v1)
        
        # 预计算所有顶点的法向量
        normals = []
        for i in range(n_vertices):
            normal = self._calculate_normal(i, vertices, faces, vertex_to_faces)
            normals.append(normal)
        normals = np.array(normals)
        
        # 计算每个顶点的曲率
        for i in range(n_vertices):
            neighbors = adjacency[i]
            if len(neighbors) < 3:
                continue
            
            # 获取当前顶点的法向量
            normal = normals[i]
            
            # 计算邻居顶点的法向量变化
            normal_variation = 0
            for neighbor in neighbors:
                neighbor_normal = normals[neighbor]
                dot_product = np.dot(normal, neighbor_normal)
                angle = np.arccos(np.clip(dot_product, -1, 1))
                normal_variation += angle
            
            curvatures[i] = normal_variation / len(neighbors)
        
        # 归一化曲率
        max_curvature = np.max(curvatures)
        if max_curvature > 0:
            curvatures = curvatures / max_curvature
        
        print("曲率计算完成")
        return curvatures
    
    def _calculate_normal(self, vertex_idx, vertices, faces, vertex_to_faces):
        """
        计算顶点的法向量
        
        Args:
            vertex_idx: 顶点索引
            vertices: 顶点数组
            faces: 面数组
            vertex_to_faces: 顶点到面的映射
            
        Returns:
            法向量
        """
        # 找到包含该顶点的所有面
        face_indices = vertex_to_faces[vertex_idx]
        vertex_faces = [faces[i] for i in face_indices]
        
        # 计算每个面的法向量
        normals = []
        for face in vertex_faces:
            # 确保面索引是整数
            v0, v1, v2 = int(face[0]), int(face[1]), int(face[2])
            p0 = vertices[v0]
            p1 = vertices[v1]
            p2 = vertices[v2]
            
            # 计算面法向量
            v1v0 = p1 - p0
            v2v0 = p2 - p0
            normal = np.cross(v1v0, v2v0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            normals.append(normal)
        
        # 平均法向量
        if normals:
            avg_normal = np.mean(normals, axis=0)
            norm = np.linalg.norm(avg_normal)
            if norm > 0:
                avg_normal = avg_normal / norm
            return avg_normal
        else:
            return np.array([0, 0, 1])
    
    def generate_sphere(self, radius=1.0, resolution=50, output_path="sphere.obj", density_factor=2.0, visualize_as_point_cloud=False, point_cloud_downsample=0.0, uniform_density=True, uniform_sampling=False):
        """
        生成球体的OBJ文件
        
        Args:
            radius: 球体半径
            resolution: 分辨率
            output_path: 输出OBJ文件路径
            density_factor: 密度调节系数
            visualize_as_point_cloud: 是否以点云形式可视化
            point_cloud_downsample: 点云下采样因子，0.0表示不采样
            uniform_density: 是否使用均匀密度分布
            uniform_sampling: 是否使用均匀采样生成网格
            
        Returns:
            str: 生成的OBJ文件路径
        """
        print(f"generate_sphere called with uniform_sampling={uniform_sampling}")
        if uniform_sampling:
            print(f"Calling generate_uniform_sphere with output_path={output_path}")
            return self.generate_uniform_sphere(radius, resolution, output_path, density_factor, visualize_as_point_cloud, point_cloud_downsample)
        else:
            def sphere_func(u, v):
                # 参数化球体
                theta = 2 * np.pi * u
                if uniform_density:
                    # 使用改进的参数化方法，减少两极密度
                    # 将v从[0,1]映射到[0,π]，使用sin(phi/2)来均匀分布点
                    phi = 2 * np.arcsin(np.sqrt(v))
                else:
                    phi = np.pi * v
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)
                return x, y, z
            
            return self.generate_surface(
                sphere_func,
                resolution=(resolution, resolution),
                bounds=((0, 1), (0, 1)),
                output_path=output_path,
                parametric=True,
                density_factor=density_factor,
                visualize_as_point_cloud=visualize_as_point_cloud,
                point_cloud_downsample=point_cloud_downsample
            )
    
    def generate_uniform_sphere(self, radius=1.0, resolution=50, output_path="uniform_sphere.obj", density_factor=1.0, visualize_as_point_cloud=False, point_cloud_downsample=0.0):
        """
        生成均匀采样的球体网格，确保各向对称，每个网格单元为正方形
        
        Args:
            radius: 球体半径
            resolution: 分辨率
            output_path: 输出OBJ文件路径
            density_factor: 密度调节系数
            visualize_as_point_cloud: 是否以点云形式可视化
            point_cloud_downsample: 点云下采样因子，0.0表示不采样
            
        Returns:
            str: 生成的OBJ文件路径
        """
        print(f"生成均匀采样的球体网格，分辨率: {resolution}")
        
        # 生成顶点
        vertices = []
        faces = []
        
        # 计算纬度和经度的步长
        # 为了保证各向对称，使用相同的分辨率
        num_lat = resolution
        num_lon = resolution
        
        # 生成顶点
        for i in range(num_lat + 1):
            # 计算纬度角度，使用sin(phi/2)来均匀分布点
            phi = 2 * np.arcsin(np.sqrt(i / num_lat))
            for j in range(num_lon + 1):
                # 计算经度角度
                theta = 2 * np.pi * j / num_lon
                
                # 计算顶点坐标
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)
                
                vertices.append((x, y, z))
        
        # 生成面
        for i in range(num_lat):
            for j in range(num_lon):
                # 计算四个顶点的索引
                v0 = i * (num_lon + 1) + j
                v1 = (i + 1) * (num_lon + 1) + j
                v2 = (i + 1) * (num_lon + 1) + (j + 1)
                v3 = i * (num_lon + 1) + (j + 1)
                
                # 添加两个三角形
                faces.append((v0 + 1, v1 + 1, v2 + 1))  # OBJ文件从1开始计数
                faces.append((v0 + 1, v2 + 1, v3 + 1))
        
        # 写入OBJ文件
        with open(output_path, 'w') as f:
            # 写入顶点
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
            # 写入面
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")
        
        print(f"均匀采样球体生成完成，保存到: {output_path}")
        print(f"顶点数: {len(vertices)}, 面数: {len(faces)}")
        
        # 可视化生成的网格
        self.visualize_mesh(vertices, faces, title=f"均匀采样的球体: {os.path.basename(output_path)}", as_point_cloud=visualize_as_point_cloud, point_cloud_downsample=point_cloud_downsample)
        
        return output_path
    
    def generate_uniform_ellipsoid(self, semi_axes=(1.0, 1.0, 1.0), resolution=50, output_path="uniform_ellipsoid.obj", density_factor=1.0, visualize_as_point_cloud=False, point_cloud_downsample=0.0):
        """
        生成均匀采样的椭球体网格，确保各向对称，每个网格单元为长方形
        
        Args:
            semi_axes: 椭球的三个半轴长度
            resolution: 分辨率
            output_path: 输出OBJ文件路径
            density_factor: 密度调节系数
            visualize_as_point_cloud: 是否以点云形式可视化
            point_cloud_downsample: 点云下采样因子，0.0表示不采样
            
        Returns:
            str: 生成的OBJ文件路径
        """
        print(f"生成均匀采样的椭球体网格，分辨率: {resolution}")
        
        # 生成顶点
        vertices = []
        faces = []
        
        # 计算纬度和经度的步长
        # 为了保证各向对称，使用相同的分辨率
        num_lat = resolution
        num_lon = resolution
        
        # 提取半轴长度
        a, b, c = semi_axes
        
        # 生成顶点
        for i in range(num_lat + 1):
            # 计算纬度角度，使用sin(phi/2)来均匀分布点
            phi = 2 * np.arcsin(np.sqrt(i / num_lat))
            for j in range(num_lon + 1):
                # 计算经度角度
                theta = 2 * np.pi * j / num_lon
                
                # 计算顶点坐标
                x = a * np.sin(phi) * np.cos(theta)
                y = b * np.sin(phi) * np.sin(theta)
                z = c * np.cos(phi)
                
                vertices.append((x, y, z))
        
        # 生成面
        for i in range(num_lat):
            for j in range(num_lon):
                # 计算四个顶点的索引
                v0 = i * (num_lon + 1) + j
                v1 = (i + 1) * (num_lon + 1) + j
                v2 = (i + 1) * (num_lon + 1) + (j + 1)
                v3 = i * (num_lon + 1) + (j + 1)
                
                # 添加两个三角形
                faces.append((v0 + 1, v1 + 1, v2 + 1))  # OBJ文件从1开始计数
                faces.append((v0 + 1, v2 + 1, v3 + 1))
        
        # 写入OBJ文件
        with open(output_path, 'w') as f:
            # 写入顶点
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
            # 写入面
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")
        
        print(f"均匀采样椭球体生成完成，保存到: {output_path}")
        print(f"顶点数: {len(vertices)}, 面数: {len(faces)}")
        
        # 可视化生成的网格
        self.visualize_mesh(vertices, faces, title=f"均匀采样的椭球体: {os.path.basename(output_path)}", as_point_cloud=visualize_as_point_cloud, point_cloud_downsample=point_cloud_downsample)
        
        return output_path
    
    def generate_torus(self, radius=1.0, tube_radius=0.3, resolution=50, output_path="torus.obj", density_factor=2.0, visualize_as_point_cloud=False, point_cloud_downsample=0.0):
        """
        生成圆环的OBJ文件
        
        Args:
            radius: 圆环半径
            tube_radius: 管半径
            resolution: 分辨率
            output_path: 输出OBJ文件路径
            density_factor: 密度调节系数
            visualize_as_point_cloud: 是否以点云形式可视化
            point_cloud_downsample: 点云下采样因子，0.0表示不采样
            
        Returns:
            str: 生成的OBJ文件路径
        """
        def torus_func(u, v):
            # 参数化圆环
            theta = 2 * np.pi * u
            phi = 2 * np.pi * v
            x = (radius + tube_radius * np.cos(phi)) * np.cos(theta)
            y = (radius + tube_radius * np.cos(phi)) * np.sin(theta)
            z = tube_radius * np.sin(phi)
            return x, y, z
        
        return self.generate_surface(
            torus_func,
            resolution=(resolution, resolution),
            bounds=((0, 1), (0, 1)),
            output_path=output_path,
            parametric=True,
            density_factor=density_factor,
            visualize_as_point_cloud=visualize_as_point_cloud,
            point_cloud_downsample=point_cloud_downsample
        )
    
    def generate_saddle(self, resolution=50, scale=1.0, output_path="saddle.obj", density_factor=2.0, visualize_as_point_cloud=False, point_cloud_downsample=0.0):
        """
        生成马鞍面的OBJ文件
        
        Args:
            resolution: 分辨率
            scale: 缩放因子
            output_path: 输出OBJ文件路径
            density_factor: 密度调节系数
            visualize_as_point_cloud: 是否以点云形式可视化
            point_cloud_downsample: 点云下采样因子，0.0表示不采样
            
        Returns:
            str: 生成的OBJ文件路径
        """
        def saddle_func(x, y):
            # 马鞍面方程: z = x^2 - y^2
            return (x**2 - y**2) * scale
        
        # 确保resolution是一个整数
        if isinstance(resolution, tuple):
            u_res, v_res = resolution
        else:
            u_res = v_res = resolution
        
        return self.generate_surface(
            saddle_func,
            resolution=(u_res, v_res),
            bounds=((-scale, scale), (-scale, scale)),
            output_path=output_path,
            density_factor=density_factor,
            visualize_as_point_cloud=visualize_as_point_cloud,
            point_cloud_downsample=point_cloud_downsample
        )
    
    def fit_edge_curve(self, edge_points, num_samples=50):
        """
        基于最小二乘的边缘拟合算法，生成闭合包围圈
        平滑处理时保证位置连续，确保生成的拟合边缘能够连接成包围圈，覆盖原分区
        
        Args:
            edge_points: 边缘点列表
            num_samples: 采样点数量
            
        Returns:
            拟合曲线上的采样点（闭合包围圈）
        """
        print("拟合边缘曲线...")
        
        # 转换为numpy数组
        points = np.array(edge_points)
        n_points = len(points)
        
        if n_points <= 4:  # 小于等于4个点时，直接采用原网格
            print("边缘点数量较少，直接采用原网格点")
            # 确保返回足够数量的点
            if n_points < num_samples:
                # 重复原有点以达到指定数量
                sampled_points = []
                for i in range(num_samples):
                    idx = i % n_points
                    sampled_points.append(tuple(points[idx]))
            else:
                # 使用原有点
                sampled_points = [tuple(p) for p in points[:num_samples]]
            # 确保闭合
            if sampled_points:
                sampled_points[-1] = sampled_points[0]
            print(f"边缘曲线处理完成，采样点数量: {len(sampled_points)}")
            return sampled_points
        
        # 1. 点云预处理 - 去除噪声点
        print("预处理点云...")
        filtered_points = self._preprocess_points(points)
        
        # 2. 直接使用闭合三次样条拟合，避免RANSAC直线检测在球面上的问题
        print("使用闭合三次样条拟合...")
        sampled_points = self._fit_closed_cubic_spline(filtered_points, num_samples)
        
        # 3. 确保形成包围圈
        print("确保形成包围圈...")
        sampled_points = self._ensure_enclosure(sampled_points, filtered_points)
        
        # 4. 确保经过原网格点
        print("确保经过原网格点...")
        sampled_points = self._ensure_pass_through_original_points(sampled_points, filtered_points)
        
        print(f"边缘曲线拟合完成，采样点数量: {len(sampled_points)}")
        return sampled_points
    
    def _preprocess_points(self, points):
        """
        预处理点云，去除噪声点
        """
        # 计算点云的统计信息
        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        
        # 去除离群点（超过2倍标准差的点）
        filtered_points = []
        for point in points:
            if all(np.abs(point - mean) < 2 * std):
                filtered_points.append(point)
        
        return np.array(filtered_points) if filtered_points else points
    
    def _sort_edge_points_by_distance(self, points):
        """
        基于距离的边缘点排序
        """
        if len(points) <= 1:
            return points
        
        # 从第一个点开始
        sorted_points = [points[0]]
        remaining_points = list(points[1:])
        
        while remaining_points:
            last_point = sorted_points[-1]
            # 找到最近的点
            distances = [np.linalg.norm(p - last_point) for p in remaining_points]
            nearest_idx = np.argmin(distances)
            sorted_points.append(remaining_points.pop(nearest_idx))
        
        return np.array(sorted_points)
    
    def _detect_line_segments_ransac(self, points):
        """
        使用RANSAC算法检测直线段
        """
        from sklearn.linear_model import RANSACRegressor
        
        line_segments = []
        remaining_points = list(points)
        
        while len(remaining_points) >= 5:
            # 转换为2D空间
            X = np.array([p[0] for p in remaining_points]).reshape(-1, 1)
            y = np.array([p[1] for p in remaining_points])
            
            # 使用RANSAC检测直线
            ransac = RANSACRegressor(min_samples=3, residual_threshold=0.1)
            try:
                ransac.fit(X, y)
                inlier_mask = ransac.inlier_mask_
                
                # 收集内点
                inliers = [remaining_points[i] for i, mask in enumerate(inlier_mask) if mask]
                if len(inliers) >= 3:
                    # 按x坐标排序
                    inliers.sort(key=lambda p: p[0])
                    line_segments.append(np.array(inliers))
                    # 移除内点
                    remaining_points = [p for i, p in enumerate(remaining_points) if not inlier_mask[i]]
                else:
                    break
            except:
                break
        
        return line_segments
    
    def _split_into_line_curve_segments(self, points, line_segments):
        """
        分割点云为直线段和曲线段
        """
        if not line_segments:
            # 如果没有检测到直线段，使用均匀分割
            num_segments = 4
            segment_size = len(points) // num_segments
            line_segments = [points[i*segment_size:(i+1)*segment_size] for i in range(num_segments)]
            curve_segments = []
        else:
            # 基于直线段分割曲线段
            curve_segments = []
            all_line_points = set()
            for segment in line_segments:
                for point in segment:
                    all_line_points.add(tuple(point))
            
            # 收集非直线点作为曲线段
            curve_points = [p for p in points if tuple(p) not in all_line_points]
            if curve_points:
                # 均匀分割曲线点
                num_curve_segments = max(1, len(line_segments) - 1)
                curve_segment_size = len(curve_points) // num_curve_segments
                curve_segments = [curve_points[i*curve_segment_size:(i+1)*curve_segment_size] for i in range(num_curve_segments)]
        
        return line_segments, curve_segments
    
    def _fit_segments(self, line_segments, curve_segments):
        """
        拟合直线段和曲线段
        """
        fitted_segments = []
        
        # 拟合直线段
        for i, segment in enumerate(line_segments):
            if len(segment) >= 2:
                line_params = self._fit_line(segment)
                fitted_segments.append(('line', line_params))
        
        # 拟合曲线段
        for i, segment in enumerate(curve_segments):
            if len(segment) >= 3:
                curve = self._fit_curve(segment)
                if curve:
                    fitted_segments.append(('curve', curve))
        
        # 确保至少有两个段
        if len(fitted_segments) < 2:
            # 如果只有直线段，添加曲线段
            if len(line_segments) >= 1:
                # 使用最后一个直线段的点创建曲线段
                last_line = line_segments[-1]
                if len(last_line) >= 3:
                    curve = self._fit_curve(last_line)
                    if curve:
                        fitted_segments.append(('curve', curve))
        
        return fitted_segments
    
    def _fit_line(self, points):
        """
        拟合直线
        """
        from sklearn.linear_model import LinearRegression
        
        # 转换为2D空间（投影到最佳拟合平面）
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        covariance = np.dot(centered_points.T, centered_points)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # 选择前两个特征向量作为平面
        plane_basis = eigenvectors[:, -2:]
        projected_points = np.dot(centered_points, plane_basis)
        
        # 拟合直线
        X = projected_points[:, 0].reshape(-1, 1)
        y = projected_points[:, 1]
        model = LinearRegression().fit(X, y)
        
        # 构建直线参数
        start_x = projected_points[0, 0]
        start_pred = model.predict([[start_x]])[0]
        end_x = projected_points[-1, 0]
        end_pred = model.predict([[end_x]])[0]
        
        # 转换回3D空间
        start_point = centroid + np.dot(plane_basis, np.array([start_x, start_pred]))
        end_point = centroid + np.dot(plane_basis, np.array([end_x, end_pred]))
        
        return {
            'start': start_point,
            'end': end_point,
            'direction': end_point - start_point
        }
    
    def _fit_curve(self, points):
        """
        拟合曲线（使用三次样条）
        """
        if len(points) < 3:
            return None
        
        # 确保points是numpy数组
        points = np.array(points)
        
        t = np.linspace(0, 1, len(points))
        spline_x = CubicSpline(t, points[:, 0])
        spline_y = CubicSpline(t, points[:, 1])
        spline_z = CubicSpline(t, points[:, 2])
        
        return {
            'spline_x': spline_x,
            'spline_y': spline_y,
            'spline_z': spline_z,
            'length': len(points)
        }
    
    def _fit_closed_cubic_spline(self, points, num_samples):
        """
        拟合闭合的三次样条曲线
        """
        # 对于点数量较少的情况，使用简单的线性插值
        if len(points) < 3:
            sampled_points = []
            for i in range(num_samples):
                t = i / max(1, num_samples - 1)
                # 线性插值
                idx = int(t * (len(points) - 1))
                next_idx = (idx + 1) % len(points)
                weight = t * (len(points) - 1) - idx
                point = points[idx] * (1 - weight) + points[next_idx] * weight
                sampled_points.append(tuple(point))
            # 确保首尾相连
            if sampled_points:
                sampled_points[-1] = sampled_points[0]
            return sampled_points
        
        # 正常情况：使用三次样条
        t = np.linspace(0, 1, len(points))
        
        # 确保t是严格递增的
        t = np.unique(t)
        if len(t) < len(points):
            # 如果有重复值，重新生成
            t = np.linspace(0, 1, len(points))
        
        spline_x = CubicSpline(t, points[:, 0])
        spline_y = CubicSpline(t, points[:, 1])
        spline_z = CubicSpline(t, points[:, 2])
        
        sample_t = np.linspace(0, 1, num_samples)
        sampled_points = []
        for t_val in sample_t:
            x = spline_x(t_val)
            y = spline_y(t_val)
            z = spline_z(t_val)
            sampled_points.append((x, y, z))
        
        # 确保首尾相连
        if sampled_points:
            sampled_points[-1] = sampled_points[0]
        
        return sampled_points
    
    def _generate_closed_sampled_points(self, fitted_segments, num_samples):
        """
        生成闭合的采样点，确保位置连续
        """
        if not fitted_segments:
            return []
        
        sampled_points = []
        points_per_segment = num_samples // len(fitted_segments)
        
        for i, (segment_type, segment_data) in enumerate(fitted_segments):
            if segment_type == 'line':
                # 直线采样
                direction = segment_data['direction']
                length = np.linalg.norm(direction)
                if length > 0:
                    for j in range(points_per_segment):
                        t = j / max(1, points_per_segment - 1)
                        point = segment_data['start'] + t * direction
                        sampled_points.append(tuple(point))
            elif segment_type == 'curve':
                # 曲线采样
                for j in range(points_per_segment):
                    t = j / max(1, points_per_segment - 1)
                    x = segment_data['spline_x'](t)
                    y = segment_data['spline_y'](t)
                    z = segment_data['spline_z'](t)
                    sampled_points.append((x, y, z))
        
        # 补充剩余点
        while len(sampled_points) < num_samples:
            if sampled_points:
                last_point = np.array(sampled_points[-1])
                second_last_point = np.array(sampled_points[-2]) if len(sampled_points) > 1 else last_point
                direction = last_point - second_last_point
                new_point = last_point + direction * 0.1
                sampled_points.append(tuple(new_point))
            else:
                sampled_points.append((0, 0, 0))
        
        # 确保闭合和位置连续
        if sampled_points:
            # 确保首尾相连
            sampled_points[-1] = sampled_points[0]
            
            # 确保位置连续，检查相邻点之间的距离
            i = 0
            while i < len(sampled_points) - 1:
                current_point = np.array(sampled_points[i])
                next_point = np.array(sampled_points[i + 1])
                distance = np.linalg.norm(next_point - current_point)
                
                # 如果距离过大，在中间插入点以保证连续
                if distance > 0.1:  # 阈值可以根据实际情况调整
                    num_intermediate_points = int(distance / 0.1) + 1
                    for j in range(1, num_intermediate_points):
                        t = j / num_intermediate_points
                        intermediate_point = current_point * (1 - t) + next_point * t
                        sampled_points.insert(i + j, tuple(intermediate_point))
                    i += num_intermediate_points
                else:
                    i += 1
        
        # 确保返回指定数量的点
        return sampled_points[:num_samples]
    
    def _ensure_enclosure(self, sampled_points, original_points):
        """
        确保生成的边缘能够形成包围圈，覆盖原分区
        同时保证平滑处理时位置连续
        """
        if not sampled_points or not len(original_points):
            return sampled_points
        
        # 计算原始点云的边界
        min_coords = np.min(original_points, axis=0)
        max_coords = np.max(original_points, axis=0)
        center = (min_coords + max_coords) / 2
        
        # 计算原始点云的半径
        max_radius = max(np.linalg.norm(p - center) for p in original_points)
        
        # 调整采样点，确保覆盖整个区域
        adjusted_points = []
        for point in sampled_points:
            # 计算点到中心的向量
            vec = np.array(point) - center
            distance = np.linalg.norm(vec)
            
            # 如果点在原始点云内部，向外扩展
            if distance < max_radius * 0.9:
                scale_factor = max_radius * 1.1 / distance
                new_point = center + vec * scale_factor
                adjusted_points.append(tuple(new_point))
            else:
                adjusted_points.append(point)
        
        # 确保闭合和位置连续
        if adjusted_points:
            # 确保首尾相连
            adjusted_points[-1] = adjusted_points[0]
            
            # 确保位置连续，检查相邻点之间的距离
            for i in range(len(adjusted_points) - 1):
                current_point = np.array(adjusted_points[i])
                next_point = np.array(adjusted_points[i + 1])
                distance = np.linalg.norm(next_point - current_point)
                
                # 如果距离过大，在中间插入点以保证连续
                if distance > 0.1:  # 阈值可以根据实际情况调整
                    num_intermediate_points = int(distance / 0.1) + 1
                    for j in range(1, num_intermediate_points):
                        t = j / num_intermediate_points
                        intermediate_point = current_point * (1 - t) + next_point * t
                        adjusted_points.insert(i + j, tuple(intermediate_point))
        
        return adjusted_points
    
    def _ensure_pass_through_original_points(self, sampled_points, original_points):
        """
        确保拟合结果经过原网格点
        
        Args:
            sampled_points: 拟合后的采样点
            original_points: 原始网格点
            
        Returns:
            调整后的采样点，确保经过原网格点
        """
        if not sampled_points or not len(original_points):
            return sampled_points
        
        # 转换为numpy数组
        sampled_array = np.array(sampled_points)
        original_array = np.array(original_points)
        
        # 限制处理的原始点数量，避免性能问题
        max_points = 50  # 设置最大处理点数
        if len(original_array) > max_points:
            # 均匀采样原始点
            step = len(original_array) // max_points
            original_array = original_array[::step]
        
        # 确保经过所有原始点
        adjusted_points = list(sampled_array)
        
        for original_point in original_array:
            # 检查原始点是否已经在采样点中
            # 优化：只计算到调整后点的距离，而不是原始采样点
            current_array = np.array(adjusted_points)
            distances = np.linalg.norm(current_array - original_point, axis=1)
            min_distance = np.min(distances)
            
            # 如果原始点不在采样点中，找到最近的两个点并插入
            if min_distance > 0.01:  # 阈值可以根据实际情况调整
                # 找到最近的两个点
                nearest_idx = np.argmin(distances)
                next_idx = (nearest_idx + 1) % len(adjusted_points)
                
                # 插入原始点到最近的两个点之间
                adjusted_points.insert(next_idx, original_point)
        
        # 确保闭合
        if adjusted_points:
            adjusted_points[-1] = adjusted_points[0]
        
        # 确保位置连续
        i = 0
        max_iterations = len(adjusted_points) * 2  # 防止无限循环
        iteration_count = 0
        
        while i < len(adjusted_points) - 1 and iteration_count < max_iterations:
            current_point = adjusted_points[i]
            next_point = adjusted_points[i + 1]
            distance = np.linalg.norm(next_point - current_point)
            
            # 如果距离过大，在中间插入点以保证连续
            if distance > 0.1:  # 阈值可以根据实际情况调整
                num_intermediate_points = min(int(distance / 0.1) + 1, 5)  # 限制插入点数量
                for j in range(1, num_intermediate_points):
                    t = j / num_intermediate_points
                    intermediate_point = current_point * (1 - t) + next_point * t
                    adjusted_points.insert(i + j, intermediate_point)
                i += num_intermediate_points
            else:
                i += 1
            iteration_count += 1
        
        # 限制最终点数量，避免过多点导致性能问题
        max_final_points = 200
        if len(adjusted_points) > max_final_points:
            step = len(adjusted_points) // max_final_points
            adjusted_points = adjusted_points[::step]
            # 确保闭合
            if adjusted_points:
                adjusted_points[-1] = adjusted_points[0]
        
        # 转换回元组列表
        return [tuple(p) for p in adjusted_points]
    
    def project_to_mesh(self, points, vertices, faces):
        """
        将点列投影到原始网格中的三角面
        
        Args:
            points: 点列
            vertices: 原始网格的顶点列表
            faces: 原始网格的面列表
            
        Returns:
            投影后的点列
        """
        print("将点列投影到原始网格中的三角面...")
        
        # 转换为numpy数组
        vertices_np = np.array(vertices)
        faces_np = np.array(faces) - 1  # 转换为0-based索引
        
        projected_points = []
        
        for point in points:
            point_np = np.array(point)
            min_distance = float('inf')
            closest_projection = None
            
            # 遍历所有三角面
            for face in faces_np:
                v0, v1, v2 = face
                p0 = vertices_np[v0]
                p1 = vertices_np[v1]
                p2 = vertices_np[v2]
                
                # 计算点到三角面的投影
                projection = self._point_triangle_projection(point_np, p0, p1, p2)
                
                # 计算投影点到原始点的距离
                distance = np.linalg.norm(projection - point_np)
                
                # 更新最近的投影
                if distance < min_distance:
                    min_distance = distance
                    closest_projection = projection
            
            if closest_projection is not None:
                projected_points.append(tuple(closest_projection))
            else:
                # 如果没有找到投影，使用原始点
                projected_points.append(point)
        
        print("点列投影完成")
        return projected_points
    
    def _point_triangle_projection(self, point, p0, p1, p2):
        """
        计算点到三角面的投影
        
        Args:
            point: 点坐标
            p0, p1, p2: 三角面的三个顶点
            
        Returns:
            投影点坐标
        """
        # 计算三角面的法向量
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        
        if normal_norm < 1e-6:
            # 三角面退化，返回最近的顶点
            distances = [np.linalg.norm(point - p0), np.linalg.norm(point - p1), np.linalg.norm(point - p2)]
            min_idx = np.argmin(distances)
            return [p0, p1, p2][min_idx]
        
        normal = normal / normal_norm
        
        # 计算点到三角面的距离
        distance = np.dot(normal, point - p0)
        
        # 计算投影点
        projection = point - distance * normal
        
        # 检查投影点是否在三角面内
        if self._point_in_triangle(projection, p0, p1, p2):
            return projection
        else:
            # 如果投影点在三角面外，计算到三角面边界的最近点
            edge1 = self._point_segment_projection(projection, p0, p1)
            edge2 = self._point_segment_projection(projection, p1, p2)
            edge3 = self._point_segment_projection(projection, p2, p0)
            
            distances = [np.linalg.norm(projection - edge1), np.linalg.norm(projection - edge2), np.linalg.norm(projection - edge3)]
            min_idx = np.argmin(distances)
            return [edge1, edge2, edge3][min_idx]
    
    def _point_in_triangle(self, point, p0, p1, p2):
        """
        检查点是否在三角面内
        
        Args:
            point: 点坐标
            p0, p1, p2: 三角面的三个顶点
            
        Returns:
            是否在三角面内
        """
        # 计算三个边的法向量
        v0 = p2 - p0
        v1 = p1 - p0
        v2 = point - p0
        
        # 计算点积
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        
        # 计算重心坐标
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        # 检查点是否在三角面内
        return (u >= 0) and (v >= 0) and (u + v <= 1)
    
    def _point_segment_projection(self, point, p0, p1):
        """
        计算点到线段的投影
        
        Args:
            point: 点坐标
            p0, p1: 线段的两个端点
            
        Returns:
            投影点坐标
        """
        v = p1 - p0
        v_norm = np.linalg.norm(v)
        
        if v_norm < 1e-6:
            return p0
        
        v = v / v_norm
        
        # 计算点到线段起点的向量
        w = point - p0
        
        # 计算点到线段的投影参数
        t = np.dot(w, v)
        t = max(0, min(t, v_norm))
        
        # 计算投影点
        projection = p0 + t * v
        return projection
    
    def visualize_mesh(self, vertices, faces, title="网格可视化", as_point_cloud=False, point_cloud_downsample=0.0, edge_points=None, new_vertices=None, new_vertices_labels=None):
        """
        可视化网格，使用黑色边缘，无填充颜色
        
        Args:
            vertices: 顶点列表
            faces: 面列表
            title: 可视化窗口标题
            as_point_cloud: 是否以点云形式可视化
            point_cloud_downsample: 点云下采样因子，0.0表示不采样
            edge_points: 边缘点列表，用黑色显示
            new_vertices: 新添加的顶点列表，用红色显示
            new_vertices_labels: 新添加的顶点对应的分区标签列表，用于设置不同颜色
        """
        print(f"可视化网格: {title}")
        
        import open3d as o3d
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title)
        
        # 添加坐标系
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(coord_frame)
        
        if as_point_cloud:
            # 以点云形式可视化
            print("以点云形式可视化...")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            
            # 下采样减少点的数量
            if point_cloud_downsample > 0:
                print(f"对点云进行下采样，采样因子: {point_cloud_downsample}")
                pcd = pcd.voxel_down_sample(voxel_size=point_cloud_downsample)
                print(f"下采样后点的数量: {len(pcd.points)}")
            
            # 设置点云颜色为灰白色
            pcd.paint_uniform_color([0.8, 0.8, 0.8])
            vis.add_geometry(pcd)
        else:
            # 以网格形式可视化
            # 创建Open3D网格对象
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            
            # 转换面索引为0-based
            faces_np = np.array(faces) - 1
            mesh.triangles = o3d.utility.Vector3iVector(faces_np)
            
            # 计算法线
            mesh.compute_vertex_normals()
            
            # 设置网格颜色为灰白色
            mesh.paint_uniform_color([0.8, 0.8, 0.8])
            
            # 添加网格
            vis.add_geometry(mesh)
            
            # 设置渲染选项 - 灰白色填充，黑色边缘
            render_option = vis.get_render_option()
            render_option.mesh_show_back_face = True
            render_option.mesh_show_wireframe = True
            render_option.line_width = 1.0
        
        # 可视化边缘点（黑色）
        if edge_points is not None and len(edge_points) > 0:
            print(f"可视化 {len(edge_points)} 个边缘点")
            edge_pcd = o3d.geometry.PointCloud()
            edge_pcd.points = o3d.utility.Vector3dVector(edge_points)
            edge_pcd.paint_uniform_color([0, 0, 0])  # 黑色
            vis.add_geometry(edge_pcd)
        
        # 可视化新添加的顶点（根据分区标签设置不同颜色）
        if new_vertices is not None and len(new_vertices) > 0:
            print(f"可视化 {len(new_vertices)} 个新添加的顶点")
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(new_vertices)
            
            if new_vertices_labels is not None and len(new_vertices_labels) == len(new_vertices):
                # 为不同分区的新顶点设置不同颜色
                unique_labels = np.unique(new_vertices_labels)
                num_labels = len(unique_labels)
                
                # 创建颜色映射
                colors = []
                for i in range(num_labels):
                    hue = (i * 0.618) % 1.0  # 黄金比例
                    r = (hue * 6.0) % 1.0
                    g = ((hue * 6.0) + 2.0) % 1.0
                    b = ((hue * 6.0) + 4.0) % 1.0
                    colors.append((r, g, b))
                
                label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
                
                # 为每个新顶点设置颜色
                vertex_colors = [label_to_color[label] for label in new_vertices_labels]
                new_pcd.colors = o3d.utility.Vector3dVector(vertex_colors)
            else:
                # 如果没有标签或标签长度不匹配，使用红色
                new_pcd.paint_uniform_color([1, 0, 0])  # 红色
            
            vis.add_geometry(new_pcd)
        
        # 设置背景颜色为白色
        render_option = vis.get_render_option()
        render_option.background_color = np.array([1, 1, 1])  # 白色背景
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        
        print("启动网格可视化窗口...")
        vis.run()
        vis.destroy_window()
        print("网格可视化完成")
    
    def reconstruct_mesh_from_point_cloud(self, point_cloud, output_path="reconstructed_mesh.obj", visualize=True):
        """
        通过点云重建三角网格
        
        Args:
            point_cloud: 点云数据，可以是numpy数组或Open3D PointCloud对象
            output_path: 输出OBJ文件路径
            visualize: 是否可视化重建结果
            
        Returns:
            str: 生成的OBJ文件路径
        """
        print("从点云重建三角网格...")
        
        import open3d as o3d
        
        # 转换为Open3D PointCloud对象
        if isinstance(point_cloud, np.ndarray):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
        elif isinstance(point_cloud, o3d.geometry.PointCloud):
            pcd = point_cloud
        else:
            raise ValueError("point_cloud参数必须是numpy数组或Open3D PointCloud对象")
        
        # 执行Delaunay三角剖分重建三角面
        print("执行Delaunay三角剖分...")
        # 首先创建一个凸包，然后进行三角剖分
        hull, _ = pcd.compute_convex_hull()
        mesh = hull
        
        # 保存重建结果
        print(f"保存重建结果到: {output_path}")
        o3d.io.write_triangle_mesh(output_path, mesh)
        
        # 可视化重建结果
        if visualize:
            print("可视化重建结果...")
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="重建网格可视化")
            
            # 添加坐标系
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            vis.add_geometry(coord_frame)
            
            # 添加重建的网格
            mesh.paint_uniform_color([1, 1, 1])
            vis.add_geometry(mesh)
            
            # 设置渲染选项
            render_option = vis.get_render_option()
            render_option.mesh_show_back_face = True
            render_option.mesh_show_wireframe = True
            render_option.line_width = 1.0
            render_option.background_color = np.array([1, 1, 1])  # 白色背景
            
            # 设置视角
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            
            vis.run()
            vis.destroy_window()
        
        print("点云重建完成")
        return output_path
