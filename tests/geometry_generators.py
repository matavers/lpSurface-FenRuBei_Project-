
import numpy as np
from typing import Tuple, List
import trimesh


def generate_cylinder(
    num_u: int = 40,
    num_v: int = 40,
    radius: float = 1.0,
    height: float = 2.0
) -> trimesh.Trimesh:
    """
    生成圆柱网格
    
    Args:
        num_u: 环向细分数量
        num_v: 轴向细分数量
        radius: 圆柱半径
        height: 圆柱高度
        
    Returns:
        trimesh.Trimesh 对象
    """
    u = np.linspace(0, 2 * np.pi, num_u, endpoint=False)
    v = np.linspace(-height / 2, height / 2, num_v)
    
    U, V = np.meshgrid(u, v)
    X = radius * np.cos(U)
    Y = radius * np.sin(U)
    Z = V
    
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    faces = []
    for i in range(num_v - 1):
        for j in range(num_u - 1):
            idx1 = i * num_u + j
            idx2 = (i + 1) * num_u + j
            idx3 = i * num_u + j + 1
            faces.append([idx1, idx2, idx3])
            
            idx1 = i * num_u + j + 1
            idx2 = (i + 1) * num_u + j
            idx3 = (i + 1) * num_u + j + 1
            faces.append([idx1, idx2, idx3])
        
        idx1 = i * num_u + (num_u - 1)
        idx2 = (i + 1) * num_u + (num_u - 1)
        idx3 = i * num_u + 0
        faces.append([idx1, idx2, idx3])
        
        idx1 = i * num_u + 0
        idx2 = (i + 1) * num_u + (num_u - 1)
        idx3 = (i + 1) * num_u + 0
        faces.append([idx1, idx2, idx3])
    
    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    noise_level = 0.01 * radius
    mesh.vertices += np.random.normal(0, noise_level, mesh.vertices.shape)
    
    return mesh


def generate_cone(
    num_u: int = 40,
    num_v: int = 40,
    base_radius: float = 1.0,
    height: float = 2.0
) -> trimesh.Trimesh:
    """
    生成圆锥网格
    
    Args:
        num_u: 环向细分数量
        num_v: 轴向细分数量
        base_radius: 底面半径
        height: 圆锥高度
        
    Returns:
        trimesh.Trimesh 对象
    """
    u = np.linspace(0, 2 * np.pi, num_u, endpoint=False)
    v = np.linspace(0, height, num_v)
    
    U, V = np.meshgrid(u, v)
    scale = V / height
    X = base_radius * scale * np.cos(U)
    Y = base_radius * scale * np.sin(U)
    Z = V
    
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    faces = []
    for i in range(num_v - 1):
        for j in range(num_u - 1):
            idx1 = i * num_u + j
            idx2 = (i + 1) * num_u + j
            idx3 = i * num_u + j + 1
            faces.append([idx1, idx2, idx3])
            
            idx1 = i * num_u + j + 1
            idx2 = (i + 1) * num_u + j
            idx3 = (i + 1) * num_u + j + 1
            faces.append([idx1, idx2, idx3])
        
        idx1 = i * num_u + (num_u - 1)
        idx2 = (i + 1) * num_u + (num_u - 1)
        idx3 = i * num_u + 0
        faces.append([idx1, idx2, idx3])
        
        idx1 = i * num_u + 0
        idx2 = (i + 1) * num_u + (num_u - 1)
        idx3 = (i + 1) * num_u + 0
        faces.append([idx1, idx2, idx3])
    
    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    noise_level = 0.01 * base_radius
    mesh.vertices += np.random.normal(0, noise_level, mesh.vertices.shape)
    
    return mesh


def generate_wavy_plane(
    num_u: int = 50,
    num_v: int = 50,
    width: float = 2.0,
    height: float = 2.0,
    amplitude: float = 0.2
) -> trimesh.Trimesh:
    """
    生成带微小波动的平面网格（无拉伸扭曲）
    
    Args:
        num_u: x方向细分数量
        num_v: y方向细分数量
        width: x方向宽度
        height: y方向高度
        amplitude: 波动幅度
        
    Returns:
        trimesh.Trimesh 对象
    """
    u = np.linspace(-width / 2, width / 2, num_u)
    v = np.linspace(-height / 2, height / 2, num_v)
    
    U, V = np.meshgrid(u, v)
    X = U
    Y = V
    Z = amplitude * np.sin(2 * np.pi * X / width) * np.cos(2 * np.pi * Y / height)
    
    vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    faces = []
    for i in range(num_v - 1):
        for j in range(num_u - 1):
            idx1 = i * num_u + j
            idx2 = (i + 1) * num_u + j
            idx3 = i * num_u + j + 1
            faces.append([idx1, idx2, idx3])
            
            idx1 = i * num_u + j + 1
            idx2 = (i + 1) * num_u + j
            idx3 = (i + 1) * num_u + j + 1
            faces.append([idx1, idx2, idx3])
    
    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    return mesh


if __name__ == "__main__":
    print("Testing geometry generators...")
    
    cylinder = generate_cylinder()
    print(f"Cylinder: {len(cylinder.vertices)} vertices, {len(cylinder.faces)} faces")
    cylinder.export("test_cylinder.obj")
    
    cone = generate_cone()
    print(f"Cone: {len(cone.vertices)} vertices, {len(cone.faces)} faces")
    cone.export("test_cone.obj")
    
    plane = generate_wavy_plane()
    print(f"Wavy plane: {len(plane.vertices)} vertices, {len(plane.faces)} faces")
    plane.export("test_wavy_plane.obj")
    
    print("Test files generated!")
