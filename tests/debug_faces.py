"""
Debug: test face array shape
"""
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_generators import generate_cylinder

print("Testing faces array...")
trimesh_mesh = generate_cylinder()

vertices = np.array(trimesh_mesh.vertices)
faces = np.array(trimesh_mesh.faces)

print(f"Vertices shape: {vertices.shape}")
print(f"Faces shape: {faces.shape}")
print(f"Faces:\n{faces[:5]}...")

# 正确的 PyVista 格式
n_faces = len(faces)
print(f"\nNumber of faces: {n_faces}")

# 每个面需要 [3, v0, v1, v2]
# 方法 1: 正确的拼接方式
face_count = np.full(n_faces, 3, dtype=np.int32)
print(f"Face count shape: {face_count.shape}")

# 逐个拼接
pv_faces_list = []
for i in range(n_faces):
    pv_faces_list.append(3)
    pv_faces_list.extend(faces[i])
pv_faces = np.array(pv_faces_list, dtype=np.int32)
print(f"\nMethod 1 (list): pv_faces shape: {pv_faces.shape}")
print(f"pv_faces[:12]: {pv_faces[:12]}")

# 方法 2: 用 NumPy
pv_faces_np = np.empty((n_faces, 4), dtype=np.int32)
pv_faces_np[:, 0] = 3
pv_faces_np[:, 1:] = faces
pv_faces_np = pv_faces_np.flatten()
print(f"\nMethod 2 (NumPy reshape): pv_faces shape: {pv_faces_np.shape}")
print(f"pv_faces[:12]: {pv_faces_np[:12]}")
print(f"\nBoth methods produce same result: {np.array_equal(pv_faces, pv_faces_np)}")

print("\nDebug complete.")
