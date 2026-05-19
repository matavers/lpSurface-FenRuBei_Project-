"""
Quick test of PyVista visualization only
"""
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry_generators import generate_cylinder
from tests.visualizer_pyvista import create_pyvista_mesh

print("Testing PyVista mesh creation...")
trimesh_mesh = generate_cylinder()

vertices = np.array(trimesh_mesh.vertices)
faces = np.array(trimesh_mesh.faces)

print(f"Vertices: {vertices.shape}")
print(f"Faces: {faces.shape}")

mesh = create_pyvista_mesh(vertices, faces)

if mesh is not None:
    print(f"Success! Mesh created.")
    print(f"Points: {mesh.n_points}")
    print(f"Faces: {mesh.n_cells}")
    print(f"Bounds: {mesh.bounds}")
    
    # 快速检查是否有效
    import pyvista as pv
    print(f"Mesh valid: {pv.is_valid}")
    
    # 快速显示检查
    print("\nDisplaying mesh for quick check...")
    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
    plotter.add_mesh(mesh, color='lightgray', show_edges=True)
    plotter.show(screenshot="debug_mesh.png", auto_close=False)
    print("Screenshot saved as 'debug_mesh.png'")
    plotter.close()
    print("Debug test complete.")
