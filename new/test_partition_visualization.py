import numpy as np
import open3d as o3d
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meshProcessor import MeshProcessor
from new.newPartitoner import NewPartitioner


def create_partition_colors(num_partitions):
    colors = []
    for i in range(num_partitions):
        hue = (i * 0.618) % 1.0
        r = abs(np.sin(hue * np.pi))
        g = abs(np.sin((hue + 0.33) * np.pi))
        b = abs(np.sin((hue + 0.66) * np.pi))
        colors.append([r, g, b])
    return colors


def generate_sphere_mesh(radius=1.0, resolution=20):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def partitions_to_labels(partitions, num_vertices):
    labels = np.zeros(num_vertices, dtype=int)
    for label_idx, region in enumerate(partitions):
        for vertex_idx in region:
            labels[vertex_idx] = label_idx
    return labels


def run_new_partitioner(sphere_radius=1.0, sphere_resolution=20, alpha=2.0, theta_attr=1.5):
    print("Generating sphere mesh...")
    mesh = generate_sphere_mesh(radius=sphere_radius, resolution=sphere_resolution)
    vertices = np.asarray(mesh.vertices)
    num_vertices = len(vertices)
    print(f"Vertices: {num_vertices}, Faces: {len(mesh.triangles)}")

    print("Creating MeshProcessor...")
    mesh_processor = MeshProcessor(mesh)

    print("Running partition...")
    partitioner = NewPartitioner(mesh_processor)
    partitions, edge_midpoints = partitioner.partition_surface(alpha=alpha, theta_attr=theta_attr)

    partition_labels = partitions_to_labels(partitions, num_vertices)

    print(f"Number of partitions: {len(partitions)}")
    print(f"Number of edge midpoints: {len(edge_midpoints)}")

    return vertices, partition_labels, edge_midpoints


def create_visualization(vertices, vertex_colors, edge_midpoints, show_boundary, point_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(vertex_colors)

    geometries = [pcd]

    if show_boundary and len(edge_midpoints) > 0:
        boundary_pcd = o3d.geometry.PointCloud()
        boundary_pcd.points = o3d.utility.Vector3dVector(edge_midpoints)
        boundary_pcd.paint_uniform_color([1, 0, 0])
        geometries.append(boundary_pcd)

    return geometries


def update_colors(vertices, partition_labels, partition_colors, selected_partitions, num_partitions):
    vertex_colors = np.zeros((len(vertices), 3))
    unselected_color = np.array([0.9, 0.9, 0.9])

    for i in range(len(vertices)):
        label = partition_labels[i]
        if selected_partitions is not None and label not in selected_partitions:
            vertex_colors[i] = unselected_color
        else:
            color_idx = label % len(partition_colors)
            vertex_colors[i] = partition_colors[color_idx]

    return vertex_colors


def main():
    print("=" * 50)
    print("Partition Visualization - NewPartitioner")
    print("=" * 50)

    sphere_radius = 1.0
    sphere_resolution = 20
    alpha = 2.0
    theta_attr = 1.5

    print("\nGenerating sphere and running partition...")
    vertices, partition_labels, edge_midpoints = run_new_partitioner(
        sphere_radius=sphere_radius,
        sphere_resolution=sphere_resolution,
        alpha=alpha,
        theta_attr=theta_attr
    )

    num_partitions = len(np.unique(partition_labels))
    print(f"\nUnique partitions: {num_partitions}")

    partition_colors = create_partition_colors(num_partitions)

    selected_partitions = None
    show_boundary = True
    point_size = 10.0

    while True:
        print("\n" + "=" * 50)
        print("Partition Selection Menu")
        print("=" * 50)
        print("Current selection: ", end="")
        if selected_partitions is None:
            print("All partitions")
        elif len(selected_partitions) == 1:
            print(f"Partition {selected_partitions[0]}")
        else:
            print(f"Partitions {selected_partitions[0]}-{selected_partitions[-1]}")
        print(f"Boundary display: {'ON' if show_boundary else 'OFF'}")
        print(f"Point size: {point_size}")
        print("=" * 50)
        print("Options:")
        print("  1 - Show all partitions")
        print("  2 - Select single partition")
        print("  3 - Select partition range")
        print("  4 - Toggle boundary points")
        print("  5 - Increase point size")
        print("  6 - Decrease point size")
        print("  7 - Visualize")
        print("  0 - Exit")
        print("=" * 50)
        print(f"\nTotal partitions: {num_partitions}")

        choice = input("\nEnter choice: ").strip()

        if choice == '0':
            print("Exiting...")
            break

        elif choice == '1':
            selected_partitions = None
            print("Showing all partitions")

        elif choice == '2':
            try:
                idx = int(input(f"Enter partition index (0-{num_partitions-1}): "))
                if 0 <= idx < num_partitions:
                    selected_partitions = [idx]
                    print(f"Selected partition {idx}")
                else:
                    print(f"Invalid index. Must be between 0 and {num_partitions-1}")
            except ValueError:
                print("Please enter a valid number")

        elif choice == '3':
            try:
                start = int(input(f"Enter start index (0-{num_partitions-1}): "))
                end = int(input(f"Enter end index (0-{num_partitions-1}): "))
                if 0 <= start <= end < num_partitions:
                    selected_partitions = list(range(start, end + 1))
                    print(f"Selected partitions {start} to {end}")
                else:
                    print(f"Invalid range. Must be 0 <= start <= end < {num_partitions}")
            except ValueError:
                print("Please enter valid numbers")

        elif choice == '4':
            show_boundary = not show_boundary
            print(f"Boundary display: {'ON' if show_boundary else 'OFF'}")

        elif choice == '5':
            point_size = min(50.0, point_size + 2.0)
            print(f"Point size: {point_size}")

        elif choice == '6':
            point_size = max(1.0, point_size - 2.0)
            print(f"Point size: {point_size}")

        elif choice == '7':
            print("\nCreating visualization...")

            vertex_colors = update_colors(vertices, partition_labels, partition_colors, selected_partitions, num_partitions)
            geometries = create_visualization(vertices, vertex_colors, edge_midpoints, show_boundary, point_size)

            title = "All Partitions" if selected_partitions is None else \
                    f"Partition {selected_partitions[0]}" if len(selected_partitions) == 1 else \
                    f"Partitions {selected_partitions[0]}-{selected_partitions[-1]}"

            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"Partition Visualization - {title}", width=1280, height=960)

            for geom in geometries:
                vis.add_geometry(geom)

            render_option = vis.get_render_option()
            render_option.background_color = np.array([1.0, 1.0, 1.0])
            render_option.point_size = point_size

            view_control = vis.get_view_control()
            view_control.set_lookat(np.array([0.0, 0.0, 0.0]))
            view_control.set_up(np.array([0.0, 1.0, 0.0]))
            view_control.set_front(np.array([3.0, 0.0, 0.0]))
            view_control.set_zoom(0.3)

            print(f"Displaying: {title}")
            print("Close visualization window to continue...")

            vis.run()
            vis.destroy_window()

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
