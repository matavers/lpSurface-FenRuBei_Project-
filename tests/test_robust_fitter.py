"""
稳健直纹面拟合测试脚本
目录结构预期:
  ./new/robustRuledFitter.py
  ./new/newPartitoner.py
  ./core/meshProcessor.py
  ./tests/test_robust_fitter.py
"""

import sys
import os
import warnings
import numpy as np
import argparse

# ==========================================
# 1. 解决目录兼容性：将根目录加入 Python 路径
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("警告: PyVista不可用，将只输出终端日志，不进行三维可视化。")

try:
    from new.newPartitoner import NewPartitioner
    from new.robustRuledFitter import RobustRuledSurfaceFitter
    from tests.test_geodesic_boundaries import create_test_surface
except ImportError as e:
    print(f"导入失败，请检查目录结构: {e}")
    sys.exit(1)


# ==========================================
# 2. 为原位投影编写的全局拓扑渲染函数
# ==========================================
def visualize_robust_patches(
    vertices: np.ndarray, 
    faces: np.ndarray, 
    surface_grids: list, 
    screenshot_path: str = None, 
    interactive: bool = False
):
    """
    可视化原网格与拟合出的独立直纹面片（原位投影模式）
    :param surface_grids: 包含 {'indices': [...], 'points': [...]} 的列表
    """
    if not PYVISTA_AVAILABLE:
        return

    plotter = pv.Plotter(off_screen=not interactive)

    # 1. 创建一个拷贝，用于存放变成直纹面的新网格
    new_vertices = vertices.copy()
    
    # 将每个分区平滑后的顶点，写回到总顶点数组中
    for patch in surface_grids:
        idx_list = patch['indices']
        smoothed_pts = patch['points']
        new_vertices[idx_list] = smoothed_pts

    # 2. 渲染原始网格 (半透明白色，作为对比用的幽灵底模)
    old_mesh = pv.PolyData(vertices, np.hstack([np.full((len(faces), 1), 3), faces]))
    plotter.add_mesh(old_mesh, opacity=0.15, color='white', style='surface')

    # 3. 渲染平滑成直纹面后的新网格
    new_mesh = pv.PolyData(new_vertices, np.hstack([np.full((len(faces), 1), 3), faces]))
    
    plotter.add_mesh(
        new_mesh,
        color='lightblue',
        opacity=1.0,
        smooth_shading=True,  # 开启平滑着色，直纹面的反光会非常丝滑
        show_edges=True,      # 打开网格线，可以看到原始拓扑结构
        edge_color='gray'
    )

    plotter.add_axes()
    plotter.add_title("In-situ Ruled Surface Projection (Smooth & Trimmed)", font_size=12)

    if screenshot_path and interactive:
        print("弹出交互窗口，请旋转查看...")
        plotter.show(screenshot=screenshot_path, auto_close=False)
        print(f"截图已保存至: {screenshot_path}")
    elif screenshot_path:
        plotter.screenshot(screenshot_path)
        print(f"截图已保存至: {screenshot_path}")
    elif interactive:
        print("弹出交互窗口，请旋转查看...")
        plotter.show()
    plotter.close()


# ==========================================
# 3. 核心测试主流程
# ==========================================
def run_robust_test(surface_type='cylinder', output_dir='test_output', interactive=True):
    print(f"\n{'='*60}")
    print(f"启动新算法测试 (Robust Ruled Fitter - In-situ Projection) - {surface_type}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    print("步骤 1: 生成/加载测试网格...")
    nurbs, mesh_processor = create_test_surface(surface_type)
    vertices = np.asarray(mesh_processor.vertices)
    print(f"  > 顶点数: {len(vertices)}, 面数: {len(mesh_processor.faces)}")

    print("\n步骤 2: 执行区域生长分区...")
    partitioner = NewPartitioner(mesh_processor)
    
    edges = np.array(mesh_processor.edge_vertices)
    avg_edge_length = np.mean(np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1))
    target_partitions = 15
    partition_alpha = 0.8
    R_max = 38.0 * avg_edge_length
    theta_attr = 52.0

    optimized_benchmarks, regions_dict, _, _, _, _ = partitioner.partition_with_optimization(
        initial_num_benchmarks=target_partitions,
        alpha=partition_alpha,
        R_max=R_max,
        theta_attr=theta_attr,
        max_iterations=30,
    )
    
    partitions_ordered = [regions_dict[b] for b in optimized_benchmarks]
    print(f"  > 成功划分 {len(partitions_ordered)} 个子区域")

    print("\n步骤 3: 全局参数化线性拟合与原位投影...")
    
    fitter = RobustRuledSurfaceFitter(
        vertices=vertices, 
        degree=3, 
        num_ctrl_pts=8, 
        reg_weight=0.05
    )

    surface_grids = []
    valid_patches = 0

    for pid, region in enumerate(partitions_ordered):
        idx_list = list(region)
        
        if len(idx_list) < fitter.n_cp + 2:
            print(f"  > 分区 {pid}: 点数太少 ({len(idx_list)}), 忽略.")
            continue
            
        try:
            # 1. 拟合分区数学曲面
            fit_result = fitter.fit_partition(idx_list)
            
            # 2. 原位投影：将原顶点直接平滑吸附到该曲面上，完美继承原始边界形状
            smoothed_pts = fitter.project_original_vertices(fit_result)
            
            # 3. 保存原始索引和投影后的新坐标
            surface_grids.append({
                'indices': idx_list,
                'points': smoothed_pts
            })
            valid_patches += 1
            print(f"  > 分区 {pid}: 投影成功 (涵盖 {len(idx_list)} 个顶点)")
            
        except Exception as e:
            print(f"  > 分区 {pid}: 处理失败 -> {e}")

    print(f"  > 处理完成。共平滑投影 {valid_patches} 个直纹面区域。")

    print("\n步骤 4: 渲染三维结果...")
    screenshot_path = os.path.join(output_dir, f'robust_insitu_{surface_type}.png')
    visualize_robust_patches(
        vertices=vertices,
        faces=np.asarray(mesh_processor.faces),
        surface_grids=surface_grids,
        screenshot_path=screenshot_path,
        interactive=interactive
    )
    print(f"\n{'='*60}")
    print("测试完毕。")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass

    parser = argparse.ArgumentParser(description='稳健直纹面新算法测试 (原位投影裁剪)')
    parser.add_argument('--surface', type=str, default='cylinder', choices=['cylinder', 'cone', 'wavy'], help='测试曲面类型')
    parser.add_argument('--no_ui', action='store_true', help='关闭交互式弹窗')
    
    args = parser.parse_args()
    run_robust_test(surface_type=args.surface, interactive=not args.no_ui)