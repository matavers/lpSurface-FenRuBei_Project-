import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.nurbsProcessor import NURBSProcessor

# 创建圆柱面
cylinder = NURBSProcessor.create_cylinder(radius=1.0, height=2.0)

# 计算曲面上的点
point = cylinder.evaluate(u=0.5, v=0.5)
print(f"点坐标: {point}")

# 计算高斯曲率
gaussian_curvature = cylinder.calculate_gaussian_curvature(u=0.5, v=0.5)
print(f"高斯曲率: {gaussian_curvature}")

# 可视化曲面
cylinder.visualize(resolution_u=50, resolution_v=20)

# 保存 NURBS 数据
cylinder.save_nurbs_data("cylinder.npz")

# 加载 NURBS 数据
loaded_cylinder = NURBSProcessor.load_nurbs_data("cylinder.npz")