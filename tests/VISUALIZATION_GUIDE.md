# 分区算法可视化指南

## 快速开始

### 1. 交互式演示

运行简单的可视化演示：

```bash
cd d:\Projects\lpSurface\GM
.venv\Scripts\activate
python .\tests\demo_visualize.py --shape cylinder
```

可选形状：`cylinder`, `cone`, `wavy_plane`

### 2. 完整测试

运行完整的测试并生成可视化结果：

```bash
# 快速测试
python .\tests\test_new_full_pipeline.py --quick --shapes cylinder

# 启用交互式可视化
python .\tests\test_new_full_pipeline.py --quick --shapes cylinder --interactive

# 完整测试所有形状
python .\tests\test_new_full_pipeline.py --shapes cylinder cone wavy_plane
```

## 可视化功能说明

### 1. 三维可视化

- **彩色网格** (.ply文件)：每个分区用不同颜色表示，基准点高亮红色
- **边界中点** (.ply文件)：绿色点显示分区边界
- **交互式可视化**：使用 Open3D 打开可旋转、缩放、平移

### 2. 二维投影

生成三个视图的二维投影：
- XY视图（俯视图）
- XZ视图（侧视图）
- YZ视图（前视图）

### 3. 统计图表

- **分区大小分布**：直方图显示各分区大小
- **覆盖统计**：显示未覆盖比例、平均覆盖、总重叠等

### 4. 摘要报告

生成文本文件包含所有测试指标的汇总。

## 输出文件结构

```
test_output/
├── cylinder_fixed/
│   ├── cylinder_fixed_colored_mesh.ply          # 彩色网格（固定基准点）
│   ├── cylinder_fixed_edge_midpoints.ply         # 边界点（固定基准点）
│   ├── cylinder_fixed_partition_sizes.png       # 分区大小分布
│   ├── cylinder_fixed_coverage_stats.png        # 覆盖统计
│   ├── cylinder_fixed_2d_xy.png                 # XY投影
│   ├── cylinder_fixed_2d_xz.png                 # XZ投影
│   ├── cylinder_fixed_2d_yz.png                 # YZ投影
│   └── cylinder_fixed_report.txt                # 摘要报告
│
├── cylinder_optimized/
│   ├── cylinder_optimized_colored_mesh.ply      # 彩色网格（优化后）
│   ├── cylinder_optimized_edge_midpoints.ply     # 边界点（优化后）
│   ├── ... (其他文件同上)
│
├── cylinder_convergence.png                     # 收敛曲线
├── cylinder_alpha_sensitivity.png               # alpha参数敏感性
└── cylinder_theta_sensitivity.png               # theta参数敏感性
```

## 查看 PLY 文件

### 方法1：使用 MeshLab

下载安装 [MeshLab](https://www.meshlab.net/)，直接打开 .ply 文件。

### 方法2：使用 Open3D（Python）

```python
import open3d as o3d

# 查看彩色网格
mesh = o3d.io.read_triangle_mesh("test_output/cylinder_optimized_colored_mesh.ply")
o3d.visualization.draw_geometries([mesh])

# 查看边界点
pcd = o3d.io.read_point_cloud("test_output/cylinder_optimized_edge_midpoints.ply")
o3d.visualization.draw_geometries([pcd])

# 同时查看
mesh = o3d.io.read_triangle_mesh("test_output/cylinder_optimized_colored_mesh.ply")
pcd = o3d.io.read_point_cloud("test_output/cylinder_optimized_edge_midpoints.ply")
o3d.visualization.draw_geometries([mesh, pcd])
```

### 方法3：使用 CloudCompare

下载安装 [CloudCompare](https://www.danielgm.net/cc/)，功能更强大。

## 测试指标说明

### 覆盖指标

- **未覆盖比例**：未被任何分区覆盖的顶点百分比（理想值为 0%）
- **平均覆盖次数**：每个顶点被分区覆盖的平均次数（理想值为 1.0）
- **总重叠**：所有顶点的（覆盖次数 - 1）之和（越小越好）
- **最大覆盖**：单个顶点被覆盖的最大次数

### 形状指标

- **各向异性**：分区的长宽比（圆柱应大于5，平面接近1）
- **边界平直度**：边界点到拟合直线的平均偏差（越小越直）

### 基准点指标

- **基准点数量**：最终使用的基准点数量
- **冗余基准点比例**：可以被删除而不影响覆盖的基准点百分比

## 常见问题

### 1. 分区太小

**原因**：theta_attr 太小或 alpha 太大导致有效距离膨胀

**解决**：
- 增大 theta_attr（如从15度增至30度）
- 减小 alpha（如从2.0减至0.0）
- 增大 R_max（如从10倍增至20倍平均边长）

### 2. 分区太大且重叠严重

**原因**：theta_attr 太大或 R_max 太大

**解决**：
- 减小 theta_attr
- 减小 R_max
- 增加初始基准点数量

### 3. 未覆盖顶点较多

**原因**：theta_attr 太小，某些顶点无法加入任何分区

**解决**：
- 增大 theta_attr
- 增加更多基准点
- 运行优化过程（分区_with_optimization）

## 自定义使用

### 在自己的代码中使用可视化

```python
from tests.visualizer_enhanced import save_complete_visualization, visualize_interactive

# 保存可视化结果
save_complete_visualization(
    mesh=your_mesh,
    partitions=your_partitions,
    vertex_to_partitions=vertex_to_partitions_map,
    edge_midpoints=edge_midpoints,
    benchmarks=benchmark_indices,
    coverage=coverage_array,
    output_prefix="my_output",
    interactive=True  # 是否显示交互式窗口
)

# 只显示交互式可视化
visualize_interactive(
    mesh=your_mesh,
    vertex_to_partitions=vertex_to_partitions_map,
    edge_midpoints=edge_midpoints,
    benchmarks=benchmark_indices,
    window_name="我的可视化"
)
```

## 参数参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| alpha | 0.0 | 曲率拉伸强度（0=各向同性） |
| theta_attr | 30.0 | 法向夹角阈值（度） |
| R_max | 10倍平均边长 | 最大有效距离 |
| initial_num_benchmarks | 20 | 初始基准点数量 |
| max_iterations | 50 | 最大迭代次数 |

## 推荐参数配置

### 圆柱/圆锥

```python
alpha=0.0
theta_attr=30.0
initial_num_benchmarks=15
```

### 无拉伸平面

```python
alpha=0.0
theta_attr=20.0
initial_num_benchmarks=20
```

### 复杂自由曲面

```python
alpha=1.0
theta_attr=25.0
initial_num_benchmarks=30
max_iterations=100
```
