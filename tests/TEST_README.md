# 分区算法测试套件使用说明

## 概述

本测试套件用于验证新分区算法的功能，包括几何适应性、优化收敛性、分区重叠正确性等。

## 文件结构

```
tests/
├── geometry_generators.py  # 几何生成模块（圆柱、圆锥、波浪平面）
├── metrics.py               # 评价指标模块
├── visualizers.py           # 可视化模块
└── test_new_full_pipeline.py # 主测试脚本
```

## 安装依赖

```bash
pip install trimesh open3d numpy matplotlib scipy scikit-learn
```

## 运行测试

### 快速测试模式

```bash
cd tests
python test_new_full_pipeline.py --quick --shapes cylinder
```

### 完整测试模式

```bash
cd tests
python test_new_full_pipeline.py --shapes cylinder cone wavy_plane --output-dir test_results
```

### 参数说明

- `--output-dir`: 输出目录（默认: `test_output`）
- `--quick`: 快速模式，减少迭代次数
- `--shapes`: 测试形状列表（默认: `cylinder cone wavy_plane`）

## 输出结果

### 控制台输出

- 迭代过程：基准点数、未覆盖数、总重叠、平均覆盖
- 最终指标：未覆盖比例、平均覆盖、冗余基准点比例、各向异性等

### 输出文件

测试完成后，在输出目录下会生成：

1. **可视化结果**
   - `{shape}_fixed_colored_mesh.ply` - 固定基准点的着色网格
   - `{shape}_fixed_edge_midpoints.ply` - 固定基准点的边缘中点
   - `{shape}_optimized_colored_mesh.ply` - 优化后的着色网格
   - `{shape}_optimized_edge_midpoints.ply` - 优化后的边缘中点

2. **图表**
   - `{shape}_convergence.png` - 收敛曲线
   - `{shape}_alpha_sensitivity.png` - alpha参数敏感性
   - `{shape}_theta_sensitivity.png` - theta_attr参数敏感性

## 评价指标说明

### 几何一致性

- **分区各向异性比**：λ_max/λ_min（PCA计算）
  - 圆柱：> 5（沿轴向伸长）
  - 圆锥：> 3（沿母线）
  - 平面：≈ 1（各向同性）

- **边界平直度**：边界点到拟合直线的平均偏差（越小越平直）

### 覆盖与重叠

- **未覆盖顶点比例**：应为0
- **平均覆盖次数**：接近1（理想<1.2）
- **冗余基准点比例**：独有顶点为空的比例，应为0
- **总重叠**：Σ(c(v)-1)，越小越好

### 计算效率

- 总耗时
- 迭代次数

## 使用示例

### 生成几何并预处理

```python
from tests.geometry_generators import generate_cylinder
from core.meshProcessor import MeshProcessor
import open3d as o3d

# 生成几何
trimesh_mesh = generate_cylinder()

# 转换为Open3D格式并预处理
o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
mesh_processor = MeshProcessor(o3d_mesh)
```

### 单独运行分区

```python
from new.newPartitioner import NewPartitioner
from new.basePointDetermine import BasePointInitializer

partitioner = NewPartitioner(mesh_processor)

# 方式1: 带优化的完整流程
benchmarks, regions, coverage, vertex_to_partitions, edges, iter_data = \
    partitioner.partition_with_optimization(
        initial_num_benchmarks=20,
        max_iterations=50
    )

# 方式2: 使用固定基准点
initializer = BasePointInitializer(mesh_processor, 20)
fixed_benchmarks = initializer.sample('uniform')
partitions, vertex_to_partitions, edges = \
    partitioner.partition_surface(fixed_benchmarks)
```

### 自定义评价指标

```python
from tests.metrics import evaluate_full_partition

metrics = evaluate_full_partition(
    benchmarks, regions_dict, coverage,
    mesh_processor.vertices, edge_midpoints
)
```

## 测试要点

### 1. 几何适应性
- 圆柱：分区沿轴向呈长条形
- 圆锥：分区沿母线呈三角形
- 平面：分区近似圆形

### 2. 优化收敛
- 总重叠单调下降
- 未覆盖数快速降为0
- 基准点数趋于稳定

### 3. 分区重叠
- 重叠信息被正确保留
- 边界中点准确反映分区交界

## 常见问题

### 1. 依赖库问题
```bash
# 安装所有依赖
pip install -r requirements.txt
```

### 2. 内存问题
- 使用 `--quick` 模式减少计算量
- 调整网格生成参数减少顶点数

### 3. 可视化查看
```bash
# 使用MeshLab或CloudCompare查看PLY文件
meshlab test_output/cylinder_optimized_colored_mesh.ply
```
