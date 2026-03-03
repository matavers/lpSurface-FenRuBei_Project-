# 五轴加工路径规划系统

## 项目简介

本项目是一个五轴加工路径规划系统，用于生成复杂曲面的高效加工路径。系统支持表面分区、工具方向场生成、等残留高度场计算、刀具路径规划、非球形刀具模型、G代码导出和加工动画可视化等功能，可用于叶轮、模具等复杂零件的加工。

**最新更新（2026年2月）**：
- 实现完整的非球形刀具模型，支持椭球形刀具
- 优化路径生成算法，支持等值线自动连接
- 实现完整的G代码导出功能，支持五轴加工
- 增强可视化工具，支持加工动画创建
- 性能优化，使用NumPy向量化操作加速计算

**最新更新（2026年3月）**：
- 实现基于曲率的网格生成算法，支持均匀采样和密度增加
- 实现边缘拟合算法，使用样条曲线拟合分区边缘
- 添加点云支持和点云重建功能
- 优化可视化显示，边缘显示为线条，移除颜色填充
- 新增测试脚本 `test_partition_edge_fitting.py`
- 添加分区结果的保存和加载功能
- 集成所有功能到主管道，增加分区大小

## 项目结构

```
├── core/                    # 核心算法模块
│   ├── __init__.py
│   ├── isoScallopField.py   # 等残留高度场生成
│   ├── meshProcessor.py     # 网格处理
│   ├── nonSphericalTool.py  # 非球面刀具模型
│   ├── pathGenerator.py     # 路径生成
│   ├── surfaceGenerator.py  # 曲面生成器
│   ├── surfacePartitioner.py # 表面分区
│   ├── tarCalculator.py     # TAR计算
│   └── toolOrientationField.py # 工具方向场生成
├── utils/                   # 工具函数
│   ├── __init__.py
│   ├── geometryTools.py     # 几何计算工具
│   ├── optimizationTools.py # 优化工具
│   ├── validation.py        # 加工验证工具
│   └── visualization.py     # 可视化工具
├── data/                    # 数据存储
├── config/                  # 配置文件
│   └── settings.json
├── tests/                   # 测试文件
│   ├── test_system.py       # 系统测试
│   ├── test_blade.py        # 叶轮刀片测试
│   ├── test_surface_generator.py # 曲面生成测试
│   ├── test_performance.py  # 性能测试
│   └── test_partition_edge_fitting.py # 分区边缘拟合测试
├── output/                  # 输出目录
├── main.py                  # 主程序入口
├── requirements.txt         # 依赖包
└── README.md                # 项目说明
```

## 安装与依赖

### 依赖包

项目依赖以下包：
- open3d
- numpy
- scipy
- networkx

**可选依赖**：
- matplotlib (用于可视化功能)
- CGAL (用于加速几何计算)
- community (用于Louvain聚类算法)
- leidenalg (用于Leiden聚类算法)

**注意**：系统会在可选依赖不可用时使用替代实现，确保核心功能正常运行。

### 安装方法

1. 克隆项目到本地
2. 可以在pyCharm中打开，会自动配置环境
3. （备用）手动安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 基本使用

1. 准备网格文件（支持.obj格式）
2. 运行主程序：
   ```bash
   python main.py path/to/mesh.obj
   ```

### 测试程序

tips:如果使用python虚拟环境，先激活虚拟环境
   ```bash
   .\venv\Scripts\activate
   ```

1. 运行系统测试（耗时10分钟以上）：
   ```bash
   python tests/test_system.py
   ```

2. 运行叶轮刀片测试：
   ```bash
   python tests/test_blade.py
   ```

3. 运行曲面生成测试：
   ```bash
   python tests/test_surface_generator.py
   ```

4. 运行分区边缘拟合测试：
   ```bash
   python tests/test_partition_edge_fitting.py
   ```

### 曲面生成器使用方法

曲面生成器用于生成自定义曲面的OBJ文件，可用于测试和验证加工路径规划。支持基于曲率的网格加密和点云功能。

```python
from core import SurfaceGenerator

# 创建曲面生成器
generator = SurfaceGenerator()

# 定义自定义函数
def custom_func(x, y):
    return 0.5 * (np.sin(5 * x) + np.cos(5 * y))

# 生成自定义曲面（支持密度因子参数）
generator.generate_surface(
    custom_func,
    resolution=(30, 30),
    bounds=((-1, 1), (-1, 1)),
    output_path="custom_surface.obj",
    density_factor=2.0  # 密度因子，控制网格加密程度
)

# 生成预设曲面（支持分辨率参数）
generator.generate_sphere(output_path="sphere.obj", resolution=50)
generator.generate_torus(output_path="torus.obj", resolution=30)
generator.generate_saddle(output_path="saddle.obj", resolution=(40, 40))

# 点云功能
# 从点云重建网格
point_cloud = generator.generate_point_cloud()
reconstructed_mesh = generator.reconstruct_mesh_from_point_cloud(point_cloud)

# 边缘拟合
# 拟合分区边缘为样条曲线
edge_points = [...]  # 边缘点列表
fitted_curve = generator.fit_edge_curve(edge_points)

# 在曲线上采样点
curve_points = generator.sample_curve(fitted_curve, num_points=100)

# 将点投影到原始网格
projected_points = generator.project_to_mesh(curve_points, original_mesh)
```

## 核心功能

1. **网格处理**：加载和处理3D网格模型，支持曲率估计和几何特征提取
2. **表面分区**：基于几何特征对表面进行分区，支持Louvain和Leiden聚类算法，支持权重优化
3. **工具方向场**：生成优化的工具方向，支持种子点选择和贪心TAR选择算法
4. **等残留高度场**：计算等残留高度的标量场，支持泊松方程求解和固定点迭代优化
5. **刀具路径生成**：生成高效的五轴加工路径，支持等值线自动连接和模拟退火优化
6. **非球形刀具模型**：支持椭球形、圆柱形、球形、锥形、自定义刀具，实现完整的碰撞检测和切削宽度计算
7. **加工验证**：检查刀具路径碰撞，计算残留高度，评估路径平滑度和方向稳定性
8. **G代码导出**：支持五轴加工的完整G代码生成，包括工具长度补偿和进给速度优化
9. **曲面生成**：生成自定义曲面的OBJ文件，支持多种预设曲面
10. **基于曲率的网格生成**：支持均匀采样和基于局部曲率的密度增加
11. **边缘拟合**：使用样条曲线拟合分区边缘，在曲线上采样并投影到原始网格
12. **点云支持**：支持点云可视化和点云重建功能
13. **分区结果保存和加载**：支持将分区结果保存到文件并加载
14. **可视化**：实时可视化分区、方向场和路径，支持加工动画创建，优化边缘显示为线条

## 系统配置

配置文件位于 `config/settings.json`，可根据需要调整参数：

- 刀具参数
- 分区参数
- 可视化设置
- 路径生成参数

## 输出结果

系统运行后在 `output/` 目录生成以下文件：

- `metrics.json`：加工路径的性能指标
- `tool_path_*.csv`：各分区的刀具路径

## 性能优化

系统使用以下技术提高性能：

1. **NumPy向量化**：使用NumPy向量化操作替代Python循环，显著提升计算速度
2. **数据结构优化**：使用高效的数据结构（如deque用于BFS）提高算法效率
3. **算法优化**：采用贪心算法、距离矩阵优化等高效算法
4. **并行计算**：支持多线程并行处理（可选）
5. **内存管理**：优化内存使用，减少不必要的内存分配

## 注意事项

1. 网格模型应尽量简化，以提高计算速度
2. 对于复杂模型，可能需要调整分区参数
3. 可视化功能需要Matplotlib库支持

## 测试结果

系统已在以下场景进行测试：

1. 球体模型（半径10mm，分辨率50）
2. 叶轮刀片模型
3. 复杂曲面模型
4. 自定义曲面模型（通过曲面生成器生成）

### 球体模型测试结果

**测试配置**：
- 网格：球体，半径10mm，4902个顶点，9800个面
- 刀具：椭球形，半轴[9.0, 3.0]
- 残留高度：0.4mm

**测试结果**：
- 总处理时间：1513.45秒（约25分钟）
- 分区数量：4901个
- 路径总长度：644.71mm
- 路径数量：43条
- 路径点数量：4254个

**验证结果**：
- 碰撞检测：失败（存在碰撞）
- 平均残留高度：0.019mm
- 最大残留高度：0.278mm
- 路径平滑度：0.102（0-1）
- 方向稳定性：0.865（0-1）

### 边缘拟合测试结果

**测试配置**：
- 网格：修改后的球体模型（`data/models/modified_sphere.obj`）
- 分区算法：Leiden聚类
- 边缘拟合：样条曲线

**测试结果**：
- 分区数量：根据模型复杂度自动调整
- 边缘拟合时间：< 1秒
- 可视化：边缘显示为黑色线条，原始模型为灰白色
- 分区结果：保存到 `data/models/partition_labels.npy`

### 加工验证功能

加工验证功能用于评估生成的刀具路径质量，包括：

1. **碰撞检测**：检查刀具路径是否与表面发生碰撞
2. **残留高度计算**：计算路径之间的残留高度，确保满足加工要求
3. **路径平滑度评估**：评估路径的平滑程度，减少加工振动
4. **方向稳定性评估**：评估工具方向的变化，确保加工稳定性

**使用方法**：

```python
from utils.validation import MachiningValidator

# 初始化验证器
validator = MachiningValidator(mesh_processor, tool)

# 运行验证
report = validator.generate_report(tool_paths)

# 打印验证报告
print(report['summary'])

# 可视化验证结果
validator.visualize_validation(tool_paths)
```

### 曲面生成器测试

曲面生成器可以生成以下类型的曲面：
- 自定义函数曲面
- 球体
- 圆环
- 马鞍面

生成的曲面可以直接用于加工路径规划测试。