# 五轴加工路径规划系统

## 项目简介

本项目是一个五轴加工路径规划系统，用于生成复杂曲面的高效加工路径。系统支持表面分区、工具方向场生成、等残留高度场计算、刀具路径规划、非球形刀具模型、G代码导出和加工动画可视化等功能，可用于叶轮、模具等复杂零件的加工。

**最新更新（2026年3月）**：
- 实现基于算法version2的高级表面分区器，使用Leiden聚类算法
- 基于新指标的分区算法：局部曲率相似性、最大切削宽度差异、直纹面逼近误差
- 优化可视化逻辑，在一个窗口中显示颜色块标示的分区和中点
- 添加分区边缘中点提取功能
- 修复各种错误，确保程序稳定运行
- 优化性能，使用并行计算加速邻接矩阵构建
- 添加run_partition_only模式，只运行分区并保存数据
- 更新命令行参数，使用--path参数指定OBJ文件路径
- 添加网格可视化功能，在执行核心算法前可视化当前操作的网格
- 优化参数验证逻辑，确保参数组合的正确性
- 实现直纹面拟合功能，根据逼近误差阈值确定直纹面类型
- 添加--developable-fit和--developable-error命令行参数，控制直纹面拟合功能

## 项目结构

```
├── core/                    # 核心算法模块
│   ├── __init__.py
│   ├── advancedSurfacePartitioner.py # 基于新指标的高级表面分区器
│   ├── isoScallopField.py   # 等残留高度场生成
│   ├── meshProcessor.py     # 网格处理
│   ├── nonSphericalTool.py  # 非球面刀具模型
│   ├── pathGenerator.py     # 路径生成
│   ├── surfaceGenerator.py  # 曲面生成器
│   ├── toolOrientationField.py # 工具方向场生成
├── utils/                   # 工具函数
│   ├── __init__.py
│   ├── geometryTools.py     # 几何计算工具
│   ├── validation.py        # 加工验证工具
│   ├── visualization.py     # 可视化工具
│   └── visualize_results.py # 结果可视化工具
├── data/                    # 数据存储
│   └── models/              # 模型文件
├── config/                  # 配置文件
│   └── settings.json
├── tests/                   # 测试文件
│   ├── test_system.py       # 系统测试
│   ├── test_edge_midpoint.py # 边缘中点测试
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
- igraph (用于Leiden聚类算法)

**注意**：系统会在可选依赖不可用时使用替代实现，确保核心功能正常运行。

### 安装方法

1. 克隆项目到本地
2. 可以在pyCharm中打开，会自动配置环境
3. （备用）手动安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

### GPU加速配置步骤

1. 安装CUDA 12.8：
   - 从NVIDIA官网下载并安装CUDA 12.8 Toolkit
   - 确保系统环境变量已正确设置

2. 安装PyTorch（CUDA 12.8版本）：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

3. 验证GPU加速：
   ```python
   import torch
   print(torch.cuda.is_available())  # 应返回True
   print(torch.cuda.get_device_name(0))  # 应显示GPU名称
   ```

4. 调整批处理大小以优化GPU内存使用：
   - 打开`config/settings.json`文件
   - 修改`neural_network.batch_size`参数
   - 对于RTX 5060 Laptop，建议batch_size设置为32-64

## 使用方法

### 基本使用

1. 准备网格文件（支持.obj格式）
2. 运行主程序：
   ```bash
   # 使用曲面函数生成网格
   python main.py --surface=sphere
   
   # 直接使用OBJ文件
   python main.py --mesh-algorithm=obj --path=test_sphere.obj
   ```

3. 只运行分区并保存数据：
   ```bash
   # 使用曲面函数
   python main.py --surface=sphere --partition-only
   
   # 直接使用OBJ文件
   python main.py --mesh-algorithm=obj --path=test_sphere.obj --partition-only
   ```

### 命令行参数

系统支持以下命令行参数：

| 参数 | 描述 | 可选值 | 默认值 |
|------|------|--------|--------|
| `--partition-only` | 只运行分区并保存数据，不执行刀具路径规划 | - | 无 |
| `--mesh-algorithm` | 网格生成算法 | delaunay_cocone, bpa, poisson, tsdf, obj | delaunay_cocone |
| `--surface` | 曲面函数名称 | sphere, torus, saddle | 无 |
| `--resolution` | 曲面采样分辨率 | 整数 | 50 |
| `--path` | OBJ文件路径（仅与--mesh-algorithm=obj共存） | 文件路径 | 无 |
| `--developable-fit` | 开启直纹面逼近功能，不计算刀具路径 | - | 无 |
| `--developable-error` | 直纹面逼近误差阈值 | 浮点数 | 0.01 |

### 参数验证规则

- `--path` 只能与 `--mesh-algorithm=obj` 共存
- `--surface` 只能与非 obj 算法共存
- 当 `--mesh-algorithm=obj` 时，必须指定 `--path`
- 当使用 `--surface` 时，不能指定 `--path`
- 必须指定 `--path` 或 `--surface` 中的一个
- 如果指定 `--developable-fit`，则开启直纹面逼近功能，不计算刀具路径
- 如果计算刀具路径，则不逼近直纹面

### 使用示例

1. 使用默认算法和曲面函数：
   ```bash
   python main.py --surface=sphere
   ```

2. 使用指定算法和曲面函数：
   ```bash
   python main.py --surface=sphere --mesh-algorithm=bpa --resolution=50
   ```

3. 直接使用OBJ文件（不进行重建）：
   ```bash
   python main.py --mesh-algorithm=obj --path=test_sphere.obj
   ```

4. 只运行分区（使用曲面函数）：
   ```bash
   python main.py --surface=sphere --partition-only
   ```

5. 只运行分区（直接使用OBJ文件）：
   ```bash
   python main.py --mesh-algorithm=obj --path=test_sphere.obj --partition-only
   ```

6. 综合使用多个参数：
   ```bash
   python main.py --surface=torus --mesh-algorithm=poisson --resolution=100 --partition-only
   ```

7. 开启直纹面拟合功能：
   ```bash
   python main.py --surface=sphere --developable-fit --developable-error=0.01
   ```

8. 开启直纹面拟合功能（使用OBJ文件）：
   ```bash
   python main.py --mesh-algorithm=obj --path=test_sphere.obj --developable-fit --developable-error=0.005
   ```

### 测试程序

**tips**:如果使用python虚拟环境，先激活虚拟环境
   ```bash
   .\venv\Scripts\activate
   ```

1. 运行系统测试：
   ```bash
   python tests/test_system.py
   ```

2. 运行边缘中点测试：
   ```bash
   python tests/test_edge_midpoint.py
   ```

3. 运行分区边缘拟合测试：
   ```bash
   python tests/test_partition_edge_fitting.py
   ```

### 结果可视化

使用可视化工具查看结果：
```bash
python utils/visualize_results.py
```

该工具提供了一个GUI界面，允许用户浏览output目录下的所有计算结果文件夹，并选择特定的结果文件夹进行可视化。

## 核心功能

1. **网格处理**：加载和处理3D网格模型，支持曲率估计和几何特征提取
2. **表面分区**：基于新指标的高级表面分区器，使用Leiden聚类算法
   - 局部曲率相似性
   - 最大切削宽度差异
   - 直纹面逼近误差
3. **工具方向场**：生成优化的工具方向，支持种子点选择和贪心TAR选择算法
4. **等残留高度场**：计算等残留高度的标量场，支持泊松方程求解和固定点迭代优化
5. **刀具路径生成**：生成高效的五轴加工路径，支持等值线自动连接和模拟退火优化
6. **非球形刀具模型**：支持椭球形、圆柱形、球形、锥形、自定义刀具，实现完整的碰撞检测和切削宽度计算
7. **加工验证**：检查刀具路径碰撞，计算残留高度，评估路径平滑度和方向稳定性
8. **G代码导出**：支持五轴加工的完整G代码生成，包括工具长度补偿和进给速度优化
9. **曲面生成**：生成自定义曲面的OBJ文件，支持多种预设曲面
10. **边缘中点提取**：提取分区边缘的中点，用于可视化和后续处理
11. **可视化**：在一个窗口中显示颜色块标示的分区和中点，优化边缘显示
12. **网格可视化**：在执行核心算法前可视化当前操作的网格，确认采样效果
13. **分区结果保存和加载**：支持将分区结果保存到文件并加载
14. **神经网络直纹面拟合**：基于PointNet++和编码器-解码器架构的直纹面拟合，支持三角形和四边形分区
15. **GPU加速**：利用NVIDIA GeForce RTX 5060 Laptop GPU进行并行计算，提高拟合速度

## 系统配置

配置文件位于 `config/settings.json`，可根据需要调整参数：

- 刀具参数
- 分区参数（resolution参数控制分区数量）
- 可视化设置
- 路径生成参数
- 神经网络参数（batch_size, learning_rate等）

## GPU环境要求

### CUDA版本兼容性
- **推荐CUDA版本**：12.8
- **GPU要求**：NVIDIA GeForce RTX 5060 Laptop或更高
- **驱动版本**：32.0.15.7324或更高

### 性能优化建议
- 确保GPU驱动程序已更新至最新版本
- 调整batch_size参数以充分利用GPU内存
- 对于大型模型，考虑使用混合精度训练
- 在推理时启用CUDA推理加速

## 神经网络直纹面拟合

### 架构概述
- **编码器**：使用PointNet++处理内部点云，一维卷积处理边缘点列
- **解码器**：三层全连接网络生成B样条曲线控制点
- **损失函数**：Chamfer距离、边缘约束、平滑项、端点约束

### 输入数据格式
- **内部点云**：无序三维点集（1000-5000点）
- **边缘点列**：每条边的有序点序列，附带相邻分区标记
- **分区类型**：三角形或四边形（one-hot向量）

### 与原有系统的接口方式
- 保持原有`fit_developable_surfaces`函数接口不变
- 内部自动切换到神经网络模型进行拟合
- 支持回退到传统算法（当神经网络拟合失败时）

## 输出结果

系统运行后在 `output/` 目录生成以下文件：

- `metrics.json`：加工路径的性能指标
- `partition_labels.npy`：分区标签
- `edge_midpoints.npy`：边缘中点
- `vertices.npy`：网格顶点
- `triangles.npy`：网格三角形
- `edge_points.npy`：边缘点
- `tool_path_*.csv`：各分区的刀具路径
- `developable_surfaces.json`：直纹面拟合结果（当开启直纹面拟合时）

## 性能优化

系统使用以下技术提高性能：

1. **NumPy向量化**：使用NumPy向量化操作替代Python循环，显著提升计算速度
2. **并行计算**：使用ThreadPoolExecutor并行处理邻接矩阵构建
3. **数据结构优化**：使用高效的数据结构提高算法效率
4. **算法优化**：采用Leiden聚类算法，提供更好的分区质量和性能
5. **内存管理**：优化内存使用，减少不必要的内存分配

## 注意事项

1. 网格模型应尽量简化，以提高计算速度
2. 对于复杂模型，可能需要调整分区参数（resolution）
3. 可视化功能需要Matplotlib库支持
4. Leiden聚类算法需要leidenalg和igraph库支持

## 测试结果

系统已在以下场景进行测试：

1. 球体模型（半径10mm，分辨率50）
2. 复杂曲面模型
3. 自定义曲面模型（通过曲面生成器生成）

### 球体模型测试结果

**测试配置**：
- 网格：球体，半径10mm，4902个顶点，9800个面
- 刀具：椭球形，半轴[9.0, 3.0]
- 残留高度：0.4mm
- 分区算法：Leiden聚类，resolution=0.05

**测试结果**：
- 总处理时间：约25分钟
- 分区数量：约380个
- 路径总长度：约640mm
- 路径数量：约40条
- 路径点数量：约4200个

### 分区边缘拟合测试结果

**测试配置**：
- 网格：修改后的球体模型（`data/models/modified_sphere.obj`）
- 分区算法：Leiden聚类

**测试结果**：
- 分区数量：根据模型复杂度自动调整
- 边缘中点提取：约5000个中点
- 可视化：在一个窗口中显示颜色块标示的分区和红色中点
- 分区结果：保存到 `output/[timestamp]/partition_labels.npy`

## 高级功能

### 自定义分区参数

可以通过修改配置文件或命令行参数调整分区参数：

```python
# 在main.py中修改分区参数
system.config['algorithm']['partition_resolution'] = 0.05  # 降低分辨率，减少分区数量
```

### 并行计算

系统默认使用并行计算加速邻接矩阵构建，可以通过调整batch_size参数优化性能：

```python
# 在advancedSurfacePartitioner.py中修改批处理大小
batch_size = 1000  # 调整批处理大小
```

### 结果可视化

可视化工具提供了以下功能：
- 浏览和选择结果文件夹
- 可视化分区、方向场、刀具路径
- 支持跳过点云可视化以提高性能
- 显示边缘中点和分区边缘

## 故障排除

1. **Leiden聚类失败**：如果leidenalg库未安装，系统会自动使用Louvain聚类作为替代
2. **可视化窗口无响应**：尝试勾选"跳过点云可视化"复选框
3. **分区数量过多**：调整resolution参数，增大值以减少分区数量
4. **内存不足**：尝试使用简化的网格模型

## 未来计划

1. 支持更多类型的刀具模型
2. 实现更高级的路径优化算法
3. 添加更多的加工验证功能
4. 支持更多的文件格式
5. 提供更详细的用户界面

## 许可证

本项目为开源项目，采用MIT许可证。
