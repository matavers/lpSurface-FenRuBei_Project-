# 五轴加工路径规划系统

## 算法流程

本系统实现了从网格到刀具路径的完整流程，包括以下步骤：

1. **网格加载/生成**：加载OBJ文件或生成参数化网格（如圆锥面）
2. **几何属性计算**：计算高斯曲率、主曲率和法向量
3. **对称性检测**：检测旋转、平移、反射和螺旋对称性
4. **表面分区**：基于对称性和几何指标进行分区
5. **刀具方向场生成**：为每个顶点生成最优刀具方向
6. **等残留高度场生成**：计算等残留高度的标量场
7. **等值线提取**：从标量场提取等值线
8. **刀具路径生成**：使用PathGenerator生成最终刀具路径
9. **可视化与结果保存**：可视化分区、方向场和刀具路径，保存结果

## 核心组件

### 1. NURBS曲面处理器 (`utils/nurbsSurfaceProcessor.py`)

**功能**：
- 创建测试NURBS曲面
- 计算曲面上任意点的坐标
- 计算法向量
- 计算高斯曲率和主曲率
- 从曲面采样点云（支持均匀采样和自适应采样）
- 将点云转换为三角网格

**使用示例**：
```python
from utils.nurbsSurfaceProcessor import NURBSSurfaceProcessor

# 创建处理器
processor = NURBSSurfaceProcessor()

# 创建测试曲面
surface = processor.create_test_surface()

# 采样点云
points, normals, curvatures, principal_curvatures = processor.sample_points(
    resolution_u=30,
    resolution_v=30,
    adaptive=True,
    curvature_threshold=0.1
)

# 创建网格
mesh = processor.create_mesh()
```

### 2. 指标计算器 (`core/indicatorCalculator.py`)

**功能**：
- 计算TAR（Tool Accessible Region）
- 计算高斯曲率相似性
- 计算几何连续性相似性
- 计算直纹面逼近误差相似性
- 计算综合相似性

**使用示例**：
```python
from core.indicatorCalculator import IndicatorCalculator

# 创建指标计算器
calculator = IndicatorCalculator(mesh_processor, tool)

# 计算相似性指标
gaussian_sim = calculator.calculate_gaussian_curvature_similarity(0, 1)
geometric_sim = calculator.calculate_geometric_continuity_similarity(0, 1)
developable_sim = calculator.calculate_developable_surface_error_similarity(0, 1)
combined_sim = calculator.calculate_combined_similarity(0, 1)
```

### 3. 高级表面分区器 (`core/advancedSurfacePartitioner.py`)

**功能**：
- 构建基于新指标的加权邻接矩阵
- 执行Leiden聚类分区
- 应用对称性约束
- 拟合分区边界
- 执行二次分区（可选）
- 确保分区连通性

**使用示例**：
```python
from core.advancedSurfacePartitioner import AdvancedSurfacePartitioner

# 创建分区器
partitioner = AdvancedSurfacePartitioner(
    mesh_processor,
    tool,
    resolution=0.1,
    enable_secondary_partitioning=True  # 启用二次分区
)

# 执行分区
labels, edge_midpoints = partitioner.partition_surface()
```

### 4. 路径生成器 (`core/pathGenerator.py`)

**功能**：
- 连接等值线形成连续路径
- 优化路径序列
- 计算刀位点
- 生成最终刀具路径
- 导出为G代码

**使用示例**：
```python
from core.pathGenerator import PathGenerator

# 创建路径生成器
path_generator = PathGenerator(mesh_processor, iso_curves, tool_orientations, tool)

# 生成刀具路径
tool_paths = path_generator.generate_final_path()

# 导出为G代码
path_generator.export_to_gcode(tool_paths['paths'], 'output.gcode')
```

## 完整工作流

**脚本**：`scripts/new_algorithm_workflow.py`

**功能**：实现从NURBS曲面到刀具路径的完整流程

**运行方式**：
```bash
python scripts/new_algorithm_workflow.py
```

**输出**：
- 分区结果（可视化）
- 刀具路径（CSV文件）
- 路径可视化

## 测试方法

### 1. 单元测试

**测试NURBS曲面处理器**：
```bash
python -c "from utils.nurbsSurfaceProcessor import NURBSSurfaceProcessor; processor = NURBSSurfaceProcessor(); surface = processor.create_test_surface(); points, normals, curvatures, principal_curvatures = processor.sample_points(resolution_u=20, resolution_v=20, adaptive=False); print('NURBS处理器测试成功')"
```

**测试指标计算器**：
```bash
python -c "from core.meshProcessor import MeshProcessor; from core.nonSphericalTool import NonSphericalTool; from core.indicatorCalculator import IndicatorCalculator; import numpy as np; import open3d as o3d; pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3)); mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5); mesh_processor = MeshProcessor(mesh); mesh_processor.gaussian_curvatures = np.random.rand(100); mesh_processor.principal_curvatures = np.random.rand(100, 2); mesh_processor.rolled_error = np.random.rand(100); tool = NonSphericalTool(profile_type='ellipsoidal', params={'semi_axes': [1.0, 0.5]}); calculator = IndicatorCalculator(mesh_processor, tool); sim = calculator.calculate_combined_similarity(0, 1); print('指标计算器测试成功')"
```

### 2. 集成测试

**运行完整工作流**：
```bash
python scripts/new_algorithm_workflow.py
```

## 配置参数

### NURBS曲面处理器参数
- `resolution_u`：u方向的采样分辨率
- `resolution_v`：v方向的采样分辨率
- `adaptive`：是否使用自适应采样
- `curvature_threshold`：曲率阈值，用于自适应采样

### 高级表面分区器参数
- `resolution`：聚类分辨率参数，控制分区数量
- `enable_secondary_partitioning`：是否启用二次分区

### 指标计算器参数
- `sigma_k`：高斯曲率相似性的带宽参数
- `sigma_n`：几何连续性相似性的法向变化带宽参数
- `sigma_r`：直纹面逼近误差相似性的带宽参数
- `weights`：综合相似性的权重 (高斯曲率权重, 几何连续性权重, 直纹面误差权重)

## 注意事项

1. **计算性能**：对于复杂曲面，采样分辨率和分区数量会影响计算时间
2. **内存使用**：高分辨率采样可能会占用大量内存
3. **依赖项**：需要安装以下依赖：
   - numpy
   - scipy
   - open3d
   - networkx
   - leidenalg (可选，用于Leiden聚类)
   - igraph (可选，用于Leiden聚类)

## 常见问题

1. **NURBS曲面创建失败**：检查控制点格式是否正确
2. **分区数量过多**：调整`resolution`参数
3. **二次分区不执行**：确保`enable_secondary_partitioning`为True，且分区大小足够大
4. **刀具路径生成失败**：检查网格质量和分区结果

## 性能优化

1. **向量化计算**：核心算法已优化为使用向量化计算
2. **并行处理**：加权邻接矩阵构建使用了并行计算
3. **缓存机制**：指标计算器使用缓存减少重复计算
4. **KD树加速**：对称性检测使用KD树加速最近邻搜索
5. **采样优化**：方向场可视化使用顶点采样减少计算量

## 未来改进

1. **支持更多网格格式**：增加对更多网格文件格式的支持
2. **更高级的采样策略**：实现更智能的自适应采样算法
3. **优化对称性检测**：进一步改进对称性检测的性能和准确性
4. **更多刀具类型支持**：扩展对不同刀具类型的支持
5. **G代码生成优化**：改进G代码生成的质量和效率

## 联系信息

如有问题或建议，请联系开发团队。
