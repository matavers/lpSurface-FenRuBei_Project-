# 新算法实现说明

## 算法流程

新算法实现了从NURBS曲面到刀具路径的完整流程，包括以下步骤：

1. **NURBS曲面创建**：创建或加载NURBS曲面
2. **几何属性计算**：计算高斯曲率、主曲率和法向量
3. **点云采样**：按照算法中的采样方式生成点云
4. **网格创建**：将点云转换为三角网格
5. **指标计算**：计算TAR和3个新的指标（高斯曲率相似性、几何连续性相似性、直纹面逼近误差相似性）
6. **加权邻接矩阵构建**：使用新指标构建加权邻接矩阵
7. **Leiden分区**：运行Leiden聚类算法进行分区
8. **二次分区**：（可选）基于直纹面拟合误差进行二次分区
9. **刀具路径生成**：为每个分区生成刀具路径

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

1. **向量化计算**：NURBS曲面处理器已优化为使用向量化计算
2. **并行处理**：加权邻接矩阵构建使用了并行计算
3. **缓存机制**：指标计算器使用缓存减少重复计算

## 未来改进

1. **支持更多NURBS曲面格式**：增加对外部NURBS文件的支持
2. **更高级的采样策略**：实现更智能的自适应采样算法
3. **优化分区算法**：进一步改进Leiden聚类的参数设置
4. **更多刀具类型支持**：扩展对不同刀具类型的支持

## 联系信息

如有问题或建议，请联系开发团队。
