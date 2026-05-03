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

### 1. 网格处理器 (`core/meshProcessor.py`)

**功能**：
- 处理网格数据，提取顶点、面和邻接关系
- 计算几何属性（曲率、法向量等）
- 计算面面积和顶点面积
- 计算最大切削宽度
- 计算直纹面逼近误差

**使用示例**：
```python
from core.meshProcessor import MeshProcessor
import open3d as o3d

# 加载或创建网格
mesh = o3d.io.read_triangle_mesh("model.obj")

# 创建网格处理器
mesh_processor = MeshProcessor(mesh)

# 计算几何属性
mesh_processor.calculate_max_cutting_width(tool)
mesh_processor.calculate_rolled_error()
```

### 2. 非球面刀具模型 (`core/nonSphericalTool.py`)

**功能**：
- 支持多种刀具类型（椭球形、圆柱形、球形、锥形、自定义）
- 生成刀具轮廓曲线
- 计算有效切削半径
- 执行碰撞检测
- 计算切削宽度

**使用示例**：
```python
from core.nonSphericalTool import NonSphericalTool

# 创建椭球形刀具
tool = NonSphericalTool(
    profile_type='ellipsoidal',
    params={'semi_axes': [9.0, 3.0], 'shank_diameter': 6.0, 'tool_length': 50.0}
)

# 计算有效切削半径
effective_radius = tool.calculate_effective_radius(gamma=0.5, tilt_angle=0.1)

# 执行碰撞检测
is_collision = tool.check_collision_simple(surface_point, surface_normal, tool_orientation)
```

### 3. 指标计算器 (`core/indicatorCalculator.py`)

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

### 4. 高级表面分区器 (`core/advancedSurfacePartitioner.py`)

**功能**：
- 构建基于新指标的加权邻接矩阵
- 执行Leiden聚类分区
- 执行谱聚类（可选）
- 检测和应用对称性约束（旋转、平移、反射、螺旋）
- 确保分区连通性
- 提取分区边界中点

**使用示例**：
```python
from core.advancedSurfacePartitioner import AdvancedSurfacePartitioner

# 创建分区器
partitioner = AdvancedSurfacePartitioner(
    mesh_processor,
    tool,
    resolution=0.1,
    alpha=0.3,
    global_field='rolled_error',
    symmetry_types=['rotation', 'reflection']
)

# 执行分区
labels, edge_midpoints = partitioner.partition_surface(clustering_method='leiden')
```

### 5. 工具方向场生成器 (`core/toolOrientationField.py`)

**功能**：
- 为每个分区选择种子点
- 贪心算法选择每个顶点的TAR
- 拉普拉斯平滑方向场
- 局部重定向确保方向在TAR内

**使用示例**：
```python
from core.toolOrientationField import ToolOrientationField

# 创建方向场生成器
orientation_field = ToolOrientationField(mesh_processor, partition_labels, tool)

# 生成工具方向场
tool_orientations = orientation_field.generate_field()
```

### 6. 等残留高度场生成器 (`core/isoScallopField.py`)

**功能**：
- 计算等残留高度的标量场
- 从标量场提取等值线
- 优化等值线顺序

**使用示例**：
```python
from core.isoScallopField import IsoScallopFieldGenerator

# 创建等残留高度场生成器
scallop_field = IsoScallopFieldGenerator(mesh_processor, tool_orientations, tool)

# 计算标量场
scalar_field = scallop_field.calculate_scallop_field()

# 提取等值线
iso_curves = scallop_field.extract_iso_curves(scalar_field)
```

### 7. 路径生成器 (`core/pathGenerator.py`)

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

### 8. NURBS曲面处理器 (`core/nurbsProcessor.py`)

**功能**：
- 创建标准NURBS曲面（圆柱、球面、圆锥）
- 计算曲面上任意点的坐标
- 计算法向量和导数
- 计算高斯曲率和主曲率
- 生成网格表示
- 可视化NURBS曲面
- 保存和加载NURBS数据

**使用示例**：
```python
from core.nurbsProcessor import NURBSProcessor

# 创建圆柱面
cylinder = NURBSProcessor.create_cylinder(radius=1.0, height=2.0)

# 计算曲面上的点
point = cylinder.evaluate(u=0.5, v=0.5)

# 计算高斯曲率
gaussian_curvature = cylinder.calculate_gaussian_curvature(u=0.5, v=0.5)

# 可视化曲面
cylinder.visualize(resolution_u=50, resolution_v=20)
```

## 完整工作流

**脚本**：`main.py`

**功能**：实现从NURBS曲面到刀具路径的完整流程

**运行方式**：
```bash
python main.py
```

**输出**：
- 分区结果（可视化）
- 刀具路径（CSV文件）
- 路径可视化

## 测试方法

### 1. 单元测试

**测试NURBS曲面处理器**：
```bash
python tests/test_nurbs.py
```

**测试圆锥曲面**：
```bash
python tests/test_cone_surface.py
```

**测试球体曲面**：
```bash
python tests/test_sphere_surface.py
```

**测试环面曲面**：
```bash
python tests/test_torus_surface.py
```

**测试鞍面曲面**：
```bash
python tests/test_saddle_surface.py
```

### 2. 集成测试

**运行完整工作流**：
```bash
python main.py
```

## 配置参数

### 网格处理器参数
- 无需额外参数，自动从网格提取数据

### 非球面刀具参数
- `profile_type`：刀具轮廓类型 ('ellipsoidal', 'cylindrical', 'spherical', 'conical', 'custom')
- `params`：刀具参数，根据类型不同而不同

### 高级表面分区器参数
- `resolution`：聚类分辨率参数，控制分区数量
- `alpha`：全局引导强度参数，范围[0,1]
- `global_field`：全局场类型，可选值：'rolled_error', 'curvature', 'cutting_width'
- `symmetry_types`：对称性类型列表，可选值：'rotation', 'translation', 'reflection', 'helical', 'combined'

### 指标计算器参数
- `sigma_k`：高斯曲率相似性的带宽参数
- `sigma_n`：几何连续性相似性的法向变化带宽参数
- `sigma_r`：直纹面逼近误差相似性的带宽参数
- `weights`：综合相似性的权重 (高斯曲率权重, 几何连续性权重, 直纹面误差权重)

### NURBS处理器参数
- `radius`：半径
- `height`：高度
- `resolution`：控制点分辨率

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
   - mathutils

## 常见问题

1. **NURBS曲面创建失败**：检查参数是否正确
2. **分区数量过多**：调整`resolution`参数
3. **对称性检测失败**：确保模型具有明显的对称性
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
6. **GUI界面**：开发图形用户界面，提高用户体验

## 联系信息

如有问题或建议，请联系开发团队。
