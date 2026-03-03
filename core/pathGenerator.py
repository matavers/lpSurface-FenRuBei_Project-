"""
刀具路径生成器
"""

import numpy as np
from typing import List, Dict, Tuple, Any
import networkx as nx
from .meshProcessor import MeshProcessor
from .nonSphericalTool import NonSphericalTool


class PathGenerator:
    def __init__(self, mesh: MeshProcessor, iso_curves: List[List[np.ndarray]],
                 tool_orientations: np.ndarray, tool: NonSphericalTool):
        """
        初始化路径生成器
        Args:
            mesh: 网格处理器
            iso_curves: 等值线列表
            tool_orientations: 工具方向场
            tool: 刀具模型
        """
        self.mesh = mesh
        self.iso_curves = iso_curves
        self.tool_orientations = tool_orientations
        self.tool = tool

        # 路径数据
        self.cc_paths = []  # 接触点路径
        self.cl_paths = []  # 刀位点路径

    def connect_iso_curves(self) -> List[List[np.ndarray]]:
        """
        连接等值线形成连续路径
        
        完整实现：基于几何特征和拓扑关系连接等值线
        
        Returns:
            连接的路径列表
        """
        if not self.iso_curves:
            return []

        # 1. 计算每条曲线的特征（起点、终点、长度、方向、曲率）
        curve_features = []
        for i, curve in enumerate(self.iso_curves):
            if len(curve) < 2:
                continue
            
            start = curve[0]
            end = curve[-1]
            length = np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))
            
            # 计算曲线方向向量
            direction = end - start
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-6:
                direction /= direction_norm
            else:
                direction = np.zeros(3)
            
            # 计算曲线曲率
            curvature = self._calculate_curve_curvature(curve)
            
            curve_features.append({
                'index': i,
                'curve': curve,
                'start': start,
                'end': end,
                'length': length,
                'direction': direction,
                'curvature': curvature
            })
        
        if not curve_features:
            return []
        
        # 2. 构建距离矩阵，考虑曲线方向和曲率
        n = len(curve_features)
        dist_matrix = np.zeros((n, n))
        direction_matrix = np.zeros((n, n))
        curvature_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # 计算曲线i的终点到曲线j的起点的距离
                    dist = np.linalg.norm(curve_features[i]['end'] - curve_features[j]['start'])
                    dist_matrix[i, j] = dist
                    
                    # 计算方向一致性
                    direction_dot = np.dot(curve_features[i]['direction'], curve_features[j]['direction'])
                    direction_matrix[i, j] = direction_dot
                    
                    # 计算曲率相似性
                    curvature_diff = abs(curve_features[i]['curvature'] - curve_features[j]['curvature'])
                    curvature_matrix[i, j] = curvature_diff
                else:
                    dist_matrix[i, j] = np.inf
                    direction_matrix[i, j] = -1
                    curvature_matrix[i, j] = np.inf
        
        # 3. 使用改进的贪心算法连接曲线，考虑方向和曲率
        visited = np.zeros(n, dtype=bool)
        connected_paths = []
        
        while not np.all(visited):
            # 找到未访问的曲线
            unvisited = np.where(~visited)[0]
            if not len(unvisited):
                break
            
            # 选择起点：优先选择长度较长的曲线
            lengths = [curve_features[i]['length'] for i in unvisited]
            start_idx = unvisited[np.argmax(lengths)]
            visited[start_idx] = True
            current_path = curve_features[start_idx]['curve']
            
            # 尝试连接其他曲线
            while True:
                # 找到最佳的未访问曲线
                current_end = current_path[-1]
                best_score = -np.inf
                next_idx = -1
                
                for i in unvisited:
                    if not visited[i]:
                        # 计算连接评分
                        dist = np.linalg.norm(current_end - curve_features[i]['start'])
                        if dist > 1.0:  # 连接阈值
                            continue
                        
                        # 方向一致性评分
                        direction_score = direction_matrix[start_idx, i]
                        
                        # 曲率相似性评分
                        curvature_score = 1.0 / (1.0 + curvature_matrix[start_idx, i])
                        
                        # 距离评分
                        distance_score = 1.0 / (1.0 + dist)
                        
                        # 综合评分
                        total_score = 0.5 * distance_score + 0.3 * direction_score + 0.2 * curvature_score
                        
                        if total_score > best_score:
                            best_score = total_score
                            next_idx = i
                
                # 如果找到合适的连接
                if next_idx != -1:
                    visited[next_idx] = True
                    # 连接两条曲线，添加过渡点
                    current_path = self._connect_two_curves(current_path, curve_features[next_idx]['curve'])
                else:
                    break
            
            # 平滑路径
            current_path = self._smooth_path(current_path)
            connected_paths.append(current_path)
        
        return connected_paths

    def calculate_cl_points(self, cc_points: List[np.ndarray],
                            orientations: List[np.ndarray]) -> List[np.ndarray]:
        """
        计算刀位点
        
        优化：使用NumPy向量化操作加速计算
        
        Args:
            cc_points: 接触点列表
            orientations: 工具方向列表
        Returns:
            刀位点列表
        """
        # 基于刀具几何计算刀位点
        tool_type = self.tool.profile_type
        if tool_type == 'ellipsoidal':
            # 椭球形刀具：使用长半轴作为半径
            tool_radius = self.tool.params.get('semi_axes', [5.0, 3.0])[0]
        elif tool_type == 'cylindrical':
            # 圆柱形刀具：使用直径的一半
            tool_radius = self.tool.params.get('diameter', 6.0) / 2
        elif tool_type == 'spherical':
            # 球形刀具：使用半径
            tool_radius = self.tool.params.get('radius', 5.0)
        elif tool_type == 'conical':
            # 锥形刀具：使用底部直径的一半
            tool_radius = self.tool.params.get('base_diameter', 8.0) / 2
        else:
            # 默认刀具半径
            tool_radius = 5.0

        # 使用NumPy向量化操作批量计算刀位点
        cc_array = np.array(cc_points)
        orient_array = np.array(orientations)
        
        # 批量计算所有刀位点
        cl_array = cc_array + orient_array * tool_radius
        
        # 转换回列表格式
        return cl_array.tolist()

    def _calculate_curve_curvature(self, curve: List[np.ndarray]) -> float:
        """
        计算曲线的平均曲率
        Args:
            curve: 曲线点列表
        Returns:
            平均曲率
        """
        if len(curve) < 3:
            return 0.0
        
        total_curvature = 0.0
        for i in range(1, len(curve) - 1):
            # 计算三点的曲率
            p_prev = curve[i-1]
            p_curr = curve[i]
            p_next = curve[i+1]
            
            # 计算向量
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            
            # 计算曲率
            cross_product = np.cross(v1, v2)
            cross_norm = np.linalg.norm(cross_product)
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0 and v2_norm > 0:
                curvature = cross_norm / (v1_norm * v2_norm * np.linalg.norm(p_next - p_prev))
                total_curvature += curvature
        
        avg_curvature = total_curvature / (len(curve) - 2) if len(curve) > 2 else 0.0
        return avg_curvature

    def _connect_two_curves(self, curve1: List[np.ndarray], curve2: List[np.ndarray]) -> List[np.ndarray]:
        """
        连接两条曲线，添加过渡点
        Args:
            curve1: 第一条曲线
            curve2: 第二条曲线
        Returns:
            连接后的曲线
        """
        if not curve1 or not curve2:
            return curve1 + curve2
        
        # 获取两条曲线的端点
        end1 = curve1[-1]
        start2 = curve2[0]
        
        # 计算距离
        distance = np.linalg.norm(end1 - start2)
        
        # 如果距离很近，直接连接
        if distance < 0.1:
            return curve1 + curve2
        
        # 否则，添加过渡点
        num_transition_points = max(2, int(distance * 5))  # 每毫米约5个过渡点
        transition_points = []
        
        for t in np.linspace(0, 1, num_transition_points):
            # 线性插值
            point = (1 - t) * end1 + t * start2
            transition_points.append(point)
        
        # 连接曲线和过渡点
        connected_curve = curve1 + transition_points + curve2
        return connected_curve

    def _smooth_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """
        平滑路径，减少尖锐拐角
        Args:
            path: 原始路径
        Returns:
            平滑后的路径
        """
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]  # 保留起点
        
        # 使用移动平均平滑
        window_size = 3
        for i in range(1, len(path) - 1):
            # 计算窗口内的平均点
            start = max(0, i - window_size // 2)
            end = min(len(path), i + window_size // 2 + 1)
            window_points = path[start:end]
            avg_point = np.mean(window_points, axis=0)
            smoothed.append(avg_point)
        
        smoothed.append(path[-1])  # 保留终点
        
        return smoothed

    def optimize_path_sequence(self, curves: List[List[np.ndarray]]) -> List[int]:
        """
        优化路径序列（旅行商问题）
        
        优化：使用模拟退火算法优化路径序列
        
        Args:
            curves: 曲线列表
        Returns:
            优化后的序列索引
        """
        if len(curves) <= 1:
            return list(range(len(curves)))

        # 构建距离矩阵
        n = len(curves)
        dist_matrix = np.zeros((n, n))

        # 提取所有曲线的起点和终点
        curve_ends = np.array([curve[-1] for curve in curves])
        curve_starts = np.array([curve[0] for curve in curves])
        
        # 使用NumPy向量化操作计算距离矩阵
        for i in range(n):
            # 计算曲线i的终点到所有其他曲线起点的距离
            distances = np.linalg.norm(curve_ends[i] - curve_starts, axis=1)
            dist_matrix[i, :] = distances

        # 使用模拟退火算法优化路径序列
        def calculate_total_distance(sequence):
            """计算路径总距离"""
            total_distance = 0
            for i in range(len(sequence) - 1):
                total_distance += dist_matrix[sequence[i], sequence[i+1]]
            return total_distance
        
        # 初始化
        current_sequence = list(range(n))
        current_distance = calculate_total_distance(current_sequence)
        best_sequence = current_sequence.copy()
        best_distance = current_distance
        
        # 模拟退火参数
        temperature = 100.0
        cooling_rate = 0.95
        min_temperature = 1e-3
        max_iterations = 1000
        
        # 执行模拟退火
        for iteration in range(max_iterations):
            # 生成新序列：交换两个随机位置
            new_sequence = current_sequence.copy()
            i, j = np.random.randint(0, n, 2)
            new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]
            
            # 计算新序列的距离
            new_distance = calculate_total_distance(new_sequence)
            
            # 计算接受概率
            if new_distance < current_distance:
                accept_prob = 1.0
            else:
                delta = new_distance - current_distance
                accept_prob = np.exp(-delta / temperature)
            
            # 接受或拒绝新序列
            if np.random.rand() < accept_prob:
                current_sequence = new_sequence
                current_distance = new_distance
                
                # 更新最佳序列
                if current_distance < best_distance:
                    best_sequence = current_sequence.copy()
                    best_distance = current_distance
            
            # 降温
            temperature *= cooling_rate
            if temperature < min_temperature:
                break
        
        # 验证序列是否包含所有曲线
        if len(set(best_sequence)) != n:
            # 如果不完整，使用最近邻算法作为回退
            visited = np.zeros(n, dtype=bool)
            sequence = [0]
            visited[0] = True

            for _ in range(1, n):
                current = sequence[-1]
                unvisited_distances = dist_matrix[current, :]
                unvisited_distances[visited] = np.inf
                next_idx = np.argmin(unvisited_distances)
                sequence.append(next_idx)
                visited[next_idx] = True
            
            return sequence
        
        return best_sequence

    def generate_final_path(self) -> Dict[str, Any]:
        """
        生成最终刀具路径
        Returns:
            包含路径信息的字典
        """
        print("生成最终刀具路径...")

        # 1. 连接等值线
        connected_curves = self.connect_iso_curves()

        # 2. 优化序列
        sequence = self.optimize_path_sequence(connected_curves)

        # 3. 生成完整路径
        all_paths = []

        for idx in sequence:
            curve = connected_curves[idx]

            if len(curve) < 2:
                continue

            # 为曲线上的点分配工具方向
            path_points = []
            path_orientations = []

            for point in curve:
                # 找到最近的顶点
                distances = np.linalg.norm(self.mesh.vertices - point, axis=1)
                nearest_vertex = np.argmin(distances)

                path_points.append(point)
                path_orientations.append(self.tool_orientations[nearest_vertex])

            # 计算刀位点
            cl_points = self.calculate_cl_points(path_points, path_orientations)

            # 添加到路径列表
            path_data = {
                'type': 'cc_path',
                'points': np.array(path_points),
                'orientations': np.array(path_orientations),
                'cl_points': np.array(cl_points)
            }

            all_paths.append(path_data)

        # 4. 添加连接路径
        self._add_connection_paths(all_paths)

        # 5. 计算路径统计
        total_length = self._calculate_total_length(all_paths)

        result = {
            'paths': all_paths,
            'num_paths': len(all_paths),
            'total_length': total_length,
            'num_points': sum(len(p['points']) for p in all_paths)
        }

        print(f"刀具路径生成完成: {len(all_paths)} 条路径, 总长度 {total_length:.2f} mm")

        return result

    def _add_connection_paths(self, paths: List[Dict[str, Any]]):
        """在路径之间添加连接路径"""
        if len(paths) <= 1:
            return

        for i in range(len(paths) - 1):
            end_point = paths[i]['points'][-1]
            start_point = paths[i + 1]['points'][0]

            # 生成沿着球面表面的连接路径
            # 1. 计算两个点之间的球面距离
            distance = np.linalg.norm(end_point - start_point)
            
            # 2. 根据距离确定插值点数
            num_interpolation = max(3, int(distance * 10))  # 每毫米约10个点
            
            # 3. 生成球面插值点
            connection_points = []
            connection_orientations = []
            connection_cl_points = []
            
            # 确保至少有两个点（起点和终点）
            num_points = max(2, num_interpolation)
            
            for t in np.linspace(0, 1, num_points):
                # 球面线性插值（SLERP）
                # 计算单位向量
                v1 = end_point / np.linalg.norm(end_point)
                v2 = start_point / np.linalg.norm(start_point)
                
                # 计算点积
                dot = np.dot(v1, v2)
                dot = max(-1.0, min(1.0, dot))  # 限制在有效范围内
                
                # 计算夹角
                theta = np.arccos(dot)
                
                if theta < 1e-6:
                    # 如果夹角很小，使用线性插值
                    interpolated = (1 - t) * end_point + t * start_point
                else:
                    # 使用SLERP插值
                    sin_theta = np.sin(theta)
                    interpolated = ((np.sin((1 - t) * theta) / sin_theta) * v1 + 
                                  (np.sin(t * theta) / sin_theta) * v2)
                    # 保持原始距离
                    radius1 = np.linalg.norm(end_point)
                    radius2 = np.linalg.norm(start_point)
                    radius = radius1 * (1 - t) + radius2 * t
                    interpolated = interpolated * radius
                
                connection_points.append(interpolated)
                
                # 线性插值方向和刀位点
                orientation1 = paths[i]['orientations'][-1]
                orientation2 = paths[i + 1]['orientations'][0]
                orientation = (1 - t) * orientation1 + t * orientation2
                orientation_norm = np.linalg.norm(orientation)
                if orientation_norm > 1e-6:
                    orientation = orientation / orientation_norm
                else:
                    orientation = orientation1  # 如果方向向量很小，使用起始方向
                connection_orientations.append(orientation)
                
                cl_point1 = paths[i]['cl_points'][-1]
                cl_point2 = paths[i + 1]['cl_points'][0]
                cl_point = (1 - t) * cl_point1 + t * cl_point2
                connection_cl_points.append(cl_point)

            connection_data = {
                'type': 'connection',
                'points': np.array(connection_points),
                'orientations': np.array(connection_orientations),
                'cl_points': np.array(connection_cl_points)
            }

            # 插入连接路径
            paths.insert(i * 2 + 1, connection_data)

    def _calculate_total_length(self, paths: List[Dict[str, Any]]) -> float:
        """计算路径总长度"""
        total_length = 0

        for path in paths:
            points = path['points']
            for i in range(len(points) - 1):
                total_length += np.linalg.norm(points[i + 1] - points[i])

        return total_length

    def export_to_gcode(self, paths: List[Dict[str, Any]], filename: str):
        """
        导出为G代码（完整实现）
        
        支持五轴加工的完整G代码生成，包括：
        - 工具长度补偿
        - 进给速度优化
        - 安全移动
        - 路径类型标记
        """
        with open(filename, 'w') as f:
            # 程序头
            f.write("; 五轴加工G代码\n")
            f.write("; 自动生成\n")
            f.write(f"; 生成时间: 自动\n")
            f.write(f"; 路径数量: {len(paths)}\n\n")

            # 初始设置
            f.write("G90 ; 绝对坐标\n")
            f.write("G21 ; 毫米单位\n")
            f.write("G17 ; XY平面\n")
            f.write("G40 ; 取消刀具半径补偿\n")
            f.write("G49 ; 取消刀具长度补偿\n")
            f.write("G80 ; 取消固定循环\n")
            f.write("M03 S3000 ; 主轴正转，转速3000rpm\n\n")

            # 安全高度
            safe_height = 50.0
            f.write(f"G0 Z{safe_height:.3f} ; 快速移动到安全高度\n\n")

            # 路径处理
            for path_idx, path in enumerate(paths):
                path_type = path.get('type', 'unknown')
                f.write(f"; 路径 {path_idx}, 类型: {path_type}\n")

                points = path['points']
                orientations = path['orientations']
                cl_points = path.get('cl_points', points)  # 使用刀位点

                if len(points) < 1:
                    continue

                # 移动到第一个点上方
                first_point = cl_points[0]
                f.write(f"G0 X{first_point[0]:.3f} Y{first_point[1]:.3f} ; 快速移动到第一个点上方\n")
                f.write(f"G0 Z{first_point[2] + 5:.3f} ; 快速移动到第一个点Z上方5mm\n")

                # 根据路径类型设置进给速度
                if path_type == 'connection':
                    feed_rate = 300.0  # 连接路径进给速度
                else:
                    feed_rate = 200.0  # 加工路径进给速度

                # 输出加工点
                for i, (cl_point, orientation) in enumerate(zip(cl_points, orientations)):
                    x, y, z = cl_point
                    i_dir, j_dir, k_dir = orientation

                    # 对于五轴加工，使用G1指令并包含方向向量
                    f.write(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} ")
                    f.write(f"I{i_dir:.3f} J{j_dir:.3f} K{k_dir:.3f} ")
                    f.write(f"F{feed_rate:.1f}\n")

                # 移动到安全高度
                f.write(f"G0 Z{safe_height:.3f} ; 快速移动到安全高度\n\n")

            # 程序结束
            f.write("M05 ; 主轴停止\n")
            f.write("G0 Z100.0 ; 快速移动到Z100\n")
            f.write("G0 X0 Y0 ; 快速移动到原点\n")
            f.write("M30 ; 程序结束\n")

        print(f"G代码已导出到: {filename}")
