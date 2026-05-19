"""
直纹面拟合模块

对 ``NewPartitioner`` 输出的每个分区 ``Set[int]`` 独立拟合 S(u,v)=(1-v)C0(u)+v C1(u)。

- 优先使用分区内三角子网格的**拓扑边界**得到两条准线链；若无单连通边界则回退到 **UV 周期展开** 后的 PCA 分箱。
- **端点对齐** 与拟合后 **C1 的 u 反向 (1-u)** 校正，减轻母线交叉与扭曲。
- ``sample_ruled_surface_grid`` 输出 (nu, nv, 3) 规则网格，便于 PyVista StructuredGrid 绘制半透明面片。
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from core.meshProcessor import MeshProcessor
from core.nurbsProcessor import NURBSProcessor


class GeodesicFitter:
    """测地线拟合器，用于将边界点序列拟合为测地线"""

    def __init__(
        self,
        nurbs: NURBSProcessor,
        num_control_points: int = 10,
        degree: int = 3,
        mu: float = 0.1,
        max_iterations: int = 50,
        verbose: bool = False
    ):
        """
        初始化测地线拟合器

        Args:
            nurbs: NURBS曲面处理器
            num_control_points: B样条控制点数量
            degree: B样条次数
            mu: 测地线正则项权重
            max_iterations: 最大迭代次数
            verbose: 是否打印详细信息
        """
        self.nurbs = nurbs
        self.num_control_points = num_control_points
        self.degree = degree
        self.mu = mu
        self.max_iterations = max_iterations
        self.verbose = verbose

    def compute_knot_vector(self, n: int, degree: int) -> np.ndarray:
        """
        计算均匀节点向量

        Args:
            n: 控制点数量
            degree: 样条次数

        Returns:
            节点向量
        """
        if n <= degree:
            n = degree + 1

        num_internal_knots = n - degree - 1
        if num_internal_knots <= 0:
            return np.concatenate([np.zeros(degree + 1), np.ones(degree + 1)])

        internal_knots = np.linspace(0, 1, num_internal_knots + 2)[1:-1]
        knots = np.concatenate([
            np.zeros(degree + 1),
            internal_knots,
            np.ones(degree + 1)
        ])

        return knots

    def cox_de_boor_iterative(self, t: float, knots: np.ndarray, degree: int) -> np.ndarray:
        """
        Cox-de Boor算法计算B样条基函数（迭代实现，更高效）

        Args:
            t: 参数值
            knots: 节点向量
            degree: 次数

        Returns:
            基函数值数组
        """
        n = len(knots) - degree - 1
        N = np.zeros(n)

        for i in range(n):
            if knots[i] <= t < knots[i + 1] or (t == 1.0 and i == n - 1):
                N[i] = 1.0

        for d in range(1, degree + 1):
            new_N = np.zeros(n)
            for i in range(n - d):
                denom1 = knots[i + d] - knots[i]
                denom2 = knots[i + d + 1] - knots[i + 1]

                term1 = 0.0
                if denom1 > 1e-10:
                    term1 = (t - knots[i]) / denom1 * N[i]

                term2 = 0.0
                if denom2 > 1e-10:
                    term2 = (knots[i + d + 1] - t) / denom2 * N[i + 1]

                new_N[i] = term1 + term2
            N = new_N[:n - d + 1]
            if d < degree:
                N = np.concatenate([N, np.zeros(d)])

        return N

    def basis_functions(self, t: float, knots: np.ndarray, degree: int) -> np.ndarray:
        """
        计算所有B样条基函数值

        Args:
            t: 参数值
            knots: 节点向量
            degree: 次数

        Returns:
            基函数值数组
        """
        return self.cox_de_boor_iterative(t, knots, degree)

    def basis_derivative(self, t: float, knots: np.ndarray, degree: int, order: int = 1) -> np.ndarray:
        """
        计算B样条基函数的导数（使用差分方法）

        Args:
            t: 参数值
            knots: 节点向量
            degree: 次数
            order: 导数阶数

        Returns:
            导数值数组
        """
        if order == 0:
            return self.basis_functions(t, knots, degree)

        h = 1e-5
        if order == 1:
            N_plus = self.basis_functions(min(t + h, 1.0), knots, degree)
            N_minus = self.basis_functions(max(t - h, 0.0), knots, degree)
            return (N_plus - N_minus) / (2 * h)
        elif order == 2:
            N_plus = self.basis_functions(min(t + h, 1.0), knots, degree)
            N_center = self.basis_functions(t, knots, degree)
            N_minus = self.basis_functions(max(t - h, 0.0), knots, degree)
            return (N_plus - 2 * N_center + N_minus) / (h * h)
        return np.zeros(len(knots) - degree - 1)

    def evaluate_curve(self, t: float, control_points: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
        """
        评估参数域内的B样条曲线 (u(t), v(t))

        Args:
            t: 参数值
            control_points: 控制点数组 (n, 2) - 每个点为 (u, v)
            knots: 节点向量
            degree: 次数

        Returns:
            (u, v) 参数坐标
        """
        N = self.basis_functions(t, knots, degree)
        uv = np.zeros(2)
        n_cp = min(len(control_points), len(N))
        for i in range(n_cp):
            uv += N[i] * control_points[i]
        return uv

    def evaluate_curve_derivatives(self, t: float, control_points: np.ndarray, knots: np.ndarray, degree: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        评估曲线的一阶和二阶导数

        Args:
            t: 参数值
            control_points: 控制点数组
            knots: 节点向量
            degree: 次数

        Returns:
            (一阶导数, 二阶导数)
        """
        N_deriv = self.basis_derivative(t, knots, degree, 1)
        N_deriv2 = self.basis_derivative(t, knots, degree, 2)

        du_dt = 0.0
        dv_dt = 0.0
        d2u_dt2 = 0.0
        d2v_dt2 = 0.0

        n_cp = len(control_points)
        for i in range(min(n_cp, len(N_deriv))):
            du_dt += N_deriv[i] * control_points[i, 0]
            dv_dt += N_deriv[i] * control_points[i, 1]

        for i in range(min(n_cp, len(N_deriv2))):
            d2u_dt2 += N_deriv2[i] * control_points[i, 0]
            d2v_dt2 += N_deriv2[i] * control_points[i, 1]

        return np.array([du_dt, dv_dt]), np.array([d2u_dt2, d2v_dt2])

    def project_point_to_surface(self, point: np.ndarray, initial_uv: Tuple[float, float] = None, max_iter: int = 50, tol: float = 1e-6) -> Tuple[float, float, np.ndarray]:
        """
        使用牛顿迭代法将点投影到NURBS曲面

        Args:
            point: 3D点
            initial_uv: 初始参数坐标
            max_iter: 最大迭代次数
            tol: 收敛阈值

        Returns:
            (u, v, 投影点)
        """
        if initial_uv is None:
            u, v = 0.5, 0.5
        else:
            u, v = initial_uv
        u, v = self.nurbs.clamp_parameters(u, v)

        for _ in range(max_iter):
            S, Su, Sv, _, _, _ = self.nurbs.evaluate_derivatives(u, v)

            if not np.all(np.isfinite(S)):
                break

            error = S - point

            J = np.array([[Su[0], Sv[0]], [Su[1], Sv[1]], [Su[2], Sv[2]]])
            if not np.all(np.isfinite(J)):
                break
            J_T = J.T

            try:
                delta = np.linalg.solve(J_T @ J + 1e-6 * np.eye(2), -J_T @ error)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(J_T @ J + 1e-6 * np.eye(2), -J_T @ error, rcond=None)[0]

            if not np.all(np.isfinite(delta)):
                break

            u_new, v_new = self.nurbs.clamp_parameters(u + delta[0], v + delta[1])

            if np.linalg.norm([u_new - u, v_new - v]) < tol:
                u, v = u_new, v_new
                break

            u, v = u_new, v_new

        u, v = self.nurbs.clamp_parameters(u, v)
        S = self.nurbs.evaluate(u, v)
        if not np.all(np.isfinite(S)):
            S = self.nurbs.control_points.reshape(-1, 3).mean(axis=0)
        return u, v, S

    def compute_geodesic_curvature(self, t: float, control_points: np.ndarray, knots: np.ndarray, degree: int) -> float:
        """
        计算曲线上某点处的测地曲率

        Args:
            t: 参数值
            control_points: 参数域控制点 (n, 2)
            knots: 节点向量
            degree: 次数

        Returns:
            测地曲率值
        """
        uv = self.evaluate_curve(t, control_points, knots, degree)
        du_dt, d2u_dt2 = self.evaluate_curve_derivatives(t, control_points, knots, degree)
        
        u, v = uv
        du, dv = du_dt
        d2u, d2v = d2u_dt2
        
        S, Su, Sv, Suu, Suv, Svv = self.nurbs.evaluate_derivatives(u, v)
        
        c_prime = Su * du + Sv * dv
        c_prime_norm = np.linalg.norm(c_prime)
        if c_prime_norm < 1e-10:
            return 0.0
        
        c_double_prime = Suu * du**2 + 2 * Suv * du * dv + Svv * dv**2 + Su * d2u + Sv * d2v
        
        normal = np.cross(Su, Sv)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-10:
            return 0.0
        n = normal / normal_norm
        
        cross_prod = np.cross(c_prime, c_double_prime)
        kappa_g = np.dot(cross_prod, n) / (c_prime_norm ** 3 + 1e-10)
        
        return kappa_g

    def compute_chord_length_params(self, data_points: np.ndarray) -> np.ndarray:
        """
        计算弦长参数化

        Args:
            data_points: 数据点数组 (n, 2)

        Returns:
            参数化数组 t_i
        """
        n = len(data_points)
        if n <= 1:
            return np.array([0.0])

        chord_lengths = np.zeros(n)
        for i in range(1, n):
            du = data_points[i, 0] - data_points[i-1, 0]
            dv = data_points[i, 1] - data_points[i-1, 1]
            chord_lengths[i] = chord_lengths[i-1] + np.sqrt(du*du + dv*dv)

        total_length = chord_lengths[-1]
        if total_length > 1e-8:
            return chord_lengths / total_length
        return np.linspace(0, 1, n)

    def compute_objective(self, control_points_flat: np.ndarray, data_points: np.ndarray, data_weights: np.ndarray, 
                        knots: np.ndarray, degree: int, t_params: np.ndarray) -> float:
        """
        计算目标函数值（改进版本）

        Args:
            control_points_flat: 扁平化的控制点数组
            data_points: 数据点的参数坐标 (n, 2)
            data_weights: 数据点权重
            knots: 节点向量
            degree: 次数
            t_params: 每个数据点对应的参数值

        Returns:
            目标函数值
        """
        num_cp = len(control_points_flat) // 2
        control_points = control_points_flat.reshape(num_cp, 2)
        
        fit_error = 0.0
        n_data = len(data_points)
        for i in range(n_data):
            t = t_params[i]
            uv = self.evaluate_curve(t, control_points, knots, degree)
            S_curve = self.nurbs.evaluate(uv[0], uv[1])
            S_data = self.nurbs.evaluate(data_points[i, 0], data_points[i, 1])
            fit_error += data_weights[i] * np.linalg.norm(S_curve - S_data) ** 2
        
        num_samples = 20
        kappa_integral = 0.0
        dt = 1.0 / (num_samples - 1) if num_samples > 1 else 1.0
        
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0.5
            kappa_g = self.compute_geodesic_curvature(t, control_points, knots, degree)
            kappa_integral += kappa_g ** 2 * dt
        
        return fit_error + self.mu * kappa_integral

    def fit(
        self,
        chain_vertices: List[int],
        chain_weights: List[float],
        mesh_vertices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]], dict]:
        """
        拟合测地线（真正的测地线优化）

        Args:
            chain_vertices: 边界链顶点索引
            chain_weights: 边界链顶点权重
            mesh_vertices: 网格顶点坐标

        Returns:
            (控制点数组 (n, 2), 节点向量, 参数点列表, 拟合质量信息)
        """
        if len(chain_vertices) < 2:
            if len(chain_vertices) == 1:
                p = mesh_vertices[chain_vertices[0]]
                u, v, _ = self.project_point_to_surface(p)
                return np.array([[u, v], [u, v]]), self.compute_knot_vector(2, 1), [(u, v)], \
                       {'fit_error': 0.0, 'avg_kappa_g': 0.0, 'objective': 0.0}
            return np.array([]), np.array([]), [], \
                   {'fit_error': 0.0, 'avg_kappa_g': 0.0, 'objective': 0.0}

        if self.verbose:
            print(f"  Fitting geodesic with {len(chain_vertices)} points")

        data_points = []
        for v_idx in chain_vertices:
            p = mesh_vertices[v_idx]
            u, v, _ = self.project_point_to_surface(p)
            data_points.append([u, v])
        data_points = np.array(data_points)
        
        t_params = self.compute_chord_length_params(data_points)
        
        num_cp = min(self.num_control_points, max(3, len(chain_vertices) // 2))
        num_cp = max(num_cp, self.degree + 1)

        knots = self.compute_knot_vector(num_cp, self.degree)

        initial_control_points = np.zeros((num_cp, 2))
        for i in range(num_cp):
            t = i / (num_cp - 1) if num_cp > 1 else 0.5
            idx = np.searchsorted(t_params, t)
            idx = min(max(idx, 1), len(data_points) - 1)
            t0 = t_params[idx - 1]
            t1 = t_params[idx]
            
            if t1 - t0 > 1e-8:
                alpha = (t - t0) / (t1 - t0)
                initial_control_points[i] = (1 - alpha) * data_points[idx - 1] + alpha * data_points[idx]
            else:
                initial_control_points[i] = data_points[idx]

        if not SCIPY_AVAILABLE:
            if self.verbose:
                print("  scipy not available, returning initial control points")
            return initial_control_points, knots, [(p[0], p[1]) for p in data_points], \
                   {'fit_error': 0.0, 'avg_kappa_g': 0.0, 'objective': 0.0}

        bounds = []
        for _ in range(num_cp):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))

        result = minimize(
            fun=self.compute_objective,
            x0=initial_control_points.flatten(),
            args=(data_points, np.array(chain_weights), knots, self.degree, t_params),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': self.max_iterations,
                'disp': self.verbose
            }
        )

        optimized_control_points = result.x.reshape(num_cp, 2)

        final_params = []
        for t in t_params:
            uv = self.evaluate_curve(t, optimized_control_points, knots, self.degree)
            final_params.append((uv[0], uv[1]))

        fit_error = 0.0
        for i, t in enumerate(t_params):
            uv = self.evaluate_curve(t, optimized_control_points, knots, self.degree)
            S_curve = self.nurbs.evaluate(uv[0], uv[1])
            S_data = self.nurbs.evaluate(data_points[i, 0], data_points[i, 1])
            fit_error += chain_weights[i] * np.linalg.norm(S_curve - S_data) ** 2
        fit_error /= len(chain_weights)

        num_kappa_samples = 20
        avg_kappa_g = 0.0
        for i in range(num_kappa_samples):
            t = i / (num_kappa_samples - 1) if num_kappa_samples > 1 else 0.5
            avg_kappa_g += abs(self.compute_geodesic_curvature(t, optimized_control_points, knots, self.degree))
        avg_kappa_g /= num_kappa_samples

        quality_info = {
            'fit_error': fit_error,
            'avg_kappa_g': avg_kappa_g,
            'objective': result.fun,
            'success': result.success,
            'message': result.message,
            'num_iterations': result.nit
        }

        if self.verbose:
            print(f"  Optimization completed: {result.message}")
            print(f"  Fit error: {fit_error:.6f}, Avg geodesic curvature: {avg_kappa_g:.6f}")

        return optimized_control_points, knots, final_params, quality_info


class LinearSurfaceFitter:
    """
    直纹面拟合：输入 ``List[Set[int]]`` 分区顶点集。
    准线优先来自**拓扑子网格边界**，否则使用展开 UV 后的 PCA 分箱；并对 C1 做 ``u`` 或端点序对齐。
    """

    def __init__(self, mesh: MeshProcessor, nurbs: NURBSProcessor):
        self.mesh = mesh
        self.nurbs = nurbs
        self.num_vertices = len(mesh.vertices)
        self._mesh_vertices = np.asarray(mesh.vertices, dtype=float)

    @staticmethod
    def _dedupe_chain(vertices: List[int]) -> List[int]:
        out: List[int] = []
        for v in vertices:
            if not out or out[-1] != v:
                out.append(v)
        return out

    @staticmethod
    def _unwrap_periodic_u(uv: np.ndarray, period: float = 1.0, seam_threshold: float = 0.5) -> np.ndarray:
        """
        将闭合参数域 [0, period) 上跨越接缝的 U 在数值上展开，便于 PCA / 分箱。
        若 max(u)-min(u) > seam_threshold*period，则对较小的一半坐标平移 +period。
        """
        out = np.array(uv, dtype=float, copy=True)
        u = out[:, 0]
        span = float(np.ptp(u))
        if span > seam_threshold * period:
            med = float(np.median(u))
            out[:, 0] = np.where(u < med, u + period, u)
        return out

    def _order_polyline_from_edges(self, edges: List[Tuple[int, int]]) -> List[int]:
        """将无向边列表连成一条简单路径或环（顶点索引序列）。"""
        if not edges:
            return []
        adj = defaultdict(list)
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        start = min(adj.keys())
        for v in adj:
            if len(adj[v]) == 1:
                start = v
                break
        prev = -1
        cur = start
        order: List[int] = []
        max_steps = len(edges) * 2 + 4
        for _ in range(max_steps):
            order.append(cur)
            nxt = None
            for w in adj[cur]:
                if w != prev:
                    nxt = w
                    break
            if nxt is None:
                break
            if nxt == start and len(order) >= 2:
                break
            prev, cur = cur, nxt
        return order

    @staticmethod
    def _bfs_shortest_path_tree(start: int, adj: Dict[int, List[int]]) -> Tuple[Dict[int, int], Dict[int, int]]:
        """BFS：返回 parent、dist（仅连通的顶点）。"""
        from collections import deque

        dist = {start: 0}
        parent = {start: -1}
        q = deque([start])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    q.append(w)
        return parent, dist

    @staticmethod
    def _reconstruct_path(parent: Dict[int, int], u: int, v: int) -> List[int]:
        if v not in parent:
            return []
        path_v = []
        cur = v
        while cur != -1:
            path_v.append(cur)
            cur = parent[cur]
        path_v.reverse()
        if not path_v or path_v[0] != u:
            return []
        return path_v

    def _longest_path_on_boundary_adj(self, adj: Dict[int, List[int]]) -> List[int]:
        """
        在边界子图（1D 链或环）上取尽可能长的顶点序列。
        若有度为 1 的端点，则在端点对之间取最长最短路径；否则按简单环走一圈。
        """
        if not adj:
            return []
        verts = list(adj.keys())
        leaves = [v for v in verts if len(adj[v]) == 1]

        if len(leaves) >= 2:
            best: List[int] = []
            for u in leaves:
                parent, dist = self._bfs_shortest_path_tree(u, adj)
                for v in leaves:
                    if u >= v:
                        continue
                    if v not in dist:
                        continue
                    path = self._reconstruct_path(parent, u, v)
                    if len(path) > len(best):
                        best = path
            if len(best) >= 2:
                return best

        start = min(verts)
        prev = -1
        cur = start
        order: List[int] = []
        max_steps = len(verts) * 2 + 4
        for _ in range(max_steps):
            order.append(cur)
            neighs = [w for w in adj[cur] if w != prev]
            if not neighs:
                break
            nxt = neighs[0]
            if nxt == start and len(order) >= 2:
                break
            prev, cur = cur, nxt
        return order

    def _boundary_edge_connected_components(
        self, bedges: List[Tuple[int, int]]
    ) -> List[List[Tuple[int, int]]]:
        """将边界边按顶点连通性分成若干边列表。"""
        if not bedges:
            return []
        v_adj = defaultdict(set)
        for a, b in bedges:
            v_adj[a].add(b)
            v_adj[b].add(a)
        seen: Set[int] = set()
        comps: List[List[Tuple[int, int]]] = []
        for seed in v_adj:
            if seed in seen:
                continue
            stack = [seed]
            comp_vs: Set[int] = set()
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                comp_vs.add(x)
                for y in v_adj[x]:
                    if y not in seen:
                        stack.append(y)
            cedges = [(a, b) for a, b in bedges if a in comp_vs and b in comp_vs]
            if cedges:
                comps.append(cedges)
        return comps

    def _split_ordered_boundary_into_two_chains(
        self, ordered: List[int], is_closed: bool
    ) -> Optional[Tuple[List[int], List[int]]]:
        """沿边界顶点序列的 SVD 主轴在极值处切开，得到两条准线链（开链或闭链）。"""
        if len(ordered) < 4:
            return None
        verts = self._mesh_vertices[np.array(ordered, dtype=int)]
        mu = verts.mean(axis=0)
        C = verts - mu
        try:
            _, _, vt = np.linalg.svd(C, full_matrices=False)
            axis = vt[0]
        except np.linalg.LinAlgError:
            return None
        proj = (C @ axis).ravel()
        i_lo = int(np.argmin(proj))
        i_hi = int(np.argmax(proj))
        if i_lo == i_hi:
            return None

        n = len(ordered)

        def circ_forward(i0: int, i1: int) -> List[int]:
            out: List[int] = []
            k = i0
            for _ in range(n + 2):
                out.append(ordered[k])
                if k == i1 and len(out) > 1:
                    break
                k = (k + 1) % n
            return out

        def circ_backward(i0: int, i1: int) -> List[int]:
            out: List[int] = []
            k = i0
            for _ in range(n + 2):
                out.append(ordered[k])
                if k == i1 and len(out) > 1:
                    break
                k = (k - 1) % n
            return out

        if is_closed:
            chain_a = circ_forward(i_lo, i_hi)
            chain_b = circ_backward(i_lo, i_hi)
        else:
            if i_lo > i_hi:
                i_lo, i_hi = i_hi, i_lo
            chain_a = list(ordered[i_lo : i_hi + 1])
            idx_b = list(range(i_lo, -1, -1)) + list(range(n - 1, i_hi - 1, -1))
            chain_b = [ordered[i] for i in idx_b]

        ca = self._dedupe_chain(chain_a)
        cb = self._dedupe_chain(chain_b)
        if len(ca) < 2 or len(cb) < 2:
            return None
        return ca, cb

    def _extract_topology_directrix_chains(self, partition: Set[int]) -> Optional[Tuple[List[int], List[int]]]:
        """
        由分区内三角子网格的拓扑边界边提取准线：不依赖单一完美闭环；
        各连通分量上取最长边界折线，再选最长的两条作为准线；单条长闭链则劈成两半。
        """
        pin = partition
        if len(pin) < 4:
            return None
        faces = self.mesh.faces
        ec = defaultdict(int)
        for fi in range(len(faces)):
            f = faces[fi]
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            if a not in pin or b not in pin or c not in pin:
                continue
            for e in (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))):
                ec[e] += 1
        bedges = [e for e, c in ec.items() if c == 1]
        if len(bedges) < 2:
            return None

        comps = self._boundary_edge_connected_components(bedges)
        polylines: List[List[int]] = []
        for cedges in comps:
            adj: Dict[int, List[int]] = defaultdict(list)
            for a, b in cedges:
                adj[a].append(b)
                adj[b].append(a)
            path = self._longest_path_on_boundary_adj(adj)
            path = self._dedupe_chain(path)
            if len(path) >= 2:
                polylines.append(path)

        polylines.sort(key=len, reverse=True)
        if not polylines:
            return None

        if len(polylines) >= 2:
            c0, c1 = polylines[0], polylines[1]
            c1 = self._align_chain_pair_by_endpoints(c0, c1)
            if len(c0) >= 2 and len(c1) >= 2:
                return c0, c1
            return None

        only = polylines[0]
        if len(only) < 4:
            return None
        deg_cnt: Dict[int, int] = defaultdict(int)
        for a, b in bedges:
            deg_cnt[a] += 1
            deg_cnt[b] += 1
        comp_vs = set(only)
        is_closed = all(deg_cnt.get(v, 0) == 2 for v in comp_vs)
        split = self._split_ordered_boundary_into_two_chains(only, is_closed)
        if split is None:
            return None
        c0, c1 = split
        c1 = self._align_chain_pair_by_endpoints(c0, c1)
        if len(c0) >= 2 and len(c1) >= 2:
            return c0, c1
        return None

    def _align_chain_pair_by_endpoints(self, chain0: List[int], chain1: List[int]) -> List[int]:
        """若 chain1 与 chain0 反向，则反转 chain1，使 S(u,v) 母线尽量不交叉。"""
        if not chain0 or not chain1:
            return chain1
        V = self._mesh_vertices
        p00, p01 = V[int(chain0[0])], V[int(chain0[-1])]
        p10, p11 = V[int(chain1[0])], V[int(chain1[-1])]
        d_same = np.linalg.norm(p00 - p10) + np.linalg.norm(p01 - p11)
        d_flip = np.linalg.norm(p00 - p11) + np.linalg.norm(p01 - p10)
        if d_flip < d_same:
            return list(reversed(chain1))
        return chain1

    def _refine_directrix1_u_after_fit(self, d0: Dict, d1: Dict, degree: int) -> None:
        """在样条拟合后比较 C0(u) 与 C1(u)/C1(1-u)，必要时令 C1 使用 1-u。"""
        score_same = 0.0
        score_flip = 0.0
        for tu in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            p0 = self.get_geodesic_at_t(d0['control_points'], d0['knots'], degree, float(tu))
            p1a = self.get_geodesic_at_t(d1['control_points'], d1['knots'], degree, float(tu))
            p1b = self.get_geodesic_at_t(d1['control_points'], d1['knots'], degree, float(1.0 - tu))
            score_same += float(np.sum((p0 - p1a) ** 2))
            score_flip += float(np.sum((p0 - p1b) ** 2))
        d1['u_reversed'] = bool(score_flip < score_same)

    def extract_directrix_vertex_chains(
        self,
        partition: Set[int],
        num_bins: Optional[int] = None,
    ) -> Tuple[List[int], List[int], Dict[str, np.ndarray]]:
        """
        优先用分区内三角子网格的拓扑边界劈成两条准线；否则在展开后的 (u,v) 域做 PCA 分箱。
        """
        idx = np.array(sorted(int(i) for i in partition), dtype=int)
        n = len(idx)
        meta: Dict[str, np.ndarray] = {}
        if n == 0:
            return [], [], meta

        probe = GeodesicFitter(self.nurbs, num_control_points=4, degree=1, mu=0.0, max_iterations=5)
        verts = self._mesh_vertices[idx]
        uv = np.zeros((n, 2), dtype=float)
        for i in range(n):
            u, v, _ = probe.project_point_to_surface(verts[i])
            uv[i, 0] = u
            uv[i, 1] = v

        if n == 1:
            v0 = int(idx[0])
            return [v0, v0], [v0, v0], meta

        topo = self._extract_topology_directrix_chains(partition)
        if topo is not None:
            c0, c1 = topo
            c1 = self._align_chain_pair_by_endpoints(c0, c1)
            meta['method'] = np.array(['topology'])
            meta['uv_raw'] = uv
            return c0, c1, meta

        uv_w = self._unwrap_periodic_u(uv)
        meta['method'] = np.array(['pca_bins'])
        meta['uv_raw'] = uv
        meta['uv_unwrapped'] = uv_w

        uv_mean = uv_w.mean(axis=0)
        xc = uv_w - uv_mean
        cov = np.cov(xc.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)
        e1 = eigvecs[:, order[-1]]
        e2 = eigvecs[:, order[-2]] if n > 1 else np.array([-e1[1], e1[0]], dtype=float)

        s = xc @ e1
        t = xc @ e2
        meta['uv_mean'] = uv_mean
        meta['e1'] = e1
        meta['e2'] = e2
        meta['s'] = s
        meta['t'] = t
        meta['vertex_order'] = idx

        nb = num_bins
        if nb is None:
            nb = int(max(3, min(40, round(np.sqrt(n)) + 4)))
        nb = max(3, min(60, nb))

        smin, smax = float(s.min()), float(s.max())
        if smax - smin < 1e-12:
            order_s = np.argsort(t)
            a, b = int(idx[order_s[0]]), int(idx[order_s[-1]])
            return self._dedupe_chain([a, b]), self._dedupe_chain([b, a]), meta

        edges = np.linspace(smin, smax, nb + 1)
        chain_lo: List[int] = []
        chain_hi: List[int] = []

        for b in range(nb):
            lo, hi = edges[b], edges[b + 1]
            if b == nb - 1:
                mask = (s >= lo) & (s <= hi)
            else:
                mask = (s >= lo) & (s < hi)
            if not np.any(mask):
                continue
            local_idx = np.where(mask)[0]
            t_loc = t[local_idx]
            i_lo = int(local_idx[int(np.argmin(t_loc))])
            i_hi = int(local_idx[int(np.argmax(t_loc))])
            chain_lo.append(int(idx[i_lo]))
            chain_hi.append(int(idx[i_hi]))

        chain_lo = self._dedupe_chain(chain_lo)
        chain_hi = self._dedupe_chain(chain_hi)

        if len(chain_lo) < 2 or len(chain_hi) < 2:
            order_s = np.argsort(s)
            lo_end = int(idx[order_s[0]])
            hi_end = int(idx[order_s[-1]])
            ti = int(idx[int(np.argmin(t))])
            ta = int(idx[int(np.argmax(t))])
            chain_lo = self._dedupe_chain([lo_end, hi_end])
            chain_hi = self._dedupe_chain([ti, ta])
            if len(chain_lo) < 2:
                chain_lo = [lo_end, hi_end]
            if len(chain_hi) < 2:
                chain_hi = [ti, ta] if ti != ta else [ti, lo_end]

        chain_hi = self._align_chain_pair_by_endpoints(chain_lo, chain_hi)

        if len(chain_lo) < 2 or len(chain_hi) < 2:
            chain_lo, chain_hi = self._fallback_extremal_chains_uv(idx, s, t)

        return chain_lo, chain_hi, meta

    def _fallback_extremal_chains_uv(
        self, idx: np.ndarray, s: np.ndarray, t: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """PCA 退化时沿 s、t 主轴各取两端点，保证至少各 2 个顶点（可重合）。"""
        i0, i1 = int(np.argmin(s)), int(np.argmax(s))
        j0, j1 = int(np.argmin(t)), int(np.argmax(t))
        c0 = self._dedupe_chain([int(idx[i0]), int(idx[i1])])
        c1 = self._dedupe_chain([int(idx[j0]), int(idx[j1])])
        if len(c0) < 2:
            c0 = [int(idx[i0]), int(idx[i0])]
        if len(c1) < 2:
            c1 = [int(idx[j0]), int(idx[j0])]
        return c0, c1

    def _curve_dict(
        self,
        chain_vertices: List[int],
        geodesic_fitter: GeodesicFitter,
        degree: int,
        endpoint_anchor_weight: float = 1000.0,
    ) -> Dict:
        w = [1.0] * len(chain_vertices)
        if len(w) >= 2:
            w[0] = float(endpoint_anchor_weight)
            w[-1] = float(endpoint_anchor_weight)
        cp, knots, params, quality = geodesic_fitter.fit(chain_vertices, w, self.mesh.vertices)
        return {
            'vertices': list(chain_vertices),
            'weights': w,
            'control_points': cp,
            'knots': knots,
            'params': params,
            'quality': quality,
            'degree': degree,
        }

    def fit_partition_ruled_surface(
        self,
        partition_id: int,
        partition: Set[int],
        num_control_points: Optional[int] = None,
        degree: int = 3,
        mu: float = 0.35,
        max_iterations: int = 50,
        num_bins: Optional[int] = None,
        verbose: bool = False,
        endpoint_anchor_weight: float = 1000.0,
    ) -> Dict:
        """
        对单个分区拟合直纹面（两条准线 + 线性插值）。

        ``num_control_points`` 为 None 时按链长自适应控制点数上界（且不少于 ``degree+1``）。
        """
        chain0, chain1, pca_meta = self.extract_directrix_vertex_chains(partition, num_bins=num_bins)
        if len(chain0) < 2 or len(chain1) < 2:
            idx = np.array(sorted(int(i) for i in partition), dtype=int)
            if len(idx) >= 2:
                P = self._mesh_vertices[idx]
                ax = int(np.argmax(P.max(axis=0) - P.min(axis=0)))
                o = np.argsort(P[:, ax])
                chain0 = [int(idx[int(o[0])]), int(idx[int(o[-1])])]
                ax2 = (ax + 1) % 3
                o2 = np.argsort(P[:, ax2])
                chain1 = [int(idx[int(o2[0])]), int(idx[int(o2[-1])])]
            else:
                v0 = int(next(iter(partition)))
                chain0, chain1 = [v0, v0], [v0, v0]

        cap = int(num_control_points) if num_control_points is not None else 8
        cap = max(degree + 1, cap)

        def adaptive_cp(nv: int) -> int:
            return max(degree + 1, min(cap, max(4, nv // 3)))

        nc0 = adaptive_cp(len(chain0))
        nc1 = adaptive_cp(len(chain1))

        gf0 = GeodesicFitter(
            self.nurbs,
            num_control_points=nc0,
            degree=degree,
            mu=mu,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        gf1 = GeodesicFitter(
            self.nurbs,
            num_control_points=nc1,
            degree=degree,
            mu=mu,
            max_iterations=max_iterations,
            verbose=verbose,
        )

        d0 = self._curve_dict(chain0, gf0, degree, endpoint_anchor_weight=endpoint_anchor_weight)
        d1 = self._curve_dict(chain1, gf1, degree, endpoint_anchor_weight=endpoint_anchor_weight)
        d0.setdefault('u_reversed', False)
        d1.setdefault('u_reversed', False)
        self._refine_directrix1_u_after_fit(d0, d1, degree)

        err0 = d0['quality'].get('fit_error', 0.0) or 0.0
        err1 = d1['quality'].get('fit_error', 0.0) or 0.0
        rms = self._ruled_approximation_rms(partition, d0, d1, degree)

        return {
            'partition_id': partition_id,
            'num_vertices': len(partition),
            'directrix0': d0,
            'directrix1': d1,
            'directrix_vertex_chain0': chain0,
            'directrix_vertex_chain1': chain1,
            'pca_meta': pca_meta,
            'mean_directrix_fit_error': 0.5 * (err0 + err1),
            'ruled_approximation_rms': rms,
            'directrix1_u_reversed': bool(d1.get('u_reversed', False)),
        }

    def _ruled_approximation_rms(self, partition: Set[int], d0: Dict, d1: Dict, degree: int) -> float:
        """用直纹面上较密网格点到分区顶点的最近距离 RMS 估计近似质量。"""
        if not partition:
            return 0.0
        grid = self.sample_ruled_surface_patch(d0, d1, degree, nu=28, nv=14)
        pts = self._mesh_vertices[np.array(list(partition), dtype=int)]

        diff = pts[:, None, :] - grid[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        dmin = np.sqrt(np.min(d2, axis=1))
        return float(np.sqrt(np.mean(dmin ** 2)))

    def fit_ruled_surfaces(
        self,
        partitions: List[Set[int]],
        num_control_points: Optional[int] = None,
        degree: int = 3,
        mu: float = 0.35,
        max_iterations: int = 50,
        num_bins: Optional[int] = None,
        verbose: bool = False,
        endpoint_anchor_weight: float = 1000.0,
    ) -> Dict:
        """
        对 ``NewPartitioner`` 返回的 ``partitions`` 列表逐个拟合直纹面。

        Args:
            partitions: 与 ``newPartitoner.NewPartitioner.partition_surface`` 第一个返回值相同
            num_bins: 构造准线时的参数域分箱数；None 时按分区点数自动选取
            num_control_points: 控制点数量上界；None 时按链长自适应（默认上界 8）
        """
        patches: List[Dict] = []
        sum_dir = 0.0
        sum_rms = 0.0
        n_ok = 0

        for pid, region in enumerate(partitions):
            if not region:
                print(f"  分区 {pid}: 空集，跳过", flush=True)
                continue
            if verbose:
                print(f"  分区 {pid}: {len(region)} 顶点", flush=True)
            try:
                patch = self.fit_partition_ruled_surface(
                    pid,
                    region,
                    num_control_points=num_control_points,
                    degree=degree,
                    mu=mu,
                    max_iterations=max_iterations,
                    num_bins=num_bins,
                    verbose=verbose,
                    endpoint_anchor_weight=endpoint_anchor_weight,
                )
            except Exception as ex:
                print(f"  分区 {pid}: 拟合异常 — {ex}", flush=True)
                continue
            n0 = len(patch['directrix0'].get('control_points', []))
            n1 = len(patch['directrix1'].get('control_points', []))
            if n0 < 2 or n1 < 2:
                print(
                    f"  分区 {pid}: 准线控制点过少 (n_cp0={n0}, n_cp1={n1})，跳过该分区",
                    flush=True,
                )
                continue
            patches.append(patch)
            sum_dir += patch['mean_directrix_fit_error']
            sum_rms += patch['ruled_approximation_rms']
            n_ok += 1

        if n_ok == 0:
            overall_dir = 0.0
            overall_rms = 0.0
        else:
            overall_dir = sum_dir / n_ok
            overall_rms = sum_rms / n_ok

        return {
            'patches': patches,
            'num_patches': len(patches),
            'overall_mean_directrix_fit_error': overall_dir,
            'overall_ruled_approximation_rms': overall_rms,
        }

    def _eval_surface_uv(self, u: float, v: float) -> np.ndarray:
        """在有效参数域内求 NURBS 点，过滤 NaN/Inf。"""
        u, v = self.nurbs.clamp_parameters(float(u), float(v))
        p = self.nurbs.evaluate(u, v)
        if np.all(np.isfinite(p)):
            return p
        return self.nurbs.control_points.reshape(-1, 3).mean(axis=0)

    def get_geodesic_at_t(
        self,
        control_points: np.ndarray,
        knots: np.ndarray,
        degree: int,
        t: float,
    ) -> np.ndarray:
        """在参数 t 处评估准线（设计曲面上）对应的三维点。"""
        gf = GeodesicFitter(self.nurbs)
        uv = gf.evaluate_curve(t, control_points, knots, degree)
        return self._eval_surface_uv(uv[0], uv[1])

    def sample_geodesic(
        self,
        control_points: np.ndarray,
        knots: np.ndarray,
        degree: int,
        num_samples: int = 50,
    ) -> np.ndarray:
        """沿单条准线在 [0,1] 上均匀采样三维点。"""
        gf = GeodesicFitter(self.nurbs)
        samples = np.zeros((num_samples, 3))
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0.5
            uv = gf.evaluate_curve(t, control_points, knots, degree)
            samples[i] = self._eval_surface_uv(uv[0], uv[1])
        return samples

    def evaluate_ruled_surface(
        self,
        directrix0: Dict,
        directrix1: Dict,
        u: float,
        v: float,
    ) -> np.ndarray:
        """
        计算直纹面 S(u,v)=(1-v)C0(u)+v C1(u') 上的点，u,v ∈ [0,1]。
        若 ``directrix1['u_reversed']`` 为真，则 C1 使用 u' = 1-u，与 C0 同向对齐。
        """
        deg = int(directrix0.get('degree', 3))
        uc = float(np.clip(u, 0.0, 1.0))
        u1 = (1.0 - uc) if directrix1.get('u_reversed') else uc
        p0 = self.get_geodesic_at_t(
            directrix0['control_points'], directrix0['knots'], deg, uc
        )
        p1 = self.get_geodesic_at_t(
            directrix1['control_points'], directrix1['knots'], deg, u1
        )
        vv = float(np.clip(v, 0.0, 1.0))
        return (1.0 - vv) * p0 + vv * p1

    def sample_ruled_surface_grid(
        self,
        directrix0: Dict,
        directrix1: Dict,
        degree: int,
        nu: int = 32,
        nv: int = 16,
    ) -> np.ndarray:
        """规则 (u,v) 网格上的三维点，形状 ``(nu, nv, 3)``，用于 StructuredGrid 等。"""
        grid = np.zeros((nu, nv, 3), dtype=float)
        for i in range(nu):
            u = i / (nu - 1) if nu > 1 else 0.5
            for j in range(nv):
                v = j / (nv - 1) if nv > 1 else 0.5
                grid[i, j] = self.evaluate_ruled_surface(directrix0, directrix1, u, v)
        return grid

    def sample_ruled_surface_patch(
        self,
        directrix0: Dict,
        directrix1: Dict,
        degree: int,
        nu: int = 32,
        nv: int = 16,
    ) -> np.ndarray:
        """
        在直纹面上采样规则网格点，返回形状 ``(nu * nv, 3)`` 的点云（展平）。
        """
        g = self.sample_ruled_surface_grid(directrix0, directrix1, degree, nu, nv)
        return g.reshape(-1, 3)