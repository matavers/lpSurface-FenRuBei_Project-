import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.nurbsProcessor import NURBSProcessor


def test_plane():
    """2x2 控制点，次数 1x1，应生成平面四边形"""
    cp = np.array([[[0, 0, 0], [0, 1, 0]],
                   [[1, 0, 0], [1, 1, 0]]])
    knots_u = np.array([0, 0, 1, 1])
    knots_v = np.array([0, 0, 1, 1])
    plane = NURBSProcessor(cp, knots_u, knots_v, 1, 1)

    assert np.allclose(plane.evaluate(0, 0), [0, 0, 0]), f"Corner (0,0) failed: {plane.evaluate(0, 0)}"
    assert np.allclose(plane.evaluate(1, 1), [1, 1, 0]), f"Corner (1,1) failed: {plane.evaluate(1, 1)}"
    assert np.allclose(plane.evaluate(0.5, 0.5), [0.5, 0.5, 0]), f"Center (0.5,0.5) failed: {plane.evaluate(0.5, 0.5)}"
    print("test_plane PASSED")
    return plane


def test_cylinder():
    """测试圆柱面基本属性"""
    cyl = NURBSProcessor.create_cylinder(radius=1.0, height=2.0)

    p0 = cyl.evaluate(0, 0)
    p1 = cyl.evaluate(0.25, 1)
    p2 = cyl.evaluate(0.5, 0.5)

    r0 = np.sqrt(p0[0]**2 + p0[1]**2)
    r1 = np.sqrt(p1[0]**2 + p1[1]**2)
    r2 = np.sqrt(p2[0]**2 + p2[1]**2)

    assert np.isclose(r0, 1.0, atol=0.2), f"Radius at u=0 should be 1.0, got {r0}"
    assert np.isclose(r1, 1.0, atol=0.2), f"Radius at u=0.25 should be 1.0, got {r1}"
    assert np.isclose(r2, 1.0, atol=0.2), f"Radius at u=0.5 should be 1.0, got {r2}"

    assert np.isclose(p0[2], 0.0, atol=1e-5), f"Height at v=0 should be 0, got {p0[2]}"
    assert np.isclose(p1[2], 2.0, atol=1e-5), f"Height at v=1 should be 2, got {p1[2]}"

    print("test_cylinder PASSED")
    print(f"  Cylinder test points: p0={p0}, p1={p1}, p2={p2}")
    return cyl


def test_sphere():
    """测试球面"""
    sphere = NURBSProcessor.create_sphere(radius=1.0, resolution=20)

    north = sphere.evaluate(0, 0.5)
    equator = sphere.evaluate(0.5, 0)

    r_north = np.linalg.norm(north)
    r_equator = np.linalg.norm(equator)

    assert np.isclose(r_north, 1.0, atol=0.2), f"North pole radius should be 1.0, got {r_north}"
    assert np.isclose(r_equator, 1.0, atol=0.2), f"Equator radius should be 1.0, got {r_equator}"
    assert np.isclose(north[2], 1.0, atol=0.2), f"North pole z should be 1.0, got {north[2]}"

    print("test_sphere PASSED")
    print(f"  North pole: {north}, Equator: {equator}")
    return sphere


def test_cone():
    """测试圆锥面"""
    cone = NURBSProcessor.create_cone(radius=1.0, height=2.0, resolution_u=20, resolution_v=10)

    apex = cone.evaluate(0, 1)
    base = cone.evaluate(0, 0)

    r_apex = np.sqrt(apex[0]**2 + apex[1]**2)
    r_base = np.sqrt(base[0]**2 + base[1]**2)

    assert np.isclose(r_apex, 0.0, atol=1e-5), f"Apex radius should be 0, got {r_apex}"
    assert np.isclose(r_base, 1.0, atol=0.2), f"Base radius should be 1.0, got {r_base}"
    assert np.isclose(apex[2], 2.0, atol=1e-5), f"Apex z should be 2.0, got {apex[2]}"
    assert np.isclose(base[2], 0.0, atol=1e-5), f"Base z should be 0, got {base[2]}"

    print("test_cone PASSED")
    print(f"  Apex: {apex}, Base: {base}")
    return cone


def verify_knot_vector_lengths():
    """验证节点向量长度是否正确"""
    n_u, n_v = 9, 2
    degree_u, degree_v = 3, 1
    expected_len_u = n_u + degree_u + 1
    expected_len_v = n_v + degree_v + 1

    cyl = NURBSProcessor.create_cylinder()
    assert len(cyl.knots_u) == expected_len_u, f"Cylinder knots_u: expected {expected_len_u}, got {len(cyl.knots_u)}"
    assert len(cyl.knots_v) == expected_len_v, f"Cylinder knots_v: expected {expected_len_v}, got {len(cyl.knots_v)}"

    sphere = NURBSProcessor.create_sphere(resolution=20)
    n_u_s, n_v_s = 20, 20
    expected_len_u_s = n_u_s + 3 + 1
    expected_len_v_s = n_v_s + 3 + 1
    assert len(sphere.knots_u) == expected_len_u_s, f"Sphere knots_u: expected {expected_len_u_s}, got {len(sphere.knots_u)}"
    assert len(sphere.knots_v) == expected_len_v_s, f"Sphere knots_v: expected {expected_len_v_s}, got {len(sphere.knots_v)}"

    print("verify_knot_vector_lengths PASSED")


if __name__ == "__main__":
    print("=" * 50)
    print("Running NURBS Processor Tests")
    print("=" * 50)

    test_plane()
    print()
    test_cylinder()
    print()
    verify_knot_vector_lengths()
    print()
    test_sphere()
    print()
    test_cone()

    print()
    print("=" * 50)
    print("All tests PASSED!")
    print("=" * 50)