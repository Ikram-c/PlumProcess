# a 2d implementation of the nesterov accelerated  GJK (Gilbert-Johnson-Keerthi) algorithm
# code based on -> https://github.com/AlexanderFabisch/distance3d/blob/master/distance3d/gjk/_gjk_nesterov_accelerated.py
# This script is intended to be used to check for intersections between polygons and find the distance between them
# The necessity for this in computer vision is to deal with instance segmentation models when we get overlapping polygons
# The modifications left to apply to this will be to identify the interecting polygons 
#  then-> record them so they can be thrown into a connected component labelling model to determine which object gets priority then 
# -> re-define the boundary along that edge

import numpy as np
import numba
from typing import Tuple, Callable


def gjk_nesterov_accelerated_2d_intersection(
        support1: Callable[[np.ndarray], np.ndarray],
        support2: Callable[[np.ndarray], np.ndarray]) -> bool:
    return gjk_nesterov_accelerated_2d(support1, support2)[0]


def gjk_nesterov_accelerated_2d_distance(
        support1: Callable[[np.ndarray], np.ndarray],
        support2: Callable[[np.ndarray], np.ndarray]) -> float:
    return max(gjk_nesterov_accelerated_2d(support1, support2)[1], 0.0)


def gjk_nesterov_accelerated_2d(
        support1: Callable[[np.ndarray], np.ndarray],
        support2: Callable[[np.ndarray], np.ndarray],
        max_iterations: int = 128,
        upper_bound: float = 1.79769e+308,
        tolerance: float = 1e-6,
        use_nesterov_acceleration: bool = False
) -> Tuple[bool, float, np.ndarray, int]:
    inflation = 0.0
    upper_bound += inflation
    alpha = 0.0
    inside = False
    simplex = np.zeros((3, 2), dtype=np.float64)
    simplex_len = 0
    distance = 0.0
    ray = np.array([1.0, 0.0])
    ray_len = 1.0
    ray_dir = ray.copy()
    support_point = ray.copy()

    i = 0
    while i < max_iterations:
        if ray_len < tolerance:
            distance = -inflation
            inside = True
            break

        if use_nesterov_acceleration:
            momentum = (i + 1) / (i + 3)
            y = momentum * ray + (1.0 - momentum) * support_point
            ray_dir = momentum * ray_dir + (1.0 - momentum) * y
        else:
            ray_dir = ray

        s0 = support1(-ray_dir)
        s1 = support2(ray_dir)
        simplex[simplex_len] = s0 - s1
        support_point = simplex[simplex_len]
        simplex_len += 1

        omega = ray_dir.dot(support_point) / np.linalg.norm(ray_dir)
        if omega > upper_bound:
            distance = omega - inflation
            inside = False
            break

        if use_nesterov_acceleration:
            frank_wolfe_duality_gap = 2 * ray.dot(ray - support_point)
            if frank_wolfe_duality_gap - tolerance <= 0:
                use_nesterov_acceleration = False
                simplex_len -= 1
                i += 1
                continue

        alpha = max(alpha, omega)
        diff = ray_len - alpha
        cv_check_passed = (diff - tolerance * ray_len) <= 0

        if i > 0 and cv_check_passed:
            simplex_len -= 1
            if use_nesterov_acceleration:
                use_nesterov_acceleration = False
                i += 1
                continue
            distance = ray_len - inflation
            inside = distance < tolerance
            break

        if simplex_len == 1:
            ray = support_point.copy()
        elif simplex_len == 2:
            ray, simplex_len, inside = project_line_origin_2d(simplex)
        else:
            ray, simplex_len, inside = project_triangle_origin_2d(simplex)

        if not inside:
            ray_len = np.linalg.norm(ray)
        if inside or ray_len == 0:
            distance = -inflation - 1.0
            inside = True
            break

        i += 1

    return inside, distance, simplex[:simplex_len], i


@numba.njit(cache=True)
def origin_to_point_2d(simplex: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, int]:
    simplex[0] = a.copy()
    return a.copy(), 1


@numba.njit(cache=True)
def origin_to_segment_2d(
        simplex: np.ndarray, a: np.ndarray, b: np.ndarray,
        ab: np.ndarray, ab_dot_a0: float) -> Tuple[np.ndarray, int]:
    ray = (ab.dot(b) * a + ab_dot_a0 * b) / ab.dot(ab)
    simplex[0] = b.copy()
    simplex[1] = a.copy()
    return ray, 2


@numba.njit(cache=True)
def origin_inside_triangle_2d(
        simplex: np.ndarray, a: np.ndarray, b: np.ndarray,
        c: np.ndarray) -> Tuple[np.ndarray, int, bool]:
    simplex[0] = c.copy()
    simplex[1] = b.copy()
    simplex[2] = a.copy()
    return np.zeros(2), 3, True


@numba.njit(cache=True)
def project_line_origin_2d(
        line: np.ndarray) -> Tuple[np.ndarray, int, bool]:
    a = line[1]
    b = line[0]
    ab = b - a
    d = np.dot(ab, -a)

    if d == 0:
        ray, simplex_len = origin_to_point_2d(line, a)
        return ray, simplex_len, np.all(a == 0.0)
    if d < 0:
        ray, simplex_len = origin_to_point_2d(line, a)
    else:
        ray, simplex_len = origin_to_segment_2d(line, a, b, ab, d)

    return ray, simplex_len, False


@numba.njit(cache=True)
def cross_2d(a: np.ndarray, b: np.ndarray) -> float:
    return a[0] * b[1] - a[1] * b[0]


@numba.njit(cache=True)
def triple_product_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ac = a.dot(c)
    bc = b.dot(c)
    return np.array([b[0] * ac - a[0] * bc, b[1] * ac - a[1] * bc])


@numba.njit(cache=True)
def t_b_2d(
        triangle: np.ndarray, a: np.ndarray, b: np.ndarray,
        ab: np.ndarray) -> Tuple[np.ndarray, int]:
    towards_b = ab.dot(-a)
    if towards_b < 0:
        return origin_to_point_2d(triangle, a)
    else:
        return origin_to_segment_2d(triangle, a, b, ab, towards_b)


@numba.njit(cache=True)
def project_triangle_origin_2d(
        triangle: np.ndarray) -> Tuple[np.ndarray, int, bool]:
    a = triangle[2]
    b = triangle[1]
    c = triangle[0]
    ab = b - a
    ac = c - a
    ao = -a
    ab_perp = triple_product_2d(ac, ab, ab)
    ac_perp = triple_product_2d(ab, ac, ac)

    if ab_perp.dot(ao) > 0:
        ray, simplex_len = t_b_2d(triangle, a, b, ab)
        return ray, simplex_len, False

    if ac_perp.dot(ao) > 0:
        towards_c = ac.dot(ao)
        if towards_c >= 0:
            ray, simplex_len = origin_to_segment_2d(triangle, a, c, ac, towards_c)
        else:
            ray, simplex_len = origin_to_point_2d(triangle, a)
        return ray, simplex_len, False

    return origin_inside_triangle_2d(triangle, a, b, c)


def circle_support(center: np.ndarray, radius: float) -> Callable[[np.ndarray], np.ndarray]:
    def support(direction: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return center + np.array([radius, 0.0])
        return center + radius * direction / norm
    return support


def polygon_support(vertices: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def support(direction: np.ndarray) -> np.ndarray:
        dots = vertices @ direction
        return vertices[np.argmax(dots)].copy()
    return support


def box_support_2d(center: np.ndarray, half_extents: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def support(direction: np.ndarray) -> np.ndarray:
        return center + np.sign(direction) * half_extents
    return support


def ellipse_support(center: np.ndarray, radii: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def support(direction: np.ndarray) -> np.ndarray:
        scaled = radii * direction
        norm = np.linalg.norm(scaled)
        if norm < 1e-10:
            return center + np.array([radii[0], 0.0])
        return center + radii * scaled / norm
    return support


if __name__ == "__main__":
    box1 = box_support_2d(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    box2 = box_support_2d(np.array([1.5, 0.0]), np.array([1.0, 1.0]))
    intersect, dist, simplex, iters = gjk_nesterov_accelerated_2d(box1, box2)
    print(f"Box intersection: {intersect}, distance: {dist}, iterations: {iters}")

    circle1 = circle_support(np.array([0.0, 0.0]), 1.0)
    circle2 = circle_support(np.array([3.0, 0.0]), 1.0)
    intersect, dist, simplex, iters = gjk_nesterov_accelerated_2d(circle1, circle2)
    print(f"Circle intersection: {intersect}, distance: {dist}, iterations: {iters}")

    triangle = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
    poly1 = polygon_support(triangle)
    poly2 = polygon_support(triangle + np.array([0.5, 0.5]))
    intersect, dist, simplex, iters = gjk_nesterov_accelerated_2d(poly1, poly2)
    print(f"Triangle intersection: {intersect}, distance: {dist}, iterations: {iters}")
