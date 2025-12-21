from typing import List, Tuple
import numpy as np
from PIL import Image

class PolygonGenerator:
    def __init__(
        self,
        num_vertices: int,
        min_radius: float,
        max_radius: float,
        center: Tuple[float, float],
        angle_variation: float,
    ):
        self.num_vertices = num_vertices
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.center = center
        self.angle_variation = angle_variation
        self.cartesian_points: np.ndarray = np.array([])

    def generate_polygon(self) -> np.ndarray:
        base_angles = np.linspace(0, 2 * np.pi, self.num_vertices, endpoint=False)
        radii = np.random.uniform(self.min_radius, self.max_radius, self.num_vertices)
        angle_var_abs = self.angle_variation * (2 * np.pi / self.num_vertices)
        angle_offsets = np.random.uniform(-angle_var_abs, angle_var_abs, self.num_vertices)
        angles = base_angles + angle_offsets
        x_coords = radii * np.cos(angles) + self.center[0]
        y_coords = radii * np.sin(angles) + self.center[1]
        self.cartesian_points = np.stack((x_coords, y_coords), axis=1)
        return self.cartesian_points

    def get_bounding_box(self) -> List[float]:
        if self.cartesian_points.size == 0: self.generate_polygon()
        min_coords = self.cartesian_points.min(axis=0)
        max_coords = self.cartesian_points.max(axis=0)
        min_x, min_y = min_coords
        max_x, max_y = max_coords
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def get_area(self) -> float:
        if self.cartesian_points.shape[0] < 3: return 0.0
        x, y = self.cartesian_points[:, 0], self.cartesian_points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
