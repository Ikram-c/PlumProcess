import numpy as np
from scipy.ndimage import gaussian_filter
from typing import List, Tuple
from src.heatmap_visuals.heatmap_models import BoundingBox
from src.heatmap_visuals.heatmap_config_loader import Config

class HeatmapData:
    def __init__(self, boxes: List[BoundingBox], config: Config):
        self._boxes = boxes
        self._config = config
        self._centers = self._compute_centers()
        self._hist: np.ndarray | None = None
        self._xedges: np.ndarray | None = None
        self._yedges: np.ndarray | None = None
        self._compute_histogram()

    def _compute_centers(self) -> np.ndarray:
        centers = np.zeros((len(self._boxes), 2))
        for i, box in enumerate(self._boxes):
            centers[i, 0] = box.center_x
            centers[i, 1] = box.center_y
        return centers

    def _compute_histogram(self) -> None:
        canvas = self._config["canvas"]
        bins_x = int(canvas["width"] / self._config["histogram"]["bin_divisor"])
        bins_y = int(canvas["height"] / self._config["histogram"]["bin_divisor"])
        
        self._hist, self._xedges, self._yedges = np.histogram2d(
            self._centers[:, 0],
            self._centers[:, 1],
            bins=[bins_x, bins_y],
            range=[[0, canvas["width"]], [0, canvas["height"]]],
        )

    def __len__(self) -> int:
        return len(self._boxes)

    @property
    def centers(self) -> np.ndarray:
        return self._centers

    @property
    def histogram(self) -> np.ndarray:
        if self._hist is None:
            raise ValueError("Histogram not computed")
        return self._hist

    @property
    def x_centers(self) -> np.ndarray:
        if self._xedges is None:
            raise ValueError("Edges not computed")
        return (self._xedges[:-1] + self._xedges[1:]) / 2

    @property
    def y_centers(self) -> np.ndarray:
        if self._yedges is None:
            raise ValueError("Edges not computed")
        return (self._yedges[:-1] + self._yedges[1:]) / 2

    def smoothed_histogram(self, sigma: float) -> np.ndarray:
        return gaussian_filter(self._hist, sigma=sigma)