from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter

from config import CFG


class BoxFilter:
    def __init__(self, radius: int):
        self._radius = radius
        self._size = (
            CFG.get_nested('kernel', 'size_multiplier') * radius + 
            CFG.get_nested('kernel', 'size_offset')
        )
    
    def __call__(self, img: NDArray) -> NDArray:
        return uniform_filter(img.astype(np.float64), size=self._size, mode='reflect')
    
    def __repr__(self) -> str:
        return f"BoxFilter(radius={self._radius})"


class GuidedFilter:
    def __init__(self, radius: int, eps: float = None):
        self._radius = radius
        self._eps = eps if eps != None else CFG.get_nested('numerical', 'epsilon_filter')
        self._box = BoxFilter(radius)
    
    def __call__(self, guide: NDArray, src: NDArray) -> NDArray:
        mean_g = self._box(guide)
        mean_s = self._box(src)
        mean_gs = self._box(guide * src)
        mean_gg = self._box(guide * guide)
        
        cov_gs = mean_gs - mean_g * mean_s
        var_g = mean_gg - mean_g * mean_g
        
        a = cov_gs / (var_g + self._eps)
        b = mean_s - a * mean_g
        
        mean_a = self._box(a)
        mean_b = self._box(b)
        
        return mean_a * guide + mean_b
    
    def __repr__(self) -> str:
        return f"GuidedFilter(radius={self._radius}, eps={self._eps})"