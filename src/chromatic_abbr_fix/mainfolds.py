from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from config import CFG
from filters import GuidedFilter


class LogRatioComputer:
    def __init__(self, threshold: float = None, epsilon: float = None):
        self._threshold = threshold if threshold != None else CFG.get_nested('defaults', 'log_threshold')
        self._epsilon = epsilon if epsilon != None else CFG.get_nested('numerical', 'epsilon_small')
    
    def __call__(self, target: NDArray, guide: NDArray) -> tuple[NDArray, NDArray]:
        target_safe = np.maximum(target, self._epsilon)
        guide_safe = np.maximum(guide, self._epsilon)
        
        log_ratio = np.log2(target_safe) - np.log2(guide_safe)
        
        abs_ratio = np.abs(log_ratio)
        high_diff_mask = abs_ratio > self._threshold
        
        weight = np.where(
            high_diff_mask,
            self._threshold / abs_ratio,
            np.ones_like(log_ratio)
        )
        
        clamped_ratio = np.clip(log_ratio, -self._threshold, self._threshold)
        weighted_ratio = np.where(high_diff_mask, clamped_ratio, log_ratio)
        
        return weighted_ratio, weight
    
    def __repr__(self) -> str:
        return f"LogRatioComputer(threshold={self._threshold})"


class ManifoldBuilder:
    def __init__(self, radius: int, refine: bool = True):
        self._radius = radius
        self._refine = refine
        self._guided_filter = GuidedFilter(radius)
        self._epsilon = CFG.get_nested('numerical', 'epsilon_tiny')
    
    def __call__(self, guide: NDArray, log_ratio: NDArray, 
                 weight: NDArray) -> tuple[NDArray, NDArray]:
        min_val = np.min(guide)
        max_val = np.max(guide)
        range_val = max_val - min_val + self._epsilon
        
        normalized_guide = (guide - min_val) / range_val
        weighted_ratio = log_ratio * weight
        
        lower_weight = (1.0 - normalized_guide) * weight
        upper_weight = normalized_guide * weight
        
        lower_num = self._guided_filter(guide, weighted_ratio * lower_weight)
        lower_den = self._guided_filter(guide, lower_weight)
        
        upper_num = self._guided_filter(guide, weighted_ratio * upper_weight)
        upper_den = self._guided_filter(guide, upper_weight)
        
        lower_manifold = lower_num / (lower_den + self._epsilon)
        upper_manifold = upper_num / (upper_den + self._epsilon)
        
        refinement_fn = lambda m: self._guided_filter(guide, m)
        apply_refinement = lambda m: refinement_fn(m) if self._refine else m
        
        return apply_refinement(lower_manifold), apply_refinement(upper_manifold)
    
    def __repr__(self) -> str:
        return f"ManifoldBuilder(radius={self._radius}, refine={self._refine})"


class CorrectionInterpolator:
    def __init__(self, epsilon: float = None):
        self._epsilon = epsilon if epsilon != None else CFG.get_nested('numerical', 'epsilon_tiny')
    
    def __call__(self, guide: NDArray, lower_manifold: NDArray,
                 upper_manifold: NDArray) -> NDArray:
        min_val = np.min(guide)
        max_val = np.max(guide)
        range_val = max_val - min_val + self._epsilon
        
        clip_min = CFG.get_nested('numerical', 'clip_min')
        normalize_threshold = CFG.get_nested('numerical', 'normalize_threshold')
        
        t = np.clip((guide - min_val) / range_val, clip_min, normalize_threshold)
        
        return (1.0 - t) * lower_manifold + t * upper_manifold
    
    def __repr__(self) -> str:
        return f"CorrectionInterpolator()"