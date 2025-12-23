from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from config import CFG


class ModeConstraint:
    def __init__(self, mode: int = None):
        modes = CFG['modes']
        self._mode = mode if mode != None else modes['standard']
        
        clip_min = CFG.get_nested('numerical', 'clip_min')
        
        self._constraint_map = {
            modes['standard']: lambda x: x,
            modes['brighten_only']: lambda x: np.maximum(x, clip_min),
            modes['darken_only']: lambda x: np.minimum(x, clip_min)
        }
    
    def __call__(self, correction: NDArray) -> NDArray:
        constraint_fn = self._constraint_map.get(self._mode, lambda x: x)
        return constraint_fn(correction)
    
    def __repr__(self) -> str:
        return f"ModeConstraint(mode={self._mode})"


class SafetyBlender:
    def __init__(self, safety: float = None, threshold: float = None, 
                 epsilon: float = None):
        safety_cfg = CFG['safety_blender']
        numerical_cfg = CFG['numerical']
        defaults_cfg = CFG['defaults']
        
        self._safety = safety if safety != None else defaults_cfg['safety']
        self._threshold = threshold if threshold != None else safety_cfg['threshold']
        self._epsilon = epsilon if epsilon != None else numerical_cfg['epsilon_small']
        self._blend_min = safety_cfg['blend_min']
        self._blend_max = safety_cfg['blend_max']
    
    def __call__(self, target: NDArray, corrected: NDArray, 
                 guide: NDArray) -> NDArray:
        skip_condition = self._safety <= CFG.get_nested('numerical', 'clip_min')
        
        return np.where(
            np.full_like(target, skip_condition),
            corrected,
            self._apply_safety(target, corrected, guide)
        ) if skip_condition == False else self._apply_safety(target, corrected, guide)
    
    def _apply_safety(self, target: NDArray, corrected: NDArray,
                      guide: NDArray) -> NDArray:
        original_ratio = target / (guide + self._epsilon)
        corrected_ratio = corrected / (guide + self._epsilon)
        
        ratio_change = np.abs(corrected_ratio - original_ratio)
        
        blend = np.clip(
            (ratio_change - self._threshold) / (self._threshold + self._epsilon),
            self._blend_min, 
            self._blend_max
        ) * self._safety
        
        return corrected * (1 - blend) + target * blend
    
    def __repr__(self) -> str:
        return f"SafetyBlender(safety={self._safety})"