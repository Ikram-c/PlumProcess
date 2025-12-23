from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from config import CFG
from manifolds import LogRatioComputer, ManifoldBuilder, CorrectionInterpolator
from blending import ModeConstraint, SafetyBlender


class ChannelCorrector:
    def __init__(self, radius: int = None, strength: float = None,
                 mode: int = None, refine: bool = True,
                 safety: float = None, log_threshold: float = None):
        defaults = CFG['defaults']
        
        self._radius = radius if radius != None else defaults['radius']
        self._strength = strength if strength != None else defaults['strength']
        self._mode = mode if mode != None else CFG['modes']['standard']
        self._refine = refine
        self._safety = safety if safety != None else defaults['safety']
        self._log_threshold = log_threshold if log_threshold != None else defaults['log_threshold']
        
        self._log_ratio = LogRatioComputer(self._log_threshold)
        self._manifold_builder = ManifoldBuilder(self._radius, refine)
        self._interpolator = CorrectionInterpolator()
        self._mode_constraint = ModeConstraint(self._mode)
        self._safety_blender = SafetyBlender(self._safety)
    
    def __call__(self, target: NDArray, guide: NDArray) -> NDArray:
        log_ratio, weight = self._log_ratio(target, guide)
        
        lower_manifold, upper_manifold = self._manifold_builder(
            guide, log_ratio, weight
        )
        
        correction = self._interpolator(guide, lower_manifold, upper_manifold)
        correction = self._mode_constraint(correction)
        correction = correction * self._strength
        
        corrected = guide * np.power(2.0, correction)
        
        apply_safety = lambda: self._safety_blender(target, corrected, guide)
        skip_safety = lambda: corrected
        
        clip_min = CFG.get_nested('numerical', 'clip_min')
        clip_max_mult = CFG.get_nested('numerical', 'clip_max_multiplier')
        
        result = apply_safety() if self._safety > clip_min else skip_safety()
        
        return np.clip(result, clip_min, np.max(target) * clip_max_mult)
    
    def __repr__(self) -> str:
        return (f"ChannelCorrector(radius={self._radius}, "
                f"strength={self._strength})")