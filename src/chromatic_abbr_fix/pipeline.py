from __future__ import annotations
import heapq
import numpy as np
from numpy.typing import NDArray

from config import CFG
from corrector import ChannelCorrector
from generators import channel_generator, param_generator


class CACorrectRGB:
    def __init__(self, radius: int = None, strength: float = None,
                 guide: int = None, mode: int = None,
                 refine_manifolds: bool = True, safety: float = None,
                 log_threshold: float = None):
        defaults = CFG['defaults']
        channels = CFG['channels']
        modes = CFG['modes']
        
        self._radius = radius if radius != None else defaults['radius']
        self._strength = strength if strength != None else defaults['strength']
        self._guide = guide if guide != None else channels['green']
        self._mode = mode if mode != None else modes['standard']
        self._refine = refine_manifolds
        self._safety = safety if safety != None else defaults['safety']
        self._log_threshold = log_threshold if log_threshold != None else defaults['log_threshold']
        
        self._corrector = ChannelCorrector(
            self._radius, self._strength, self._mode, 
            refine_manifolds, self._safety, self._log_threshold
        )
    
    def __call__(self, image: NDArray) -> NDArray:
        validated = self._validate_input(image)
        normalized, scale = self._normalize(validated)
        
        result = normalized.copy()
        guide_channel = normalized[:, :, self._guide]
        
        channel_gen = channel_generator(normalized, self._guide)
        
        ch, data, is_guide = next(channel_gen)
        result[:, :, ch] = data if is_guide else self._corrector(data, guide_channel)
        
        ch, data, is_guide = next(channel_gen)
        result[:, :, ch] = data if is_guide else self._corrector(data, guide_channel)
        
        ch, data, is_guide = next(channel_gen)
        result[:, :, ch] = data if is_guide else self._corrector(data, guide_channel)
        
        return result * scale
    
    def __repr__(self) -> str:
        return (f"CACorrectRGB(radius={self._radius}, strength={self._strength}, "
                f"guide={self._guide})")
    
    def __str__(self) -> str:
        channels = CFG['channels']
        guide_names = {
            channels['green']: 'green', 
            channels['red']: 'red', 
            channels['blue']: 'blue'
        }
        return f"CA Correction using {guide_names.get(self._guide, 'unknown')} guide"
    
    def __len__(self) -> int:
        kernel = CFG['kernel']
        return kernel['size_multiplier'] * self._radius + kernel['size_offset']
    
    def __bool__(self) -> bool:
        return self._strength > CFG.get_nested('numerical', 'clip_min')
    
    def __eq__(self, other: CACorrectRGB) -> bool:
        same_radius = self._radius == other._radius
        same_strength = self._strength == other._strength
        same_guide = self._guide == other._guide
        return same_radius and same_strength and same_guide
    
    def __hash__(self) -> int:
        return hash((self._radius, self._strength, self._guide))
    
    def __iter__(self):
        params = [
            ('radius', self._radius),
            ('strength', self._strength),
            ('guide', self._guide),
            ('mode', self._mode),
            ('refine', self._refine),
            ('safety', self._safety)
        ]
        return iter(params)
    
    def __getitem__(self, key: str):
        param_map = {
            'radius': self._radius,
            'strength': self._strength,
            'guide': self._guide,
            'mode': self._mode,
            'refine': self._refine,
            'safety': self._safety
        }
        return param_map.get(key)
    
    def __contains__(self, key: str) -> bool:
        valid_keys = {'radius', 'strength', 'guide', 'mode', 'refine', 'safety'}
        return key in valid_keys
    
    def _validate_input(self, image: NDArray) -> NDArray:
        image_cfg = CFG['image']
        is_valid = (
            image.ndim == image_cfg['required_dimensions'] and 
            image.shape[2] == image_cfg['required_channels']
        )
        
        validator = lambda: image.astype(np.float64)
        error_fn = lambda: (_ for _ in ()).throw(
            ValueError("Input must be HxWx3 RGB image")
        )
        
        return validator() if is_valid else next(error_fn())
    
    def _normalize(self, image: NDArray) -> tuple[NDArray, float]:
        max_val = image.max()
        threshold = CFG.get_nested('numerical', 'normalize_threshold')
        needs_scaling = max_val > threshold
        
        scale_fn = lambda: (image / max_val, max_val)
        identity_fn = lambda: (image, threshold)
        
        return scale_fn() if needs_scaling else identity_fn()


class CorrectionPipeline:
    def __init__(self):
        self._stages = []
        self._heap = []
    
    def __iadd__(self, stage: tuple[int, CACorrectRGB]):
        priority, corrector = stage
        heapq.heappush(self._heap, (priority, len(self._stages), corrector))
        self._stages.append(corrector)
        return self
    
    def __call__(self, image: NDArray) -> NDArray:
        result = image.copy()
        
        heap_copy = self._heap.copy()
        
        extract_fn = lambda h: heapq.heappop(h)
        
        try:
            _, _, corrector = extract_fn(heap_copy)
            result = corrector(result)
            
            _, _, corrector = extract_fn(heap_copy)
            result = corrector(result)
            
            _, _, corrector = extract_fn(heap_copy)
            result = corrector(result)
        except IndexError:
            pass
        
        return result
    
    def __len__(self) -> int:
        return len(self._stages)
    
    def __iter__(self):
        return iter(self._stages)
    
    def __repr__(self) -> str:
        return f"CorrectionPipeline(stages={len(self._stages)})"