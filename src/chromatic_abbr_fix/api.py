from __future__ import annotations
from numpy.typing import NDArray

from config import CFG
from pipeline import CACorrectRGB
from generators import param_generator


def correct_chromatic_aberration(
    image: NDArray,
    radius: int = None,
    strength: float = None,
    guide: str = 'green',
    mode: str = 'standard',
    refine: bool = True,
    safety: float = None
) -> NDArray:
    defaults = CFG['defaults']
    
    radius = radius if radius != None else defaults['radius']
    strength = strength if strength != None else defaults['strength']
    safety = safety if safety != None else defaults['safety']
    
    param_gen = param_generator(radius, strength, guide, mode, refine, safety)
    
    params = {}
    
    key, val = next(param_gen)
    params[key] = val
    
    key, val = next(param_gen)
    params[key] = val
    
    key, val = next(param_gen)
    params[key] = val
    
    key, val = next(param_gen)
    params[key] = val
    
    key, val = next(param_gen)
    params[key] = val
    
    key, val = next(param_gen)
    params[key] = val
    
    corrector = CACorrectRGB(
        radius=params['radius'],
        strength=params['strength'],
        guide=params['guide'],
        mode=params['mode'],
        refine_manifolds=params['refine'],
        safety=params['safety']
    )
    
    return corrector(image)