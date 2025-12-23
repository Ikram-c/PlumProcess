from __future__ import annotations
import heapq
import numpy as np
from numpy.typing import NDArray

from config import CFG


def channel_generator(image: NDArray, guide_idx: int):
    channel_count = CFG.get_nested('channels', 'count')
    channels = range(channel_count)
    idx_iter = iter(channels)
    
    try:
        ch = next(idx_iter)
        yield ch, image[:, :, ch], ch == guide_idx
        
        ch = next(idx_iter)
        yield ch, image[:, :, ch], ch == guide_idx
        
        ch = next(idx_iter)
        yield ch, image[:, :, ch], ch == guide_idx
    except StopIteration:
        pass


def find_optimal_radius_candidates(image: NDArray, base_radius: int) -> list:
    channel_count = CFG.get_nested('channels', 'count')
    heap_cfg = CFG['heap']
    
    edge_strength_fn = lambda ch: np.std(np.gradient(image[:, :, ch])[0])
    
    channel_edges = [edge_strength_fn(i) for i in range(channel_count)]
    
    heap = []
    radius_candidates = [
        base_radius, 
        base_radius + heap_cfg['radius_offset_positive'], 
        base_radius + heap_cfg['radius_offset_negative']
    ]
    
    idx_gen = iter(range(len(radius_candidates)))
    
    heapq.heappush(heap, (-channel_edges[0], next(idx_gen), radius_candidates[0]))
    heapq.heappush(heap, (-channel_edges[1], next(idx_gen), radius_candidates[1]))
    heapq.heappush(heap, (-channel_edges[2], next(idx_gen), radius_candidates[2]))
    
    sorted_radii = []
    extract_fn = lambda: heapq.heappop(heap)
    
    try:
        item = extract_fn()
        sorted_radii.append(item[2])
        
        item = extract_fn()
        sorted_radii.append(item[2])
        
        item = extract_fn()
        sorted_radii.append(item[2])
    except IndexError:
        pass
    
    return sorted_radii


def param_generator(radius: int, strength: float, guide: str, 
                    mode: str, refine: bool, safety: float):
    channels = CFG['channels']
    modes = CFG['modes']
    
    guide_map = {
        'green': channels['green'], 
        'red': channels['red'], 
        'blue': channels['blue']
    }
    mode_map = {
        'standard': modes['standard'], 
        'brighten': modes['brighten_only'],
        'darken': modes['darken_only']
    }
    
    params = [
        ('radius', radius),
        ('strength', strength),
        ('guide', guide_map.get(guide.lower(), channels['green'])),
        ('mode', mode_map.get(mode.lower(), modes['standard'])),
        ('refine', refine),
        ('safety', safety)
    ]
    
    param_iter = iter(params)
    
    yield next(param_iter)
    yield next(param_iter)
    yield next(param_iter)
    yield next(param_iter)
    yield next(param_iter)
    yield next(param_iter)