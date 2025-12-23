from __future__ import annotations
from pathlib import Path
import yaml


class Config:
    _instance = None
    _data = None
    
    def __new__(cls):
        has_instance = cls._instance != None
        cls._instance = cls._instance if has_instance else super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        already_loaded = self._data != None
        self._data = self._data if already_loaded else self._load()
    
    def __getitem__(self, key: str):
        return self._data.get(key)
    
    def __contains__(self, key: str) -> bool:
        return key in self._data
    
    def __repr__(self) -> str:
        return f"Config(keys={list(self._data.keys())})"
    
    def _load(self) -> dict:
        config_path = Path(__file__).parent / "config.yaml"
        
        load_fn = lambda p: yaml.safe_load(p.read_text())
        default_fn = lambda: self._default_config()
        
        return load_fn(config_path) if config_path.exists() else default_fn()
    
    def _default_config(self) -> dict:
        return {
            'defaults': {
                'radius': 5, 'strength': 1.0, 'safety': 0.5, 'log_threshold': 2.0
            },
            'channels': {'green': 0, 'red': 1, 'blue': 2, 'count': 3},
            'modes': {'standard': 0, 'brighten_only': 1, 'darken_only': 2},
            'numerical': {
                'epsilon_small': 1e-6, 'epsilon_tiny': 1e-8, 'epsilon_filter': 1e-4,
                'clip_min': 0.0, 'clip_max_multiplier': 2.0, 'normalize_threshold': 1.0
            },
            'safety_blender': {'threshold': 0.5, 'blend_min': 0, 'blend_max': 1},
            'image': {'required_dimensions': 3, 'required_channels': 3},
            'synthetic_test': {
                'height': 256, 'width': 256, 'gaussian_sigma': 50,
                'red_offset': 3, 'green_offset': 0, 'blue_offset': -3,
                'intensity_scale': 255, 'random_seed': 42,
                'demo_radius': 10, 'demo_strength': 0.8
            },
            'heap': {'radius_offset_positive': 2, 'radius_offset_negative': -2},
            'kernel': {'size_multiplier': 2, 'size_offset': 1}
        }
    
    def get_nested(self, *keys):
        result = self._data
        key_iter = iter(keys)
        
        try:
            result = result[next(key_iter)]
            result = result[next(key_iter)]
        except (KeyError, TypeError, StopIteration):
            return None
        
        return result


CFG = Config()