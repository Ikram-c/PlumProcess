from __future__ import annotations
import numpy as np

from config import CFG
from pipeline import CACorrectRGB


def create_synthetic_test_image() -> np.ndarray:
    test_cfg = CFG['synthetic_test']
    
    np.random.seed(test_cfg['random_seed'])
    h, w = test_cfg['height'], test_cfg['width']
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    sigma = test_cfg['gaussian_sigma']
    channel_fn = lambda offset: np.exp(-((r - offset) ** 2) / (2 * sigma ** 2))
    
    synthetic = np.stack([
        channel_fn(test_cfg['red_offset']), 
        channel_fn(test_cfg['green_offset']), 
        channel_fn(test_cfg['blue_offset'])
    ], axis=2)
    
    return (synthetic * test_cfg['intensity_scale']).astype(np.float64)


def run_demo():
    test_cfg = CFG['synthetic_test']
    channels = CFG['channels']
    
    synthetic = create_synthetic_test_image()
    
    corrector = CACorrectRGB(
        radius=test_cfg['demo_radius'], 
        strength=test_cfg['demo_strength'], 
        guide=channels['green']
    )
    corrected = corrector(synthetic)
    
    print(repr(corrector))
    print(str(corrector))
    print(f"Filter size: {len(corrector)}")
    print(f"Active: {bool(corrector)}")
    
    param_iter = iter(corrector)
    print(f"First param: {next(param_iter)}")
    print(f"Second param: {next(param_iter)}")
    
    print(f"Radius setting: {corrector['radius']}")
    print(f"Has strength: {'strength' in corrector}")
    
    print(f"Config loaded: {repr(CFG)}")
    
    return synthetic, corrected


if __name__ == "__main__":
    run_demo()

  CACorrectRGB(radius=10, strength=0.8, guide=0)
  CA Correction using green guide
  Filter size: 21
  Active: True
  First param: ('radius', 10)
  Second param: ('strength', 0.8)
  Radius setting: 10
  Has strength: True
  Config loaded: Config(keys=['defaults', 'channels', 'modes', ...])