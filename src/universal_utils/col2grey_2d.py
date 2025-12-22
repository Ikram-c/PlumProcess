import cv2
import numpy as np
from csbdeep.utils import normalize

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts a BGR/RGB image to grayscale if needed.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Grayscale image.
    """
    validate_image(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

def to_normalized_grayscale(image: np.ndarray, percentile_low: float = 1.0, percentile_high: float = 99.8) -> np.ndarray:
    """
    Converts an image to grayscale and applies percentile normalization using CSBDeep's `normalize()`.

    Args:
        image (np.ndarray): Input image (BGR or grayscale).
        percentile_low (float): Lower percentile for normalization.
        percentile_high (float): Upper percentile for normalization.

    Returns:
        np.ndarray: Normalized grayscale image (uint8).
    """
    gray = to_grayscale(image)
    norm_img = normalize(gray, percentile_low, percentile_high, axis=(0, 1))
    return (norm_img * 255).astype(np.uint8)
