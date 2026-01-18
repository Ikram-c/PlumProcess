import numpy as np
import cv2
from enum import Enum
from typing import Tuple, Union, Dict, Optional, Any

from image_utils import to_grayscale, validate_image
# Optional: from image_utils import to_normalized_grayscale

class FilterType(Enum):
    LOW_PASS = "low_pass"
    HIGH_PASS = "high_pass"
    BAND_PASS = "band_pass"
    BAND_STOP = "band_stop"


class SpatialFrequencyAnalyzer:
    def __init__(self):
        self.results = {}

    def analyze_frequency(self, image: np.ndarray, image_id: str) -> Dict[str, Any]:
        validate_image(image)
        gray = to_grayscale(image)
        # Optional: gray = to_normalized_grayscale(image)

        fft_shifted, magnitude, phase = self.compute_2d_fft(gray)
        freq_metrics = self._compute_frequency_metrics(gray, magnitude)
        directional_metrics = self._compute_directional_energy(magnitude, gray.shape)
        centroid = self._compute_frequency_centroid(magnitude, directional_metrics["distance_matrix"])

        result = {
            'fft_result': fft_shifted,
            'magnitude_spectrum': magnitude,
            'log_magnitude_spectrum': np.log(magnitude + 1),
            'phase_spectrum': phase,
            'radial_profile': freq_metrics["radial_profile"],
            'low_freq_energy_ratio': freq_metrics["low_freq_energy_ratio"],
            'mid_freq_energy_ratio': freq_metrics["mid_freq_energy_ratio"],
            'high_freq_energy_ratio': freq_metrics["high_freq_energy_ratio"],
            'directional_energy': directional_metrics["directional_energy"],
            'dominant_direction': directional_metrics["dominant_direction"],
            'frequency_centroid': centroid
        }

        self.results[image_id] = result
        return result

    def compute_2d_fft(self, image: np.ndarray, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if normalize:
            image = image.astype(np.float32) / 255.0
        fft = np.fft.fftshift(np.fft.fft2(image))
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        return fft, magnitude, phase

    def _compute_frequency_metrics(self, image: np.ndarray, magnitude: np.ndarray) -> Dict:
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        U, V = np.meshgrid(v, u)
        D = np.sqrt(U ** 2 + V ** 2)
        max_radius = int(np.sqrt(crow ** 2 + ccol ** 2))

        radial_profile = np.zeros(max_radius)
        radial_count = np.zeros(max_radius)

        for i in range(rows):
            for j in range(cols):
                r = int(D[i, j])
                if r < max_radius:
                    radial_profile[r] += magnitude[i, j]
                    radial_count[r] += 1

        radial_count[radial_count == 0] = 1
        radial_profile /= radial_count

        total_energy = np.sum(magnitude ** 2)
        low_mask = D <= max_radius * 0.25
        high_mask = D >= max_radius * 0.75
        mid_mask = (D > max_radius * 0.25) & (D < max_radius * 0.75)

        return {
            'radial_profile': radial_profile,
            'low_freq_energy_ratio': float(np.sum(magnitude[low_mask] ** 2) / total_energy),
            'mid_freq_energy_ratio': float(np.sum(magnitude[mid_mask] ** 2) / total_energy),
            'high_freq_energy_ratio': float(np.sum(magnitude[high_mask] ** 2) / total_energy)
        }

    def _compute_directional_energy(self, magnitude: np.ndarray, shape: Tuple[int, int]) -> Dict:
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        U, V = np.meshgrid(v, u)
        angles = np.arctan2(V, U)
        bins = np.linspace(-np.pi, np.pi, 36)
        directional_energy = np.zeros(len(bins) - 1)

        for i in range(len(bins) - 1):
            mask = (angles >= bins[i]) & (angles < bins[i + 1])
            directional_energy[i] = np.sum(magnitude[mask] ** 2)

        return {
            'directional_energy': directional_energy,
            'dominant_direction': float(np.rad2deg(bins[np.argmax(directional_energy)])),
            'distance_matrix': np.sqrt(U ** 2 + V ** 2)
        }

    def _compute_frequency_centroid(self, magnitude: np.ndarray, distance_matrix: np.ndarray) -> float:
        total_mag = np.sum(magnitude)
        if total_mag == 0:
            return 0.0
        return float(np.sum(magnitude * distance_matrix) / total_mag)

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        return self.results.copy()

    def get_analysis(self, image_id: str) -> Optional[Dict[str, Any]]:
        return self.results.get(image_id)

    def clear_results(self) -> None:
        self.results.clear()