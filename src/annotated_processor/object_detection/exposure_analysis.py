import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Union
import warnings
from image_utils import to_grayscale, validate_image

warnings.filterwarnings('ignore')


class ExposureAnalyzer:
    def __init__(self, overexposure_threshold: float = 240,
                 underexposure_threshold: float = 15,
                 significant_percentage: float = 2.0):
        self.overexposure_threshold = overexposure_threshold
        self.underexposure_threshold = underexposure_threshold
        self.significant_percentage = significant_percentage
        self.results = {}

    def analyze_exposure(self, image: np.ndarray, image_id: str) -> Dict:
        validate_image(image)

        luminance_metrics = self._calculate_luminance_metrics(image)
        exposure_detection = self._detect_exposure_issues(image)
        histogram_metrics = self._calculate_histogram_metrics(image)
        dynamic_range_metrics = self._calculate_dynamic_range(image)
        recommendations = self._generate_exposure_recommendations(image)

        analysis_results = {
            **luminance_metrics,
            **exposure_detection,
            **histogram_metrics,
            **dynamic_range_metrics,
            **recommendations
        }

        self.results[image_id] = analysis_results
        return analysis_results

    def _calculate_luminance_metrics(self, image: np.ndarray) -> Dict:
        if image.ndim == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
        else:
            luminance = l_channel = v_channel = image.astype(float)

        return {
            'avg_luminance': float(np.mean(luminance)),
            'median_luminance': float(np.median(luminance)),
            'std_luminance': float(np.std(luminance)),
            'avg_brightness_lab': float(np.mean(l_channel)),
            'avg_brightness_hsv': float(np.mean(v_channel)),
            'min_luminance': float(np.min(luminance)),
            'max_luminance': float(np.max(luminance)),
            'luminance_range': float(np.max(luminance) - np.min(luminance))
        }

    def _detect_exposure_issues(self, image: np.ndarray) -> Dict:
        gray = to_grayscale(image)
        total_pixels = gray.size

        overexposed_pixels = np.sum(gray >= self.overexposure_threshold)
        underexposed_pixels = np.sum(gray <= self.underexposure_threshold)

        over_pct = (overexposed_pixels / total_pixels) * 100
        under_pct = (underexposed_pixels / total_pixels) * 100

        channel_clipping = {}
        if image.ndim == 3:
            channels = cv2.split(image)
            for ch_name, ch_data in zip(['blue', 'green', 'red'], channels):
                clipped_high = np.sum(ch_data >= 250)
                clipped_low = np.sum(ch_data <= 5)
                channel_clipping[f'{ch_name}_clipped_high_percent'] = float((clipped_high / total_pixels) * 100)
                channel_clipping[f'{ch_name}_clipped_low_percent'] = float((clipped_low / total_pixels) * 100)

        return {
            'overexposed_pixels': int(overexposed_pixels),
            'overexposure_percentage': float(over_pct),
            'is_overexposed': over_pct > self.significant_percentage,
            'underexposed_pixels': int(underexposed_pixels),
            'underexposure_percentage': float(under_pct),
            'is_underexposed': under_pct > self.significant_percentage,
            'has_exposure_issues': (over_pct > self.significant_percentage or under_pct > self.significant_percentage),
            **channel_clipping
        }

    def _calculate_histogram_metrics(self, image: np.ndarray) -> Dict:
        gray = to_grayscale(image)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_norm = hist / np.sum(hist)

        shadows = np.sum(hist_norm[:85])
        midtones = np.sum(hist_norm[85:171])
        highlights = np.sum(hist_norm[171:])

        bins = np.arange(256)
        weighted_avg = np.sum(bins * hist_norm)
        entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0]))

        return {
            'shadows_percentage': float(shadows * 100),
            'midtones_percentage': float(midtones * 100),
            'highlights_percentage': float(highlights * 100),
            'histogram_center': float(weighted_avg),
            'histogram_entropy': float(entropy),
            'histogram_peak': int(np.argmax(hist)),
            'tonal_distribution': 'shadows-heavy' if shadows > 0.5 else
                                  'highlights-heavy' if highlights > 0.5 else 'balanced'
        }

    def _calculate_dynamic_range(self, image: np.ndarray) -> Dict:
        gray = to_grayscale(image)

        p1 = np.percentile(gray, 1)
        p99 = np.percentile(gray, 99)
        p5 = np.percentile(gray, 5)
        p95 = np.percentile(gray, 95)

        full_range = np.max(gray) - np.min(gray)
        effective_range_99 = p99 - p1
        effective_range_95 = p95 - p5
        range_utilization = (effective_range_99 / 255.0) * 100

        return {
            'dynamic_range_full': float(full_range),
            'dynamic_range_99p': float(effective_range_99),
            'dynamic_range_95p': float(effective_range_95),
            'range_utilization_percent': float(range_utilization),
            'p1_percentile': float(p1),
            'p99_percentile': float(p99),
            'has_good_dynamic_range': range_utilization > 70
        }

    def _generate_exposure_recommendations(self, image: np.ndarray) -> Dict:
        gray = to_grayscale(image)

        avg_brightness = np.mean(gray)
        over_pct = (np.sum(gray >= self.overexposure_threshold) / gray.size) * 100
        under_pct = (np.sum(gray <= self.underexposure_threshold) / gray.size) * 100

        exposure_comp = 0.0
        recs = []

        if over_pct > self.significant_percentage:
            if over_pct > 10:
                exposure_comp = -2.0
                recs.append("Significantly overexposed - reduce exposure by 2+ stops")
            elif over_pct > 5:
                exposure_comp = -1.0
                recs.append("Overexposed - reduce exposure by 1 stop")
            else:
                exposure_comp = -0.5
                recs.append("Slightly overexposed - reduce exposure by 0.5 stops")
        elif under_pct > self.significant_percentage:
            if under_pct > 10:
                exposure_comp = 2.0
                recs.append("Significantly underexposed - increase exposure by 2+ stops")
            elif under_pct > 5:
                exposure_comp = 1.0
                recs.append("Underexposed - increase exposure by 1 stop")
            else:
                exposure_comp = 0.5
                recs.append("Slightly underexposed - increase exposure by 0.5 stops")
        elif avg_brightness < 85:
            exposure_comp = 0.5
            recs.append("Image appears dark - consider increasing exposure")
        elif avg_brightness > 170:
            exposure_comp = -0.5
            recs.append("Image appears bright - consider reducing exposure")
        else:
            recs.append("Exposure appears well-balanced")

        if len(recs) == 1 and "well-balanced" in recs[0]:
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            if np.sum(hist[:50]) > np.sum(hist[200:]):
                recs.append("Consider slight exposure increase for better shadow detail")
            elif np.sum(hist[200:]) > np.sum(hist[:50]):
                recs.append("Consider slight exposure decrease to prevent highlight clipping")

        return {
            'exposure_compensation_stops': float(exposure_comp),
            'recommendations': recs,
            'overall_exposure_quality': self._assess_exposure_quality(over_pct, under_pct, avg_brightness)
        }

    def _assess_exposure_quality(self, over_pct: float, under_pct: float, avg_brightness: float) -> str:
        if over_pct > 10 or under_pct > 10:
            return "Poor"
        elif over_pct > 5 or under_pct > 5:
            return "Fair"
        elif over_pct < 1 and under_pct < 1 and 85 <= avg_brightness <= 170:
            return "Excellent"
        else:
            return "Good"

    def get_results(self) -> Dict:
        return self.results.copy()

    def get_analysis(self, image_id: str) -> Optional[Dict]:
        return self.results.get(image_id)

    def clear_results(self) -> None:
        self.results.clear()


# Standalone functions
def calculate_luminance(image: np.ndarray) -> Dict:
    return ExposureAnalyzer()._calculate_luminance_metrics(image)

def detect_exposure_issues(image: np.ndarray,
                           overexposure_threshold: float = 240,
                           underexposure_threshold: float = 15,
                           significant_percentage: float = 2.0) -> Dict:
    return ExposureAnalyzer(overexposure_threshold, underexposure_threshold, significant_percentage)._detect_exposure_issues(image)

def calculate_exposure_compensation(image: np.ndarray) -> Tuple[float, str]:
    r = ExposureAnalyzer()._generate_exposure_recommendations(image)
    return r['exposure_compensation_stops'], r['overall_exposure_quality']

def get_brightness_summary(image: np.ndarray) -> Dict:
    analyzer = ExposureAnalyzer()
    luminance = analyzer._calculate_luminance_metrics(image)
    histogram = analyzer._calculate_histogram_metrics(image)
    return {
        'average_brightness': luminance['avg_luminance'],
        'brightness_distribution': histogram['tonal_distribution'],
        'dynamic_range_utilization': analyzer._calculate_dynamic_range(image)['range_utilization_percent'],
        'histogram_center': histogram['histogram_center']
    }