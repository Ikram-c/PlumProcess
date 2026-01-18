from exposure_analyser import ExposureAnalyzer
from texture_analyser import TextureAnalyzer
from spatial_frequency_analysis import SpatialFrequencyAnalyzer
from laplacian_variance import LaplacianVariance
import cv2


def analyze_image_quality(image_path: str) -> dict:
    """
    Run all core image quality analyses on a given image.

    Args:
        image_path (str): Path to image file

    Returns:
        dict: Dictionary containing results from each quality analysis module
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at path: {image_path}")

    image_id = os.path.basename(image_path)

    results = {}

    # Exposure analysis
    try:
        exposure_analyzer = ExposureAnalyzer()
        results['exposure'] = exposure_analyzer.analyze_exposure(image, image_id)
    except Exception as e:
        results['exposure'] = {'error': str(e)}

    # Texture analysis
    try:
        texture_analyzer = TextureAnalyzer()
        results['texture'] = texture_analyzer.analyze_texture(image, image_id)
    except Exception as e:
        results['texture'] = {'error': str(e)}

    # Spatial frequency analysis
    try:
        freq_analyzer = SpatialFrequencyAnalyzer()
        results['frequency'] = freq_analyzer.analyze_frequency(image, image_id)
    except Exception as e:
        results['frequency'] = {'error': str(e)}

    # Laplacian variance
    try:
        lap_var_analyzer = LaplacianVariance()
        results['laplacian_variance'] = lap_var_analyzer.calculate(image)
    except Exception as e:
        results['laplacian_variance'] = {'error': str(e)}

    return results