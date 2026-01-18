# updated_main.py
import os

import cv2

from anno_info import CocoAnnotationAnalyzer
from annotation_transformer import AnnotationTransformer
from exposure_analyser import ExposureAnalyzer
from image_feature_matcher import ImageFeatureMatcher
from img_metadata import Metadata_Extractor
from laplacian_variance import LaplacianVariance
from lens_fix import lens_fix
from spatial_frequency_analysis import SpatialFrequencyAnalyzer
from texture_analyser import TextureAnalyzer


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


# Load image dataset
def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


# Get annotation features and store them
def run_base_anno_analysis(json_path, numerical_props=[], categorical_props=[]):
    analyzer = CocoAnnotationAnalyzer(json_path)
    analysis_results = {
        'properties': analyzer.discover_properties(),
        'basic_statistics': analyzer.get_basic_statistics(),
        'numerical_analyses': {},
        'categorical_analyses': {}
    }

    for prop in numerical_props:
        analysis_results['numerical_analyses'][prop] = analyzer.analyze_numerical_property(prop)

    for prop in categorical_props:
        analysis_results['categorical_analyses'][prop] = analyzer.analyze_categorical_property(prop)

    return analysis_results


def process_single_image(im_name, output_dir, json_path, lfix_arg=True):
    metadata_obj = Metadata_Extractor(im_name)
    metadata = metadata_obj.run_metadata_extractor()

    # Extract image name and dataset info for ImageFeatureMatcher
    image_name = os.path.splitext(os.path.basename(im_name))[0]
    dataset_name = "lens_corrected" if "output_dir" in im_name else "original"

    image_feature_matcher = ImageFeatureMatcher(
        cv2.imread(im_name),
        dataset_name=dataset_name,
        image_name=image_name
    )
    image_features = image_feature_matcher.analyze_and_save_all_data(output_dir)

    if lfix_arg:
        anno_base_stats = run_base_anno_analysis(json_path)
        lens_fix(im_name, metadata, output_dir)
        return [image_features, metadata, anno_base_stats]
    else:
        return [image_features, metadata]


def find_corresponding_images(original_images, corrected_dir):
    """
    Find corresponding original and corrected image pairs.

    Args:
        original_images: List of original image paths
        corrected_dir: Directory containing corrected images

    Returns:
        List of (original_path, corrected_path) tuples
    """
    corrected_images = list(absoluteFilePaths(corrected_dir))
    image_pairs = []

    for orig_path in original_images:
        orig_filename = os.path.basename(orig_path)
        # Find corresponding corrected image
        for corr_path in corrected_images:
            corr_filename = os.path.basename(corr_path)
            # You may need to adjust this matching logic based on your naming convention
            if orig_filename == corr_filename or orig_filename.split('.')[0] in corr_filename:
                image_pairs.append((orig_path, corr_path))
                break

    return image_pairs


def transform_annotations_for_dataset(image_pairs, json_path, output_dir):
    """
    Transform annotations for all image pairs in the dataset.

    Args:
        image_pairs: List of (original_path, corrected_path) tuples
        json_path: Path to COCO JSON annotations
        output_dir: Directory to save transformed annotations

    Returns:
        List of paths to transformed annotation files
    """
    transformed_annotations = []

    # Create subdirectories for outputs
    transformed_anno_dir = os.path.join(output_dir, "transformed_annotations")
    visualization_dir = os.path.join(output_dir, "transformation_visualizations")
    os.makedirs(transformed_anno_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    for i, (original_path, corrected_path) in enumerate(image_pairs):
        try:
            print(f"Processing image pair {i + 1}/{len(image_pairs)}: {os.path.basename(original_path)}")

            # Generate output paths
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            output_json_path = os.path.join(transformed_anno_dir, f"{base_name}_transformed.json")
            visualization_path = os.path.join(visualization_dir, f"{base_name}_matches.png")

            # Transform annotations for this image pair
            transformer = AnnotationTransformer(original_path, corrected_path, json_path)

            # Compute transformation matrix
            transformation_matrix = transformer.compute_transformation_matrix()
            print(f"  Computed transformation matrix with shape: {transformation_matrix.shape}")

            # Save visualization of feature matches
            try:
                transformer.save_transformation_visualization(visualization_path, use_normalized=True)
                print(f"  Saved feature match visualization: {os.path.basename(visualization_path)}")
            except Exception as viz_error:
                print(f"  Warning: Could not save visualization: {str(viz_error)}")

            # Save transformed annotations
            transformer.save_transformed_annotations(output_json_path)
            transformed_annotations.append(output_json_path)

            # Print transformation info
            transform_info = transformer.get_transformation_info()
            print(f"  Transformation status: {transform_info['status']}")

        except Exception as e:
            print(f"  Error processing {os.path.basename(original_path)}: {str(e)}")
            continue

    return transformed_annotations


def main():
    img_dir = ""
    json_path = ""
    output_dir = "output_dir"
    os.makedirs(output_dir, exist_ok=True)

    # Get list of original images
    im_list = list(absoluteFilePaths(img_dir))
    print(f"Found {len(im_list)} original images")

    # Process original images
    print("Processing original images...")
    initial_im_feats = []
    initial_metadata_list = []
    initial_anno_base_stats = []

    for im_name in im_list:
        try:
            initial_im = process_single_image(im_name, output_dir, json_path)
            initial_im_feats.append(initial_im[0])
            initial_metadata_list.append(initial_im[1])
            initial_anno_base_stats.append(initial_im[2])
        except Exception as e:
            print(f"Error processing {os.path.basename(im_name)}: {str(e)}")
            continue

    print("Done processing initial images")

    # Process corrected images
    print("Processing lens-corrected images...")
    corrected_im_feats = []
    corrected_metadata_list = []
    im_list_2 = list(absoluteFilePaths(output_dir))

    # Filter to only image files (not JSON files)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    im_list_2 = [path for path in im_list_2
                 if os.path.splitext(path.lower())[1] in image_extensions]

    print(f"Found {len(im_list_2)} corrected images")

    for im2 in im_list_2:
        try:
            lens_fixed_im = process_single_image(im2, output_dir, json_path, lfix_arg=False)
            corrected_im_feats.append(lens_fixed_im[0])
            corrected_metadata_list.append(lens_fixed_im[1])
        except Exception as e:
            print(f"Error processing corrected {os.path.basename(im2)}: {str(e)}")
            continue

    print("Done processing corrected images")

    # Find corresponding image pairs
    print("Finding corresponding image pairs...")
    image_pairs = find_corresponding_images(im_list, output_dir)
    print(f"Found {len(image_pairs)} image pairs for transformation")

    # Transform annotations for all image pairs
    if image_pairs:
        print("Transforming annotations...")
        transformed_annotation_files = transform_annotations_for_dataset(
            image_pairs, json_path, output_dir
        )
        print(f"Successfully transformed annotations for {len(transformed_annotation_files)} image pairs")

        # Print summary
        print("\nSummary:")
        print(f"  Original images processed: {len(initial_im_feats)}")
        print(f"  Corrected images processed: {len(corrected_im_feats)}")
        print(f"  Image pairs matched: {len(image_pairs)}")
        print(f"  Annotation files created: {len(transformed_annotation_files)}")
        print(f"  Transformed annotations saved in: {os.path.join(output_dir, 'transformed_annotations')}")
        print(f"  Feature match visualizations saved in: {os.path.join(output_dir, 'transformation_visualizations')}")
    else:
        print("No image pairs found for transformation")


if __name__ == "__main__":
    main()