import json
import os
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert non-serializable objects to JSON-serializable formats.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        # For custom objects, try to convert their __dict__
        try:
            return convert_to_serializable(obj.__dict__)
        except:
            return str(obj)  # Fallback to string representation
    else:
        return obj


def save_final_results_to_json(
        output_path: str,
        initial_im_feats: List = None,
        initial_metadata_list: List = None,
        initial_anno_base_stats: List = None,
        corrected_im_feats: List = None,
        corrected_metadata_list: List = None,
        image_pairs: List = None,
        transformed_annotation_files: List = None,
        clustering_analyzer=None,
        clustering_results: Dict = None,
        img_dir: str = None,
        json_path: str = None,
        output_dir: str = None
) -> str:
    """
    Save all final results from the image processing pipeline to a JSON file.

    Args:
        output_path: Path where to save the JSON file
        initial_im_feats: List of features from original images
        initial_metadata_list: List of metadata from original images
        initial_anno_base_stats: List of annotation base statistics
        corrected_im_feats: List of features from corrected images
        corrected_metadata_list: List of metadata from corrected images
        image_pairs: List of (original, corrected) image path tuples
        transformed_annotation_files: List of paths to transformed annotation files
        clustering_analyzer: Clustering analyzer object
        clustering_results: Dictionary of clustering results
        img_dir: Original images directory path
        json_path: Path to COCO annotations JSON
        output_dir: Output directory path

    Returns:
        Path to the saved JSON file
    """

    # Initialize results dictionary
    results = {
        "processing_info": {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
            "description": "Complete image processing and clustering analysis results"
        },
        "input_parameters": {
            "images_directory": img_dir,
            "annotations_file": json_path,
            "output_directory": output_dir
        },
        "processing_summary": {
            "total_original_images": len(initial_im_feats) if initial_im_feats else 0,
            "total_corrected_images": len(corrected_im_feats) if corrected_im_feats else 0,
            "total_image_pairs": len(image_pairs) if image_pairs else 0,
            "total_transformed_annotations": len(transformed_annotation_files) if transformed_annotation_files else 0,
            "clustering_enabled": clustering_analyzer is not None
        }
    }

    # Add original images results
    if initial_im_feats or initial_metadata_list or initial_anno_base_stats:
        results["original_images"] = {
            "count": len(initial_im_feats) if initial_im_feats else 0,
            "features": convert_to_serializable(initial_im_feats) if initial_im_feats else [],
            "metadata": convert_to_serializable(initial_metadata_list) if initial_metadata_list else [],
            "annotation_stats": convert_to_serializable(initial_anno_base_stats) if initial_anno_base_stats else []
        }

    # Add corrected images results
    if corrected_im_feats or corrected_metadata_list:
        results["corrected_images"] = {
            "count": len(corrected_im_feats) if corrected_im_feats else 0,
            "features": convert_to_serializable(corrected_im_feats) if corrected_im_feats else [],
            "metadata": convert_to_serializable(corrected_metadata_list) if corrected_metadata_list else []
        }

    # Add image pairs information
    if image_pairs:
        results["image_pairs"] = {
            "count": len(image_pairs),
            "pairs": [
                {
                    "original": original_path,
                    "corrected": corrected_path,
                    "original_basename": os.path.basename(original_path),
                    "corrected_basename": os.path.basename(corrected_path)
                }
                for original_path, corrected_path in image_pairs
            ]
        }

    # Add annotation transformation results
    if transformed_annotation_files:
        results["annotation_transformations"] = {
            "count": len(transformed_annotation_files),
            "transformed_files": [
                {
                    "path": file_path,
                    "basename": os.path.basename(file_path),
                    "exists": os.path.exists(file_path)
                }
                for file_path in transformed_annotation_files
            ]
        }

    # Add clustering analysis results
    if clustering_analyzer and hasattr(clustering_analyzer, 'best_result') and clustering_analyzer.best_result:
        best_result = clustering_analyzer.best_result

        clustering_data = {
            "analysis_completed": True,
            "best_algorithm": best_result.algorithm,
            "number_of_clusters": best_result.n_clusters,
            "silhouette_score": float(best_result.silhouette_score),
            "calinski_harabasz_score": float(best_result.calinski_harabasz_score),
            "davies_bouldin_score": float(best_result.davies_bouldin_score)
        }

        # Add feature importance (top 10)
        if best_result.feature_importance:
            importance_items = sorted(
                best_result.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            clustering_data["top_separating_features"] = [
                {"feature": feature, "importance_score": float(score)}
                for feature, score in importance_items
            ]

        # Add cluster distribution
        if best_result.cluster_labels is not None:
            unique_labels, counts = np.unique(best_result.cluster_labels, return_counts=True)
            clustering_data["cluster_distribution"] = [
                {"cluster_id": int(label), "count": int(count)}
                for label, count in zip(unique_labels, counts)
            ]

        # Add autoencoder info if available
        if hasattr(clustering_analyzer, 'use_autoencoder') and clustering_analyzer.use_autoencoder:
            clustering_data["autoencoder_used"] = True
            if hasattr(clustering_analyzer, 'reconstruction_error'):
                clustering_data["reconstruction_error"] = float(clustering_analyzer.reconstruction_error)
            if hasattr(clustering_analyzer, 'X_encoded') and clustering_analyzer.X_encoded is not None:
                clustering_data["original_feature_dimensions"] = len(clustering_analyzer.feature_columns)
                clustering_data["encoded_feature_dimensions"] = clustering_analyzer.X_encoded.shape[1]
                clustering_data["dimensionality_reduction_ratio"] = float(
                    clustering_analyzer.X_encoded.shape[1] / len(clustering_analyzer.feature_columns)
                )
        else:
            clustering_data["autoencoder_used"] = False

        # Add all algorithm results comparison
        if hasattr(clustering_analyzer, 'results') and clustering_analyzer.results:
            clustering_data["all_algorithms"] = {}
            for alg_name, result in clustering_analyzer.results.items():
                clustering_data["all_algorithms"][alg_name] = {
                    "n_clusters": result.n_clusters,
                    "silhouette_score": float(result.silhouette_score),
                    "calinski_harabasz_score": float(result.calinski_harabasz_score),
                    "davies_bouldin_score": float(result.davies_bouldin_score)
                }

        results["clustering_analysis"] = clustering_data
    else:
        results["clustering_analysis"] = {
            "analysis_completed": False,
            "reason": "Clustering analysis not performed or failed"
        }

    # Add quality analysis summary if available
    quality_metrics = []
    if initial_im_feats and len(initial_im_feats) > 0:
        # Try to extract common quality metrics
        sample_features = initial_im_feats[0] if initial_im_feats[0] else {}
        if isinstance(sample_features, dict):
            # Look for common quality metrics in the features
            for key in ['exposure', 'texture', 'frequency', 'laplacian_variance']:
                if key in sample_features:
                    quality_metrics.append(key)

    if quality_metrics:
        results["quality_analysis"] = {
            "metrics_analyzed": quality_metrics,
            "total_images_analyzed": len(initial_im_feats) if initial_im_feats else 0
        }

    # Add file paths and directory structure
    results["output_structure"] = {
        "base_output_directory": output_dir,
        "subdirectories": {
            "transformed_annotations": os.path.join(output_dir, "transformed_annotations") if output_dir else None,
            "transformation_visualizations": os.path.join(output_dir,
                                                          "transformation_visualizations") if output_dir else None,
            "clustering_analysis": os.path.join(output_dir, "clustering_analysis") if output_dir else None
        }
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to JSON with proper formatting
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)  # default=str for any remaining non-serializable objects

    # Print summary
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS SAVED TO JSON")
    print(f"{'=' * 60}")
    print(f"Results file: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"Original images processed: {results['processing_summary']['total_original_images']}")
    print(f"Corrected images processed: {results['processing_summary']['total_corrected_images']}")
    print(f"Image pairs matched: {results['processing_summary']['total_image_pairs']}")
    print(f"Annotations transformed: {results['processing_summary']['total_transformed_annotations']}")
    print(
        f"Clustering analysis: {'✓ Completed' if results['clustering_analysis']['analysis_completed'] else '✗ Not performed'}")

    if results['clustering_analysis']['analysis_completed']:
        print(f"Best clustering algorithm: {results['clustering_analysis']['best_algorithm']}")
        print(f"Clusters discovered: {results['clustering_analysis']['number_of_clusters']}")
        print(f"Clustering quality: {results['clustering_analysis']['silhouette_score']:.4f}")

    return output_path


def save_results_summary_only(
        output_path: str,
        processing_summary: Dict,
        clustering_summary: Dict = None
) -> str:
    """
    Save a lightweight summary of results (without detailed data).

    Args:
        output_path: Path where to save the summary JSON
        processing_summary: Dictionary with processing counts and status
        clustering_summary: Optional clustering results summary

    Returns:
        Path to the saved summary file
    """
    summary = {
        "summary_info": {
            "timestamp": datetime.now().isoformat(),
            "type": "processing_summary"
        },
        "processing": processing_summary
    }

    if clustering_summary:
        summary["clustering"] = clustering_summary

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results summary saved to: {output_path}")
    return output_path