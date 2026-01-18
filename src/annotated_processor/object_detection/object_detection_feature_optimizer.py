"""
object_detection_feature_optimizer.py

Analyzes annotation clustering results to recommend anchor boxes, augmentation policy,
class balancing, and post-processing strategies for object detection tasks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from collections import defaultdict

def recommend_anchor_boxes(cluster_centers: np.ndarray, feature_names: List[str], n=6):
    """
    Recommend anchor boxes (width, height) based on clustering centers.
    Returns top-N most representative box shapes.
    """
    # Assume feature_names contains "bbox_width" and "bbox_height" (or area/aspect)
    width_idx = feature_names.index("bbox_width") if "bbox_width" in feature_names else None
    height_idx = feature_names.index("bbox_height") if "bbox_height" in feature_names else None

    if width_idx is None or height_idx is None:
        raise ValueError("bbox_width and bbox_height features required for anchor recommendation.")

    anchors = cluster_centers[:, [width_idx, height_idx]]
    # Sort by area, pick top-N unique
    areas = anchors[:, 0] * anchors[:, 1]
    order = np.argsort(-areas)
    anchors_sorted = anchors[order]
    # Remove near-duplicates
    unique_anchors = []
    for box in anchors_sorted:
        if not any(np.allclose(box, ua, rtol=0.1) for ua in unique_anchors):
            unique_anchors.append(box)
        if len(unique_anchors) >= n:
            break
    return np.array(unique_anchors)

def suggest_augmentations(feature_importance: Dict[str, float]):
    """
    Suggests augmentation strategies based on feature importance.
    """
    recs = []
    # High importance of spatial features: recommend spatial transforms
    spatial_feats = ["bbox_x_center", "bbox_y_center", "horizontal_position", "vertical_position"]
    area_feats = ["bbox_area", "bbox_width", "bbox_height", "bbox_aspect_ratio"]
    if any(f in feature_importance and feature_importance[f] > 0.2 for f in spatial_feats):
        recs.append("Apply random translations, crops, and spatial flips.")
    if any(f in feature_importance and feature_importance[f] > 0.2 for f in area_feats):
        recs.append("Apply random scaling and aspect ratio jitter.")
    if "bbox_elongation" in feature_importance and feature_importance["bbox_elongation"] > 0.2:
        recs.append("Consider random rotations and stretching.")
    if not recs:
        recs.append("Standard color and geometric augmentations recommended.")
    return recs

def recommend_class_weights(cluster_labels: np.ndarray, df: pd.DataFrame, min_importance=0.1, feature_importance: Dict[str, float]=None):
    """
    Suggests class balancing/weighting strategies for 'hard-to-separate' clusters or classes.
    """
    # For each cluster, check if feature importance is low; these are hard to separate.
    if feature_importance is not None:
        mean_importance = np.mean(list(feature_importance.values()))
        hard_features = [k for k,v in feature_importance.items() if v < mean_importance/2]
    else:
        hard_features = []

    df = df.copy()
    df['cluster'] = cluster_labels
    hard_clusters = []
    for c in np.unique(cluster_labels):
        cdf = df[df['cluster']==c]
        if cdf.shape[0] < 10:  # skip tiny clusters
            continue
        # If cluster members span many classes or have large variance in key features, it's 'hard'
        cat_counts = cdf['category_id'].value_counts(normalize=True)
        if cat_counts.max() < 0.7:  # No dominant class in cluster
            hard_clusters.append(c)

    advice = []
    if hard_clusters:
        advice.append(
            f"Increase loss weight for objects in clusters: {hard_clusters} (these are hard to separate)."
        )
        if not df['category_id'].isnull().all():
            for c in hard_clusters:
                cats = df[df['cluster']==c]['category_id'].unique()
                advice.append(f"Clusters {c} mixes categories {cats}. Consider more data or cleaning for these.")

    if hard_features:
        advice.append(f"Objects differing mostly by {hard_features} are difficult to separate; "
                      "consider collecting more varied examples or domain-specific augmentations.")
    if not advice:
        advice.append("No particularly hard-to-separate clusters detected. Use standard class balancing.")

    return advice

def print_cluster_summary(result: Any, feature_df: pd.DataFrame):
    """
    Pretty-prints the clustering result and optimization recommendations.
    """
    print("\n--- CLUSTERING-BASED OBJECT DETECTION FEATURE OPTIMIZER ---")
    print(f"Best clustering: {result.algorithm} with {result.n_clusters} clusters")
    print("Top features for cluster separation:")
    for feat, imp in sorted(result.feature_importance.items(), key=lambda x: -x[1])[:8]:
        print(f"  {feat:25s} : {imp:.3f}")
    print("\nRecommended anchor boxes (w, h):")
    anchors = recommend_anchor_boxes(result.cluster_centers, result.feature_names)
    for a in anchors:
        print(f"  [{a[0]:.1f}, {a[1]:.1f}]")
    print("\nSuggested augmentation strategies:")
    for rec in suggest_augmentations(result.feature_importance):
        print(" -", rec)
    print("\nClass balancing / weighting advice:")
    for adv in recommend_class_weights(result.cluster_labels, feature_df, feature_importance=result.feature_importance):
        print(" -", adv)
    print("------------------------------------------------------------")

# 
# from annotation_clustering import ClusteringResult
# from object_detection_feature_optimizer import print_cluster_summary
#
# # Assume `best_result` is from EnhancedClusteringAnalyzer.best_result
# # and `feature_df` is the DataFrame used for clustering
# print_cluster_summary(best_result, feature_df)
