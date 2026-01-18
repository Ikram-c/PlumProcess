import cv2
import numpy as np
import json
import os
from typing import Tuple, List, Dict, Any
from image_feature_matcher import ImageFeatureMatcher

class AnnotationTransformer:
    """
    Applies perspective transformation to all annotations in a COCO dataset,
    updating the 'images' and 'annotations' fields to correspond to the corrected images.
    Ensures all transformed annotations remain in image bounds.
    """
    def __init__(self, coco_json_path: str, orig_img_dir: str, corrected_img_dir: str):
        """
        Args:
            coco_json_path: Path to the COCO JSON annotation file
            orig_img_dir: Directory containing original images
            corrected_img_dir: Directory containing corrected images
        """
        self.coco_json_path = coco_json_path
        self.orig_img_dir = orig_img_dir
        self.corrected_img_dir = corrected_img_dir

        # Load COCO data
        with open(coco_json_path, "r") as f:
            self.coco_data = json.load(f)

        # Map image_id to original image filename and vice versa
        self.id_to_image = {img['id']: img for img in self.coco_data.get('images', [])}
        self.file_to_id = {img['file_name']: img['id'] for img in self.coco_data.get('images', [])}

        # Build mapping from original filename to corrected path (by name)
        self.orig_to_corrected = self._build_orig_to_corrected_map()

        # For each image, cache transformation matrix if needed
        self.transformation_cache = {}

    def _build_orig_to_corrected_map(self):
        """
        Create a mapping from original image file name to corrected image file path.
        This assumes the corrected images have the same filename in a different directory.
        """
        corrected_files = {os.path.basename(f): os.path.join(root, f)
                           for root, _, files in os.walk(self.corrected_img_dir)
                           for f in files}
        mapping = {}
        for img in self.coco_data.get('images', []):
            base_name = os.path.basename(img['file_name'])
            if base_name in corrected_files:
                mapping[base_name] = corrected_files[base_name]
            else:
                print(f"Warning: Corrected version for {base_name} not found.")
        return mapping

    def compute_transformation_matrix(self, orig_path: str, corrected_path: str, min_match_count=10, use_normalized=True):
        """
        Compute the transformation matrix between an original image and its corrected version.
        Cache the matrix for repeated use.
        """
        cache_key = (orig_path, corrected_path)
        if cache_key in self.transformation_cache:
            return self.transformation_cache[cache_key]

        orig_img = cv2.imread(orig_path)
        corrected_img = cv2.imread(corrected_path)
        if orig_img is None or corrected_img is None:
            raise FileNotFoundError(f"Could not load one of: {orig_path}, {corrected_path}")

        original_name = os.path.splitext(os.path.basename(orig_path))[0]
        corrected_name = os.path.splitext(os.path.basename(corrected_path))[0]
        original_matcher = ImageFeatureMatcher(
            orig_img, dataset_name="original", image_name=original_name
        )
        corrected_matcher = ImageFeatureMatcher(
            corrected_img, dataset_name="corrected", image_name=corrected_name
        )

        kp1, des1 = original_matcher.keypoints_and_descriptors(use_normalized=use_normalized)
        kp2, des2 = corrected_matcher.keypoints_and_descriptors(use_normalized=use_normalized)

        if des1 is None or des2 is None:
            raise ValueError("Could not extract features from one or both images")
        if len(kp1) == 0 or len(kp2) == 0:
            raise ValueError("No keypoints detected in one or both images")

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) == 0:
            raise ValueError("No matches found between images")

        distances = [m.distance for m in matches]
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        threshold = mean_distance + 0.5 * std_distance
        good_matches = [m for m in matches if m.distance <= threshold]

        if len(good_matches) < min_match_count:
            good_matches = matches[:max(min_match_count, len(matches) // 2)]

        if len(good_matches) < 4:
            raise ValueError(f"Not enough good matches found: {len(good_matches)} < 4")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            confidence=0.99,
            maxIters=2000
        )
        if matrix is None:
            raise ValueError("Could not compute transformation matrix")
        self.transformation_cache[cache_key] = matrix
        return matrix

    def clamp_point(self, point, width, height):
        x, y = point
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        return x, y

    def transform_bbox(self, matrix, bbox, width, height):
        x, y, w, h = bbox
        corners = [
            (x, y),
            (x + w, y),
            (x, y + h),
            (x + w, y + h)
        ]
        transformed_corners = [self.transform_point(matrix, corner) for corner in corners]
        # Clamp corners to image bounds
        transformed_corners = [self.clamp_point(pt, width, height) for pt in transformed_corners]
        x_coords = [pt[0] for pt in transformed_corners]
        y_coords = [pt[1] for pt in transformed_corners]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        # Filter out degenerate bboxes
        if bbox_w <= 1 or bbox_h <= 1:
            return None
        return [min_x, min_y, bbox_w, bbox_h]

    def transform_segmentation(self, matrix, segmentation, width, height):
        transformed_segmentation = []
        for polygon in segmentation:
            points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            transformed_points = [self.transform_point(matrix, pt) for pt in points]
            # Clamp points
            transformed_points = [self.clamp_point(pt, width, height) for pt in transformed_points]
            flat = []
            for pt in transformed_points:
                flat.extend([pt[0], pt[1]])
            # Only keep polygons with >= 3 points and nonzero area
            if len(flat) >= 6 and self._polygon_area(transformed_points) > 1:
                transformed_segmentation.append(flat)
        return transformed_segmentation

    def _polygon_area(self, vertices):
        # Shoelace formula
        if len(vertices) < 3:
            return 0
        x = np.array([v[0] for v in vertices])
        y = np.array([v[1] for v in vertices])
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def transform_keypoints(self, matrix, keypoints, width, height):
        if len(keypoints) % 3 != 0:
            raise ValueError("Keypoints must be in groups of 3 (x,y,visibility)")
        transformed_keypoints = []
        for i in range(0, len(keypoints), 3):
            x, y, visibility = keypoints[i], keypoints[i + 1], keypoints[i + 2]
            if visibility > 0:
                tx, ty = self.transform_point(matrix, (x, y))
                tx, ty = self.clamp_point((tx, ty), width, height)
                transformed_keypoints.extend([tx, ty, visibility])
            else:
                transformed_keypoints.extend([x, y, visibility])
        return transformed_keypoints

    def transform_point(self, matrix: np.ndarray, point: Tuple[float, float]) -> Tuple[float, float]:
        src_point = np.array([[point[0], point[1]]], dtype=np.float32).reshape(-1, 1, 2)
        dst_point = cv2.perspectiveTransform(src_point, matrix)
        return (float(dst_point[0][0][0]), float(dst_point[0][0][1]))

    def transform_dataset_annotations(self, output_json_path: str) -> None:
        """
        Transform the entire dataset: update 'images' and 'annotations' for the corrected images.
        Output is a single COCO-format JSON with corrected image paths/sizes and transformed annotations.
        Ensures that all transformed annotations are in bounds.
        """
        coco_out = json.loads(json.dumps(self.coco_data))  # Deep copy

        # 1. Update images list
        new_images = []
        corrected_name_to_id = {}
        imgid_to_size = {}
        for img in coco_out['images']:
            base_name = os.path.basename(img['file_name'])
            if base_name in self.orig_to_corrected:
                corrected_path = self.orig_to_corrected[base_name]
                corrected_img = cv2.imread(corrected_path)
                if corrected_img is None:
                    print(f"Warning: Could not load corrected image for {base_name}. Keeping original size.")
                    height, width = img['height'], img['width']
                else:
                    height, width = corrected_img.shape[:2]
                new_img_entry = dict(img)  # shallow copy of original fields
                new_img_entry['file_name'] = os.path.basename(corrected_path)
                new_img_entry['width'] = width
                new_img_entry['height'] = height
                new_images.append(new_img_entry)
                corrected_name_to_id[base_name] = img['id']
                imgid_to_size[img['id']] = (width, height)
            else:
                print(f"Warning: Skipping image {base_name}, no corrected version found.")

        coco_out['images'] = new_images

        # 2. Transform all annotations, ensuring they stay in bounds
        new_annotations = []
        for ann in coco_out['annotations']:
            img_id = ann['image_id']
            if img_id not in self.id_to_image or img_id not in imgid_to_size:
                print(f"Warning: Annotation {ann['id']} refers to missing image id {img_id}. Skipping.")
                continue
            orig_img_entry = self.id_to_image[img_id]
            base_name = os.path.basename(orig_img_entry['file_name'])
            if base_name not in self.orig_to_corrected:
                continue  # skip if no corrected image

            orig_path = os.path.join(self.orig_img_dir, base_name)
            corrected_path = self.orig_to_corrected[base_name]
            width, height = imgid_to_size[img_id]
            try:
                matrix = self.compute_transformation_matrix(orig_path, corrected_path)
            except Exception as e:
                print(f"Warning: Failed to compute transformation for {base_name}: {e}. Skipping annotation {ann['id']}.")
                continue

            keep = True

            if 'bbox' in ann:
                new_bbox = self.transform_bbox(matrix, ann['bbox'], width, height)
                if new_bbox is not None:
                    ann['bbox'] = new_bbox
                else:
                    keep = False  # skip degenerate bbox

            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                new_seg = self.transform_segmentation(matrix, ann['segmentation'], width, height)
                if new_seg:
                    ann['segmentation'] = new_seg
                else:
                    keep = False  # skip if segmentation is invalid or empty

            if 'keypoints' in ann:
                ann['keypoints'] = self.transform_keypoints(matrix, ann['keypoints'], width, height)

            if keep:
                new_annotations.append(ann)

        coco_out['annotations'] = new_annotations

        # 3. Save the new JSON
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(coco_out, f, indent=2)
        print(f"Full transformed COCO JSON saved to {output_json_path}")

# Example usage:
# transformer = AnnotationTransformer(
#     coco_json_path="/path/to/your/original.json",
#     orig_img_dir="/path/to/original/images",
#     corrected_img_dir="/path/to/corrected/images"
# )
# transformer.transform_dataset_annotations("/path/to/save/transformed_dataset.json")
