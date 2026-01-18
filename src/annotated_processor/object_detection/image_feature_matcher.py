import cv2
from csbdeep.utils import normalize
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


class ImageFeatureMatcher:
    def __init__(self, img, matcher="akaze", dataset_name="unknown_dataset", image_name="unknown_image"):
        self.img = img
        self.matcher = matcher
        self.dataset_name = dataset_name
        self.image_name = image_name

        if not os.path.exists("tmp"):
            os.makedirs("tmp")

    def keypoints_and_descriptors(self, use_normalized=True):
        img_gray = self._norm_img() if use_normalized else self._prepare_gray_image(self.img)
        akaze = cv2.AKAZE_create()
        return akaze.detectAndCompute(img_gray, None)

    def _norm_img(self):
        img_gray = self._prepare_gray_image(self.img)
        norm_img = normalize(img_gray, 1, 99.8, axis=(0, 1))
        return (norm_img * 255).astype(np.uint8)

    def _prepare_gray_image(self, img):
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def keypoints_to_dataframe(self, use_normalized=True):
        keypoints, descriptors = self.keypoints_and_descriptors(use_normalized=use_normalized)
        img_source = "Normalized" if use_normalized else "Original"

        df = pd.DataFrame([{
            'dataset_name': self.dataset_name,
            'image_name': self.image_name,
            'img_source': img_source,
            'keypoint_id': i,
            'x_coord': kp.pt[0],
            'y_coord': kp.pt[1],
            'size': kp.size,
            'angle': kp.angle,
            'response': kp.response,
            'octave': kp.octave,
            'class_id': kp.class_id
        } for i, kp in enumerate(keypoints)])

        if descriptors is not None:
            for i in range(descriptors.shape[1]):
                df[f'descriptor_{i}'] = descriptors[:, i]

        return df

    def save_keypoints_csv(self, output_dir="tmp", use_normalized=True):
        df = self.keypoints_to_dataframe(use_normalized=use_normalized)
        os.makedirs(output_dir, exist_ok=True)

        norm_suffix = "normalized" if use_normalized else "original"
        filename = f"{self.image_name}_{norm_suffix}_keypoints.csv"
        output_path = os.path.join(output_dir, filename)

        df.to_csv(output_path, index=False)
        return output_path

    def plot_features(self, use_normalized=True, output_path=None):
        norm_suffix = "normalized" if use_normalized else "original"
        if output_path is None:
            output_path = os.path.join("tmp", f"{self.image_name}_{norm_suffix}_features.png")

        keypoints, _ = self.keypoints_and_descriptors(use_normalized=use_normalized)
        display_img = self._prepare_display_img(self.img, use_normalized)

        img_with_keypoints = cv2.drawKeypoints(
            display_img, keypoints, None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        img_with_keypoints_rgb = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 10))
        plt.imshow(img_with_keypoints_rgb)
        plt.title(f"{norm_suffix.capitalize()} {self.image_name} with {len(keypoints)} {self.matcher.upper()} features")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    def _prepare_display_img(self, img, use_normalized):
        img_gray = self._norm_img() if use_normalized else self._prepare_gray_image(img)
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    def analyze_and_save_all_data(self, output_dir="tmp"):
        os.makedirs(output_dir, exist_ok=True)
        results = {
            'normalized_plot': self.plot_features(True, os.path.join(output_dir, f"{self.image_name}_normalized_features.png")),
            'original_plot': self.plot_features(False, os.path.join(output_dir, f"{self.image_name}_original_features.png")),
            'normalized_csv': self.save_keypoints_csv(output_dir, True),
            'original_csv': self.save_keypoints_csv(output_dir, False)
        }

        norm_df = self.keypoints_to_dataframe(True)
        orig_df = self.keypoints_to_dataframe(False)
        combined_df = pd.concat([norm_df, orig_df], ignore_index=True)

        combined_filename = f"{self.image_name}_all_keypoints.csv"
        combined_path = os.path.join(output_dir, combined_filename)
        combined_df.to_csv(combined_path, index=False)
        results['combined_csv'] = combined_path

        return results