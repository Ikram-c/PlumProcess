import json
import os
import random
import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from bloom_filter import COCOAnnotationBloomFilter
from mock_coco_data.mock_polygen import PolygonGenerator


class CocoDatasetGenerator:

    def __init__(self, config: Dict, use_bloom_filter: bool = True):
        self.config = config
        self.use_bloom_filter = use_bloom_filter
        self.coco_data = self._initialize_coco_structure()
        self.annotation_id = 1
        self.image_id = 1
        self._initialize_bloom_filter_if_enabled()
        os.makedirs(self.config["image"]["output_dir"], exist_ok=True)

    def _initialize_bloom_filter_if_enabled(self) -> None:
        if self.use_bloom_filter:
            estimated_annotations = self._estimate_total_annotations()
            self.annotation_bloom = COCOAnnotationBloomFilter(
                capacity=int(estimated_annotations * 1.3),
                false_positive_rate=0.001
            )
            return
        self.annotation_bloom = None

    def _estimate_total_annotations(self) -> int:
        img_cfg = self.config["image"]
        poly_cfg = self.config["polygon"]
        num_images = img_cfg["num_images"]
        avg_polygons = (poly_cfg["min_num_polygons"] + poly_cfg["max_num_polygons"]) / 2
        return int(num_images * avg_polygons)

    def _initialize_coco_structure(self) -> Dict:
        num_supercats = self.config["categories"]["num_supercategories"]
        cat_prefix = self.config["categories"]["category_prefix"]
        supercat_prefix = self.config["categories"]["supercategory_prefix"]
        categories = self._build_categories(num_supercats, cat_prefix, supercat_prefix)
        return {
            "info": self.config.get("metadata", {}),
            "licenses": [self.config.get("metadata", {}).get("license", {})],
            "categories": categories,
            "images": [],
            "annotations": [],
        }

    def _build_categories(
            self,
            num_supercats: int,
            cat_prefix: str,
            supercat_prefix: str
    ) -> List[Dict]:
        categories = []
        for i in range(self.config["categories"]["num_categories"]):
            category = {
                "id": i + 1,
                "name": f"{cat_prefix}_{i+1}",
                "supercategory": f"{supercat_prefix}_{(i % num_supercats) + 1}",
            }
            categories.append(category)
        return categories

    def run(self) -> Tuple[str, str]:
        img_cfg = self.config["image"]
        for i in range(img_cfg["num_images"]):
            self._process_image()
            print(f"Generated image {i + 1}/{img_cfg['num_images']}...")
        self._save_json()
        self._print_bloom_stats_if_enabled()
        print("\nDataset generation complete.")
        return self.config["output"]["coco_json_path"], img_cfg["output_dir"]

    def _print_bloom_stats_if_enabled(self) -> None:
        has_bloom = self.annotation_bloom is not None
        if has_bloom:
            stats = self.annotation_bloom.get_stats()
            print(f"\nBloom Filter Stats:")
            print(f"  Annotations tracked: {stats['items_added']}")
            print(f"  Memory used: {stats['memory_mb']:.4f} MB")
            print(f"  Utilization: {stats['utilization']:.1%}")

    def _process_image(self) -> None:
        img_cfg = self.config["image"]
        width = self._calculate_dimension(img_cfg["width"], img_cfg["width_variation"])
        height = self._calculate_dimension(img_cfg["height"], img_cfg["height_variation"])
        image = Image.new("RGB", (width, height), tuple(img_cfg["background_color"]))
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        file_name = f"image_{self.image_id:05d}.{self.config['output']['image_format']}"
        self._add_image_entry(file_name, width, height)

        self._add_polygons_to_image(draw, width, height)

        image.paste(overlay, (0, 0), overlay)
        self._save_image(image, file_name)
        self.image_id += 1

    def _calculate_dimension(self, base: int, variation: int) -> int:
        return base + random.randint(-variation, variation)

    def _add_polygons_to_image(
            self,
            draw: ImageDraw.ImageDraw,
            width: int,
            height: int
    ) -> None:
        poly_cfg = self.config["polygon"]
        num_polygons = random.randint(
            poly_cfg["min_num_polygons"],
            poly_cfg["max_num_polygons"]
        )
        for _ in range(num_polygons):
            self._add_polygon(draw, width, height)

    def _add_polygon(
            self,
            draw: ImageDraw.ImageDraw,
            width: int,
            height: int
    ) -> None:
        poly_cfg = self.config["polygon"]
        color_cfg = self.config["colors"]

        min_dim = min(width, height)
        min_r = min_dim * poly_cfg["min_radius_ratio"]
        max_r = min_dim * poly_cfg["max_radius_ratio"]
        margin = max_r * 1.1
        center = (
            np.random.uniform(margin, width - margin),
            np.random.uniform(margin, height - margin)
        )

        gen = PolygonGenerator(
            num_vertices=random.randint(
                poly_cfg["min_vertices"],
                poly_cfg["max_vertices"]
            ),
            min_radius=min_r,
            max_radius=max_r,
            center=center,
            angle_variation=poly_cfg["angle_variation"],
        )
        points = gen.generate_polygon()
        flat_points = points.flatten().tolist()

        fill_color = tuple(random.choice(color_cfg["polygons"]))
        outline_color = tuple(color_cfg["outline"])
        draw.polygon(
            flat_points,
            fill=fill_color,
            outline=outline_color,
            width=color_cfg["outline_width"]
        )
        self._add_annotation_entry(gen, flat_points)

    def _add_image_entry(self, file_name: str, width: int, height: int) -> None:
        license_id = self.config["metadata"].get("license", {}).get("id", 1)
        entry = {
            "id": self.image_id,
            "width": width,
            "height": height,
            "file_name": file_name,
            "license": license_id,
            "date_captured": str(datetime.datetime.now()),
        }
        self.coco_data["images"].append(entry)

    def _add_annotation_entry(
            self,
            generator: PolygonGenerator,
            segmentation: List[float]
    ) -> None:
        annotation_id = self._get_next_annotation_id()
        category_id = random.choice(self.coco_data["categories"])["id"]
        entry = {
            "id": annotation_id,
            "image_id": self.image_id,
            "category_id": category_id,
            "segmentation": [segmentation],
            "area": generator.get_area(),
            "bbox": generator.get_bounding_box(),
            "iscrowd": 0,
        }
        self.coco_data["annotations"].append(entry)
        self._register_annotation_id(annotation_id)
        self.annotation_id += 1

    def _get_next_annotation_id(self) -> int:
        return self.annotation_id

    def _register_annotation_id(self, ann_id: int) -> None:
        has_bloom = self.annotation_bloom is not None
        if has_bloom:
            self.annotation_bloom.add(ann_id)

    def check_annotation_id_collision(self, ann_id: int) -> bool:
        has_bloom = self.annotation_bloom is not None
        if has_bloom:
            return self.annotation_bloom.might_exist(ann_id)
        return False

    def _save_image(self, image: Image.Image, file_name: str) -> None:
        path = os.path.join(self.config["image"]["output_dir"], file_name)
        image.save(path, quality=self.config["output"]["image_quality"])

    def _save_json(self) -> None:
        with open(self.config["output"]["coco_json_path"], "w") as f:
            json.dump(self.coco_data, f, indent=2)

    def get_bloom_filter(self) -> Optional[COCOAnnotationBloomFilter]:
        return self.annotation_bloom

    def export_bloom_state(self) -> Optional[Dict[str, Any]]:
        has_bloom = self.annotation_bloom is not None
        if has_bloom:
            return self.annotation_bloom.export_state()
        return None