from typing import Dict, List
from PIL import Image, ImageDraw
from src.mock_coco_data.mock_polygen import PolygonGenerator
import os
import json
import random
import datetime
import numpy as np








class CocoDatasetGenerator:
    def __init__(self, config: Dict):
        self.config = config
        self.coco_data = self._initialize_coco_structure()
        self.annotation_id = 1
        self.image_id = 1
        os.makedirs(self.config["image"]["output_dir"], exist_ok=True)

    def _initialize_coco_structure(self) -> Dict:
        num_supercats = self.config["categories"]["num_supercategories"]
        cat_prefix = self.config["categories"]["category_prefix"]
        supercat_prefix = self.config["categories"]["supercategory_prefix"]
        categories = [
            {
                "id": i + 1,
                "name": f"{cat_prefix}_{i+1}",
                "supercategory": f"{supercat_prefix}_{(i % num_supercats) + 1}",
            }
            for i in range(self.config["categories"]["num_categories"])
        ]
        return {
            "info": self.config.get("metadata", {}),
            "licenses": [self.config.get("metadata", {}).get("license", {})],
            "categories": categories, "images": [], "annotations": [],
        }

    def run(self):
        img_cfg = self.config["image"]
        for i in range(img_cfg["num_images"]):
            self._process_image()
            print(f"Generated image {i + 1}/{img_cfg['num_images']}...")
        self._save_json()
        print("\nDataset generation complete.")
        return self.config["output"]["coco_json_path"], img_cfg["output_dir"]

    def _process_image(self):
        img_cfg = self.config["image"]
        width = img_cfg["width"] + random.randint(-img_cfg["width_variation"], img_cfg["width_variation"])
        height = img_cfg["height"] + random.randint(-img_cfg["height_variation"], img_cfg["height_variation"])
        image = Image.new("RGB", (width, height), tuple(img_cfg["background_color"]))
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        file_name = f"image_{self.image_id:05d}.{self.config['output']['image_format']}"
        self._add_image_entry(file_name, width, height)

        poly_cfg = self.config["polygon"]
        num_polygons = random.randint(poly_cfg["min_num_polygons"], poly_cfg["max_num_polygons"])
        for _ in range(num_polygons):
            self._add_polygon(draw, width, height)

        image.paste(overlay, (0, 0), overlay)
        self._save_image(image, file_name)
        self.image_id += 1

    def _add_polygon(self, draw: ImageDraw.ImageDraw, width: int, height: int):
        poly_cfg, color_cfg = self.config["polygon"], self.config["colors"]
        min_dim = min(width, height)
        min_r, max_r = min_dim * poly_cfg["min_radius_ratio"], min_dim * poly_cfg["max_radius_ratio"]
        margin = max_r * 1.1
        center = (np.random.uniform(margin, width - margin), np.random.uniform(margin, height - margin))

        gen = PolygonGenerator(
            num_vertices=random.randint(poly_cfg["min_vertices"], poly_cfg["max_vertices"]),
            min_radius=min_r, max_radius=max_r, center=center,
            angle_variation=poly_cfg["angle_variation"],
        )
        points = gen.generate_polygon()
        flat_points = points.flatten().tolist()

        draw.polygon(
            flat_points, fill=tuple(random.choice(color_cfg["polygons"])),
            outline=tuple(color_cfg["outline"]), width=color_cfg["outline_width"]
        )
        self._add_annotation_entry(gen, flat_points)

    def _add_image_entry(self, file_name: str, width: int, height: int):
        self.coco_data["images"].append({
            "id": self.image_id, "width": width, "height": height, "file_name": file_name,
            "license": self.config["metadata"].get("license", {}).get("id", 1),
            "date_captured": str(datetime.datetime.now()),
        })

    def _add_annotation_entry(self, generator: PolygonGenerator, segmentation: List[float]):
        self.coco_data["annotations"].append({
            "id": self.annotation_id, "image_id": self.image_id,
            "category_id": random.choice(self.coco_data["categories"])["id"],
            "segmentation": [segmentation], "area": generator.get_area(),
            "bbox": generator.get_bounding_box(), "iscrowd": 0,
        })
        self.annotation_id += 1

    def _save_image(self, image: Image.Image, file_name: str):
        path = os.path.join(self.config["image"]["output_dir"], file_name)
        image.save(path, quality=self.config["output"]["image_quality"])

    def _save_json(self):
        with open(self.config["output"]["coco_json_path"], "w") as f:
            json.dump(self.coco_data, f, indent=2)
