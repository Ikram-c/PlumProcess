import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from pycocotools.coco import COCO

logger = logging.getLogger(__name__)


class CocoResolutionSplitter:

    def __init__(self, settings: dict[str, Any]):
        self.input_path = Path(settings["input_json_path"])
        self.output_dir = Path(settings["output_dir"])
        self.json_indent = settings.get("json_indent", 4)
        self.filename_sep = settings.get("filename_separator", "_")
        self.res_sep = settings.get("resolution_separator", "x")

        self._validate_input_path()

        logger.info("Loading annotation file: %s", self.input_path)
        self.coco = COCO(str(self.input_path))

        self.base_info = self.coco.dataset.get("info", {})
        self.base_licenses = self.coco.dataset.get("licenses", [])
        self.base_categories = self.coco.dataset.get("categories", [])

        self.name_part = self.input_path.stem
        self.ext_part = self.input_path.suffix

    def _validate_input_path(self) -> None:
        if self.input_path.exists():
            return
        raise FileNotFoundError(f"Input file not found: {self.input_path}")

    def _group_images_by_resolution(self) -> defaultdict[tuple[int, int], list[int]]:
        logger.info("Grouping images by resolution...")
        res_to_img_ids: defaultdict[tuple[int, int], list[int]] = defaultdict(list)

        for img_id in self.coco.getImgIds():
            img = self.coco.loadImgs([img_id])[0]
            resolution = (img["width"], img["height"])
            res_to_img_ids[resolution].append(img_id)

        logger.info("Found %d unique resolutions.", len(res_to_img_ids))
        return res_to_img_ids

    def _build_output_path(self, width: int, height: int) -> Path:
        filename = (
            f"{self.name_part}{self.filename_sep}"
            f"{width}{self.res_sep}{height}{self.ext_part}"
        )
        return self.output_dir / filename

    def _create_subset(self, img_ids: list[int]) -> dict[str, Any]:
        ann_ids = self.coco.getAnnIds(imgIds=img_ids)
        return {
            "info": self.base_info,
            "licenses": self.base_licenses,
            "categories": self.base_categories,
            "images": self.coco.loadImgs(img_ids),
            "annotations": self.coco.loadAnns(ann_ids),
        }

    def _save_subset(self, subset_data: dict[str, Any], output_path: Path) -> None:
        logger.info("Saving to %s...", output_path)
        with open(output_path, "w") as f:
            json.dump(subset_data, f, indent=self.json_indent)

    def _process_resolution(
        self,
        resolution: tuple[int, int],
        img_ids: list[int],
        output_path: Path,
    ) -> None:
        width, height = resolution
        logger.info(
            "Processing resolution: %d%s%d (%d images)",
            width,
            self.res_sep,
            height,
            len(img_ids),
        )
        subset_data = self._create_subset(img_ids)
        self._save_subset(subset_data, output_path)

    def run(self) -> dict[tuple[int, int], Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", self.output_dir)

        resolutions_map = self._group_images_by_resolution()

        resolution_file_map = {
            res: self._build_output_path(*res) for res in resolutions_map
        }

        for resolution, img_ids in resolutions_map.items():
            output_path = resolution_file_map[resolution]
            self._process_resolution(resolution, img_ids, output_path)

        logger.info("Splitting complete.")
        return resolution_file_map

    @classmethod
    def from_config(cls, config_path: Path | str) -> "CocoResolutionSplitter":
        logger.info("Loading configuration from: %s", config_path)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        settings = config.get("coco_splitter_settings")

        if settings is None:
            raise KeyError(
                f"Key 'coco_splitter_settings' not found in {config_path}."
            )

        return cls(settings)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Split a COCO JSON by resolution using a config file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the main config.yaml file.",
    )
    args = parser.parse_args()

    try:
        splitter = CocoResolutionSplitter.from_config(args.config_path)
        splitter.run()
    except Exception as e:
        logger.exception("An error occurred: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()