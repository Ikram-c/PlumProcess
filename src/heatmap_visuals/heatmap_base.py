import json
from pathlib import Path
from typing import List, Tuple
from src.heatmap_visuals.heatmap_viz import BoundingBoxHeatmapVisualiser

class CocoLoader:
    def __init__(self, json_path: Path):
        self._json_path = json_path

    def load_boxes(self) -> List[Tuple[float, float, float, float]]:
        with open(self._json_path, "r") as f:
            data = json.load(f)
        boxes = []
        for annotation in data.get("annotations", []):
            if "bbox" in annotation:
                x, y, w, h = annotation["bbox"]
                boxes.append((x, y, x + w, y + h))
        return boxes

class HeatmapApplication:
    def __init__(self, config_path: Path, output_dir: Path):
        self._config_path = config_path
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, coco_json_path: Path) -> None:
        loader = CocoLoader(coco_json_path)
        data = loader.load_boxes()
        visualiser = BoundingBoxHeatmapVisualiser(self._config_path)
        visualiser.add_boxes(data)
        visualiser.build()
        output_html = self._output_dir / "heatmap.html"
        visualiser.save_html(output_html)
        visualiser.show()

def main():
    root_dir = Path(__file__).parent
    config_path = root_dir / "heatmap_config.yaml"
    output_dir = root_dir / "output"
    coco_path = root_dir / "instances_val2017.json"
    app = HeatmapApplication(config_path, output_dir)
    app.run(coco_path)

if __name__ == "__main__":
    main()