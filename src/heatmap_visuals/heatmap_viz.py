from pathlib import Path
from typing import List, Tuple, Dict, Any

import plotly.graph_objects as go
import numpy as np

from src.heatmap_visuals.heatmap_models import BoundingBox
from src.heatmap_visuals.heatmap_config_loader import Config
from src.heatmap_visuals.heatmap_processors import HeatmapData


class BoxShapeFactory:
    def __init__(self, config: Config):
        self._config = config

    def __call__(self, box: BoundingBox, index: int) -> Dict[str, Any]:
        vis = self._config["visualisation"]
        return {
            "type": "rect",
            "x0": box.x0,
            "y0": box.y0,
            "x1": box.x1,
            "y1": box.y1,
            "line": {"color": vis["box_line_color"], "width": vis["box_line_width"]},
            "fillcolor": "rgba(255, 255, 255, 0)",
            "opacity": vis["box_opacity"],
            "layer": "above",
            "name": f"Box {index + 1}",
        }


class MenuBuilder:
    def __init__(self, config: Config, heatmap_data: HeatmapData, box_shapes: List[dict]):
        self._config = config
        self._heatmap_data = heatmap_data
        self._box_shapes = box_shapes

    def _base_menu_style(self) -> dict:
        menu = self._config["menu"]
        return {
            "pad": {"r": menu["button_pad_r"], "t": menu["button_pad_t"]},
            "showactive": True,
            "yanchor": "top",
            "y": menu["y_position"],
            "bgcolor": menu["bg_color"],
            "bordercolor": menu["border_color"],
            "font": {"color": menu["font_color"]},
        }

    def _colorscale_menu(self) -> dict:
        buttons = [
            {"args": [{"colorscale": [cs]}], "label": cs, "method": "restyle"}
            for cs in self._config["colorscales"]
        ]
        menu = {
            "type": "dropdown",
            "direction": "down",
            "buttons": buttons,
            "x": self._config["menu"]["colorscale_x"],
            "xanchor": "left",
        }
        menu.update(self._base_menu_style())
        return menu

    def _visibility_menu(self) -> dict:
        buttons = [
            {
                "args": [{"visible": [True, True]}, {"shapes": self._box_shapes}],
                "label": "Show All",
                "method": "update",
            },
            {
                "args": [{"visible": [True, False]}, {"shapes": []}],
                "label": "Hide Points & Boxes",
                "method": "update",
            },
            {
                "args": [{"visible": [True, True]}, {"shapes": []}],
                "label": "Hide Boxes Only",
                "method": "update",
            },
            {
                "args": [{"visible": [True, False]}, {"shapes": self._box_shapes}],
                "label": "Hide Points Only",
                "method": "update",
            },
        ]
        menu = {
            "type": "buttons",
            "direction": "right",
            "buttons": buttons,
            "x": self._config["menu"]["visibility_x"],
            "xanchor": "center",
        }
        menu.update(self._base_menu_style())
        return menu

    def _smoothing_menu(self) -> dict:
        smoothing = self._config["smoothing"]
        labels = ["Low Smoothing", "Medium Smoothing", "High Smoothing"]
        sigmas = [smoothing["low"], smoothing["medium"], smoothing["high"]]
        buttons = [
            {
                "args": [{"z": [self._heatmap_data.smoothed_histogram(s).T]}],
                "label": lbl,
                "method": "restyle",
            }
            for s, lbl in zip(sigmas, labels)
        ]
        menu = {
            "type": "buttons",
            "direction": "right",
            "buttons": buttons,
            "x": self._config["menu"]["smoothing_x"],
            "xanchor": "right",
        }
        menu.update(self._base_menu_style())
        return menu

    def __iter__(self):
        yield self._colorscale_menu()
        yield self._visibility_menu()
        yield self._smoothing_menu()


class BoundingBoxHeatmapVisualiser:
    def __init__(self, config_path: Path):
        self._config = Config(config_path)
        self._boxes: List[BoundingBox] = []
        self._figure: go.Figure = go.Figure()

    def add_boxes(self, box_tuples: List[Tuple[float, float, float, float]]) -> None:
        self._boxes = [BoundingBox(x0, y0, x1, y1) for x0, y0, x1, y1 in box_tuples]

    def _create_empty_figure(self) -> None:
        canvas = self._config["canvas"]
        self._figure.add_annotation(
            text="No Data Available",
            x=canvas["width"] / 2,
            y=canvas["height"] / 2,
            showarrow=False,
            font={"size": 24},
        )

    def _create_heatmap_trace(self, heatmap_data: HeatmapData) -> go.Heatmap:
        vis = self._config["visualisation"]
        layout = self._config["layout"]
        sigma = self._config["histogram"]["default_sigma"]
        return go.Heatmap(
            z=heatmap_data.smoothed_histogram(sigma).T,
            x=heatmap_data.x_centers,
            y=heatmap_data.y_centers,
            colorscale=vis["colorscale"],
            colorbar={"title": layout["colorbar_title"]},
            name="Heatmap",
            visible=True,
            hovertemplate="X: %{x:.1f}<br>Y: %{y:.1f}<br>Frequency: %{z:.2f}<extra></extra>",
        )

    def _create_scatter_trace(self, heatmap_data: HeatmapData) -> go.Scatter:
        vis = self._config["visualisation"]
        return go.Scatter(
            x=heatmap_data.centers[:, 0],
            y=heatmap_data.centers[:, 1],
            mode="markers",
            marker={
                "color": vis["marker_color"],
                "size": vis["marker_size"],
                "opacity": vis["marker_opacity"],
                "line": {
                    "width": vis["marker_line_width"],
                    "color": vis["marker_line_color"],
                },
            },
            name="Box Centers",
            visible=True,
            hovertemplate="Center X: %{x:.1f}<br>Center Y: %{y:.1f}<extra></extra>",
        )

    def _apply_layout(self, box_shapes: List[dict], menus: List[dict]) -> None:
        canvas = self._config["canvas"]
        figure = self._config["figure"]
        layout = self._config["layout"]
        vis = self._config["visualisation"]
        self._figure.update_layout(
            shapes=box_shapes,
            title=layout["title"],
            xaxis={
                "title": layout["x_axis_title"],
                "range": [0, canvas["width"]],
                "constrain": "domain",
            },
            yaxis={
                "title": layout["y_axis_title"],
                "range": [0, canvas["height"]],
                "scaleanchor": "x",
                "scaleratio": 1,
            },
            width=figure["width"],
            height=figure["height"],
            template=vis["template"],
            showlegend=True,
            updatemenus=menus,
        )

    def build(self) -> go.Figure:
        if len(self._boxes) == 0:
            self._create_empty_figure()
            return self._figure

        heatmap_data = HeatmapData(self._boxes, self._config)
        shape_factory = BoxShapeFactory(self._config)
        box_shapes = [shape_factory(box, i) for i, box in enumerate(self._boxes)]

        self._figure.add_trace(self._create_heatmap_trace(heatmap_data))
        self._figure.add_trace(self._create_scatter_trace(heatmap_data))

        menu_builder = MenuBuilder(self._config, heatmap_data, box_shapes)
        self._apply_layout(box_shapes, list(menu_builder))

        return self._figure

    def show(self) -> None:
        self._figure.show()

    def save_html(self, path: Path) -> None:
        self._figure.write_html(str(path))

    def save_image(self, path: Path) -> None:
        figure = self._config["figure"]
        self._figure.write_image(
            str(path),
            width=figure["width"],
            height=figure["height"],
        )