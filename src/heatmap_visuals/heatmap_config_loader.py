import yaml
from pathlib import Path
from typing import Any

class Config:
    def __init__(self, config_path: Path):
        self._data = self._load_config(config_path)

    def _load_config(self, config_path: Path) -> dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data