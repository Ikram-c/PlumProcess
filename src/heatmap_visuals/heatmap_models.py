from dataclasses import dataclass
from typing import Iterator

@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    def __iter__(self) -> Iterator[float]:
        yield self.x0
        yield self.y0
        yield self.x1
        yield self.y1