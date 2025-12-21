# Python script based on Uber's Research into self intersection
# Solved section for this script: 
# -> checking if a polygon is valid
# ------------------------------------------------- #
# Inputs: a list of x y 


# get two cartesian coordinates
# determine whether clockwise or anti clockwise
# take the third point
# see if it is clockwise or anto clockwise, if direction changes -> scrap the bih

# TODO: test and reformat
# TODO: loading the segmentations efficiently
# TODO: check if pylops can let us do this with several coords at once
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from sortedcontainers import SortedList


CONFIG_PATH = Path(__file__).parent / "config.yaml"

DEFAULT_EPSILON = 1e-9
DEFAULT_MIN_VERTICES = 4


def load_config(path: Path = CONFIG_PATH) -> Dict:
    if path.exists() == False:
        return {}

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_config_value(config: Dict, *keys, default=None):
    result = config
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result


class EventType(Enum):
    LEFT = 0
    RIGHT = 1


@dataclass(order=True)
class Event:
    x: float
    event_type: EventType = field(compare=False)
    segment_idx: int = field(compare=False)
    point: Tuple[float, float] = field(compare=False)


@dataclass
class Segment:
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    idx: int

    def __post_init__(self):
        needs_swap = self.p1[0] > self.p2[0] or (
            self.p1[0] == self.p2[0] and self.p1[1] > self.p2[1]
        )
        if needs_swap:
            self.p1, self.p2 = self.p2, self.p1

    def y_at_x(self, x: float) -> float:
        if self.p1[0] == self.p2[0]:
            return self.p1[1]
        t = (x - self.p1[0]) / (self.p2[0] - self.p1[0])
        return self.p1[1] + t * (self.p2[1] - self.p1[1])


@dataclass
class IntersectionResult:
    is_intersecting: bool
    point: Optional[Tuple[float, float]] = None


@dataclass
class ProcessingState:
    status: "SweepLineStatus"
    segments: List[Segment]
    n_segments: int
    processed: set
    result: Optional[Tuple[float, float]] = None


class SweepLineStatus:

    def __init__(self, eps: float):
        self.eps = eps
        self.current_x = 0.0
        self._segments: Dict[int, Segment] = {}
        self._active: SortedList = SortedList(
            key=lambda idx: self._segments[idx].y_at_x(self.current_x)
        )

    def set_x(self, x: float) -> None:
        self.current_x = x

    def insert(self, segment: Segment) -> None:
        self._segments[segment.idx] = segment
        self._active.add(segment.idx)

    def remove(self, segment_idx: int) -> None:
        if segment_idx in self._active:
            self._active.remove(segment_idx)

    def get_neighbors(self, segment_idx: int) -> Tuple[Optional[int], Optional[int]]:
        if segment_idx not in self._active:
            return None, None

        pos = self._active.index(segment_idx)
        below = self._active[pos - 1] if pos > 0 else None
        above = self._active[pos + 1] if pos < len(self._active) - 1 else None
        return below, above


class PolygonIntersectionChecker:

    def __init__(
        self,
        epsilon: Optional[float] = None,
        min_vertices: Optional[int] = None,
        config_path: Optional[Path] = None
    ):
        config = load_config(config_path or CONFIG_PATH)

        self.eps = epsilon or get_config_value(
            config, "intersection", "epsilon", default=DEFAULT_EPSILON
        )
        self.min_vertices = min_vertices or get_config_value(
            config, "intersection", "min_vertices", default=DEFAULT_MIN_VERTICES
        )

    def _ccw(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        c: Tuple[float, float]
    ) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def _segments_intersect(
        self,
        s1: Segment,
        s2: Segment
    ) -> Optional[Tuple[float, float]]:
        d1 = self._ccw(s2.p1, s2.p2, s1.p1)
        d2 = self._ccw(s2.p1, s2.p2, s1.p2)
        d3 = self._ccw(s1.p1, s1.p2, s2.p1)
        d4 = self._ccw(s1.p1, s1.p2, s2.p2)

        cross1 = (d1 > self.eps and d2 < -self.eps) or (d1 < -self.eps and d2 > self.eps)
        cross2 = (d3 > self.eps and d4 < -self.eps) or (d3 < -self.eps and d4 > self.eps)

        if cross1 and cross2:
            t = d1 / (d1 - d2)
            ix = s1.p1[0] + t * (s1.p2[0] - s1.p1[0])
            iy = s1.p1[1] + t * (s1.p2[1] - s1.p1[1])
            return (ix, iy)

        return None

    def _build_segments(self, points: np.ndarray) -> List[Segment]:
        n = len(points)
        indices = np.arange(n)
        next_indices = (indices + 1) % n

        segments = [
            Segment(tuple(points[i]), tuple(points[next_indices[i]]), i)
            for i in indices
        ]
        return segments

    def _build_events(self, segments: List[Segment]) -> SortedList:
        events = SortedList()

        left_events = [
            Event(seg.p1[0], EventType.LEFT, seg.idx, seg.p1)
            for seg in segments
        ]
        right_events = [
            Event(seg.p2[0], EventType.RIGHT, seg.idx, seg.p2)
            for seg in segments
        ]

        [events.add(e) for e in left_events]
        [events.add(e) for e in right_events]

        return events

    def _are_adjacent(self, idx1: int, idx2: int, n_segments: int) -> bool:
        diff = abs(idx1 - idx2)
        return diff == 1 or diff == n_segments - 1

    def _check_neighbor_intersection(
        self,
        seg: Segment,
        neighbor_idx: Optional[int],
        segments: List[Segment],
        n_segments: int,
        current_x: float
    ) -> Optional[Tuple[float, float]]:
        if neighbor_idx is None:
            return None

        if self._are_adjacent(seg.idx, neighbor_idx, n_segments):
            return None

        intersection = self._segments_intersect(seg, segments[neighbor_idx])

        if intersection is None:
            return None

        if intersection[0] < current_x - self.eps:
            return None

        return intersection

    def _process_left_event(
        self,
        event: Event,
        state: ProcessingState
    ) -> Optional[Tuple[float, float]]:
        seg = state.segments[event.segment_idx]
        state.status.insert(seg)

        below, above = state.status.get_neighbors(event.segment_idx)

        below_intersection = self._check_neighbor_intersection(
            seg, below, state.segments, state.n_segments, event.x
        )
        if below_intersection:
            return below_intersection

        above_intersection = self._check_neighbor_intersection(
            seg, above, state.segments, state.n_segments, event.x
        )
        return above_intersection

    def _process_right_event(
        self,
        event: Event,
        state: ProcessingState
    ) -> Optional[Tuple[float, float]]:
        below, above = state.status.get_neighbors(event.segment_idx)
        state.status.remove(event.segment_idx)

        if below is None or above is None:
            return None

        if self._are_adjacent(below, above, state.n_segments):
            return None

        intersection = self._segments_intersect(
            state.segments[below],
            state.segments[above]
        )

        if intersection is None:
            return None

        pair = tuple(sorted([below, above]))

        if pair in state.processed:
            return None

        state.processed.add(pair)

        if intersection[0] < event.x - self.eps:
            return None

        return intersection

    def _process_event(
        self,
        state: ProcessingState,
        event: Event
    ) -> ProcessingState:
        if state.result is not None:
            return state

        state.status.set_x(event.x)

        handlers = {
            EventType.LEFT: lambda: self._process_left_event(event, state),
            EventType.RIGHT: lambda: self._process_right_event(event, state)
        }

        handler = handlers.get(event.event_type)
        result = handler() if handler else None

        state.result = result
        return state

    def _process_events(
        self,
        events: SortedList,
        segments: List[Segment],
        status: SweepLineStatus,
        n_segments: int
    ) -> Optional[Tuple[float, float]]:
        initial_state = ProcessingState(
            status=status,
            segments=segments,
            n_segments=n_segments,
            processed=set(),
            result=None
        )

        final_state = reduce(self._process_event, events, initial_state)
        return final_state.result

    def check(
        self,
        points: Union[List[float], np.ndarray]
    ) -> IntersectionResult:
        points_arr = np.asarray(points, dtype=np.float64)

        if points_arr.ndim == 1:
            if points_arr.size % 2 != 0:
                raise ValueError("Flat list must have even number of elements.")
            points_arr = points_arr.reshape(-1, 2)

        if len(points_arr) < self.min_vertices:
            return IntersectionResult(False, None)

        segments = self._build_segments(points_arr)
        events = self._build_events(segments)
        status = SweepLineStatus(self.eps)
        n_segments = len(segments)

        intersection_point = self._process_events(events, segments, status, n_segments)
        is_intersecting = intersection_point is not None

        return IntersectionResult(is_intersecting, intersection_point)


def is_self_intersecting(
    points: Union[List[float], np.ndarray],
    epsilon: Optional[float] = None,
    min_vertices: Optional[int] = None
) -> bool:
    checker = PolygonIntersectionChecker(epsilon=epsilon, min_vertices=min_vertices)
    result = checker.check(points)
    return result.is_intersecting


def get_intersection_point(
    points: Union[List[float], np.ndarray],
    epsilon: Optional[float] = None,
    min_vertices: Optional[int] = None
) -> Optional[Tuple[float, float]]:
    checker = PolygonIntersectionChecker(epsilon=epsilon, min_vertices=min_vertices)
    result = checker.check(points)
    return result.point


if __name__ == "__main__":
    simple_polygon = [10.0, 10.0, 90.0, 10.0, 90.0, 90.0, 10.0, 90.0]
    bowtie_polygon = [10.0, 10.0, 90.0, 90.0, 90.0, 10.0, 10.0, 90.0]
    complex_polygon = [
        0.0, 0.0, 50.0, 0.0, 50.0, 30.0, 20.0, 30.0,
        20.0, 20.0, 40.0, 20.0, 40.0, 40.0, 0.0, 40.0
    ]

    checker = PolygonIntersectionChecker()

    print("Simple polygon (square):")
    result = checker.check(simple_polygon)
    print(f"  Self-intersecting: {result.is_intersecting}")
    print(f"  Intersection point: {result.point}")

    print("\nBowtie polygon:")
    result = checker.check(bowtie_polygon)
    print(f"  Self-intersecting: {result.is_intersecting}")
    print(f"  Intersection point: {result.point}")

    print("\nComplex polygon:")
    result = checker.check(complex_polygon)
    print(f"  Self-intersecting: {result.is_intersecting}")
    print(f"  Intersection point: {result.point}")

    print("\n--- Convenience functions ---")
    print(f"is_self_intersecting(simple): {is_self_intersecting(simple_polygon)}")
    print(f"is_self_intersecting(bowtie): {is_self_intersecting(bowtie_polygon)}")
    print(f"get_intersection_point(bowtie): {get_intersection_point(bowtie_polygon)}")

    print("\n--- Custom epsilon override ---")
    result = PolygonIntersectionChecker(epsilon=1e-12).check(bowtie_polygon)
    print(f"  Bowtie with eps=1e-12: {result.is_intersecting}")