import networkx as nx
import itertools
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Event:
    x: float
    rectangle: tuple
    is_start: bool
    rect_idx: int
    
    @property
    def sort_key(self):
        return (self.x, not self.is_start, self.rectangle[1])


class IntervalUnionQuery:
    def __init__(self, intervals: list, y_coords: list):
        assert intervals

        self.N = 1 << (len(intervals) - 1).bit_length()  # Next power of 2
        self.c = [0] * (2 * self.N)
        self.s = [0] * (2 * self.N)
        self.w = [0] * (2 * self.N)
        self.overlaps = []
        self.y_coords = y_coords
        self.active_intervals = {}
        self.sweep_events = []
        self.active_rectangles = {}

        # Build weights array
        for i, val in enumerate(intervals):
            self.w[self.N + i] = val
        for p in range(self.N - 1, 0, -1):
            self.w[p] = self.w[2 * p] + self.w[2 * p + 1]

    def union_size(self) -> float:
        return self.s[1]

    def modify_interval(self, i: int, k: int, offset: int, x_coord: float, rect_idx: int):
        for y in range(i, k):
            self.active_intervals.setdefault(y, 0)
            self.active_rectangles.setdefault(y, set())

            old_count = self.active_intervals[y]
            self.active_intervals[y] += offset
            new_count = self.active_intervals[y]

            # Update active rectangles
            (self.active_rectangles[y].add if offset == 1 else 
             self.active_rectangles[y].discard)(rect_idx)

            if old_count != new_count:
                self.sweep_events.append((
                    x_coord, y, old_count, new_count, 
                    set(self.active_rectangles[y])
                ))

        self._change(1, 0, self.N, i, k, offset)

    def find_overlaps(self):
        self.sweep_events.sort()
        active_regions = {}

        for x, y, old_count, new_count, rect_set in self.sweep_events:
            # Handle coverage ending
            for count in range(old_count, new_count, -1) if new_count < old_count else []:
                if count >= 2 and y in active_regions.get(count, {}):
                    start_x, start_rects = active_regions[count][y]
                    if x > start_x:
                        y_low = self.y_coords[y]
                        y_high = self.y_coords[y + 1] if y + 1 < len(self.y_coords) else y_low
                        self.overlaps.append((start_x, x, y_low, y_high, count, start_rects))
                    del active_regions[count][y]

            # Handle coverage starting
            for count in range(old_count + 1, new_count + 1) if new_count > old_count else []:
                if count >= 2:
                    active_regions.setdefault(count, {})[y] = (x, rect_set)

        # Filter and sort
        self.overlaps = sorted(
            [o for o in self.overlaps if o[1] > o[0]],
            key=lambda o: (-o[4], o[0], o[2])
        )

    def _change(self, p: int, start: int, span: int, i: int, k: int, offset: int):
        if start + span <= i or k <= start:
            return
        if i <= start and start + span <= k:
            self.c[p] += offset
        else:
            half = span // 2
            self._change(2 * p, start, half, i, k, offset)
            self._change(2 * p + 1, start + half, half, i, k, offset)

        self.s[p] = (
            self.w[p] if self.c[p] 
            else (0 if p >= self.N else self.s[2 * p] + self.s[2 * p + 1])
        )


def _find_transitivity(overlapping_indices: list) -> list:
    """Find transitive closure of overlapping rectangle groups."""
    g = nx.Graph()
    g.add_edges_from(
        edge for node_set in overlapping_indices 
        for edge in itertools.combinations(node_set, 2)
    )
    return [tuple(comp) for comp in nx.connected_components(g)]


def _normalize_rectangle(rect: tuple) -> tuple:
    """Normalize rectangle to ensure x0 < x1 and y0 < y1."""
    x0, y0, x1, y1 = rect
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _build_events(rectangles: list) -> list:
    """Build and sort sweep line events from rectangles."""
    events = [
        Event(rect[i], rect, is_start, idx)
        for idx, rect in enumerate(rectangles)
        for i, is_start in [(0, True), (2, False)]
    ]
    return sorted(events, key=lambda e: e.sort_key)


def _check_y_containment(inner_y: tuple, outer_y: tuple) -> bool:
    """Check if inner y-bounds are contained within outer y-bounds."""
    return outer_y[0] <= inner_y[0] and inner_y[1] <= outer_y[1]


def find_overlapping(rectangles: list) -> tuple:
    """
    Find overlapping and nested rectangles using sweep line algorithm.
    
    Returns:
        tuple: (overlap_groups, nested_indices)
            - overlap_groups: list of tuples of transitively overlapping rectangle indices
            - nested_indices: list of indices of rectangles fully contained in others
    """
    if not rectangles:
        return None, []

    normalized = [_normalize_rectangle(r) for r in rectangles]
    events = _build_events(normalized)

    # Build y-coordinate mapping
    y_coords = sorted({y for rect in normalized for y in (rect[1], rect[3])})
    y_intervals = [y_coords[i + 1] - y_coords[i] for i in range(len(y_coords) - 1)]
    y_map = {val: idx for idx, val in enumerate(y_coords)}

    interval_query = IntervalUnionQuery(y_intervals, y_coords)

    # State tracking: {rect_idx: (y0_idx, y1_idx)}
    active_rects = {}
    nesting_candidates = {}  # {inner_idx: set(potential_outer_idx)}
    nested_pairs = []

    for event in events:
        y_bounds = (y_map[event.rectangle[1]], y_map[event.rectangle[3]])
        
        if event.is_start:
            # Find active rectangles that contain this one in y-dimension
            potential_outers = {
                idx for idx, other_y in active_rects.items()
                if _check_y_containment(y_bounds, other_y)
            }
            if potential_outers:
                nesting_candidates[event.rect_idx] = potential_outers

            active_rects[event.rect_idx] = y_bounds
            interval_query.modify_interval(*y_bounds, +1, event.x, event.rect_idx)
        else:
            # Confirm nesting for candidates where outer is still active
            if event.rect_idx in nesting_candidates:
                confirmed = [
                    (event.rect_idx, outer_idx) 
                    for outer_idx in nesting_candidates[event.rect_idx]
                    if outer_idx in active_rects
                ]
                nested_pairs.extend(confirmed)
                del nesting_candidates[event.rect_idx]

            del active_rects[event.rect_idx]
            interval_query.modify_interval(*y_bounds, -1, event.x, event.rect_idx)

    interval_query.find_overlaps()

    # Extract overlap groups
    overlapping_indices = [o[5] for o in interval_query.overlaps]
    overlap_groups = _find_transitivity(overlapping_indices) if overlapping_indices else None

    # Extract nested indices
    nested_indices = sorted({pair[0] for pair in nested_pairs})

    return overlap_groups, nested_indices


def find_invalid_inds(areas: np.ndarray, overlap_groups: list) -> list:
    """Find invalid indices from overlapping groups, keeping largest area."""
    sorted_inds = np.argsort(areas)[::-1]
    
    def get_invalid_from_group(group):
        mask = np.isin(sorted_inds, list(group))
        max_idx = sorted_inds[np.argmax(mask)]
        return [idx for idx in group if idx != max_idx]
    
    invalid = itertools.chain.from_iterable(map(get_invalid_from_group, overlap_groups))
    return list(set(invalid))


def find_all_invalid_inds(rectangles: list, areas: np.ndarray) -> tuple:
    """
    Find all invalid indices including both overlapping and nested rectangles.
    
    Returns:
        tuple: (overlap_invalid_indices, nested_indices, all_invalid_indices)
    """
    overlap_groups, nested_indices = find_overlapping(rectangles)
    
    overlap_invalid = find_invalid_inds(areas, overlap_groups) if overlap_groups else []
    all_invalid = sorted(set(overlap_invalid) | set(nested_indices))
    
    return overlap_invalid, nested_indices, all_invalid


# =============================================================================
# Test Data Generation Functions
# =============================================================================

def generate_nested_rectangles(
    n_groups: int = 5,
    max_depth: int = 3,
    bounds: tuple = (0, 100, 0, 100),
    min_size: float = 5.0,
    margin_range: tuple = (2.0, 10.0),
    seed: Optional[int] = None
) -> tuple:
    """
    Generate test rectangles with guaranteed nesting relationships.
    
    Args:
        n_groups: Number of independent nesting groups to generate
        max_depth: Maximum nesting depth per group (1 = single rect, 2 = one nested, etc.)
        bounds: (x_min, x_max, y_min, y_max) bounding box for outermost rectangles
        min_size: Minimum width/height for innermost rectangles
        margin_range: (min, max) margin between nested rectangle edges
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (rectangles, expected_nested_indices, nesting_groups)
            - rectangles: list of (x0, y0, x1, y1) tuples
            - expected_nested_indices: list of indices that should be detected as nested
            - nesting_groups: list of lists showing nesting hierarchy per group
    """
    rng = np.random.default_rng(seed)
    x_min, x_max, y_min, y_max = bounds
    
    rectangles = []
    nesting_groups = []
    expected_nested = []
    
    for _ in range(n_groups):
        depth = rng.integers(1, max_depth + 1)
        group_indices = []
        
        # Generate outermost rectangle
        width = rng.uniform(min_size * depth * 2, (x_max - x_min) / 2)
        height = rng.uniform(min_size * depth * 2, (y_max - y_min) / 2)
        x0 = rng.uniform(x_min, x_max - width)
        y0 = rng.uniform(y_min, y_max - height)
        
        current_rect = (x0, y0, x0 + width, y0 + height)
        rect_idx = len(rectangles)
        rectangles.append(current_rect)
        group_indices.append(rect_idx)
        
        # Generate nested rectangles inward
        for d in range(1, depth):
            margin = rng.uniform(*margin_range)
            x0_new = current_rect[0] + margin
            y0_new = current_rect[1] + margin
            x1_new = current_rect[2] - margin
            y1_new = current_rect[3] - margin
            
            # Ensure valid rectangle
            if x1_new - x0_new < min_size or y1_new - y0_new < min_size:
                break
            
            current_rect = (x0_new, y0_new, x1_new, y1_new)
            rect_idx = len(rectangles)
            rectangles.append(current_rect)
            group_indices.append(rect_idx)
            expected_nested.append(rect_idx)
        
        nesting_groups.append(group_indices)
    
    return rectangles, sorted(expected_nested), nesting_groups


def generate_mixed_test_rectangles(
    n_nested_groups: int = 3,
    n_overlapping: int = 4,
    n_isolated: int = 3,
    bounds: tuple = (0, 100, 0, 100),
    seed: Optional[int] = None
) -> dict:
    """
    Generate a mixed test set with nested, overlapping, and isolated rectangles.
    
    Args:
        n_nested_groups: Number of nesting groups
        n_overlapping: Number of overlapping (but not nested) rectangles
        n_isolated: Number of non-overlapping isolated rectangles
        bounds: (x_min, x_max, y_min, y_max) bounding box
        seed: Random seed for reproducibility
    
    Returns:
        dict with keys:
            - rectangles: list of all rectangles
            - nested_indices: expected nested rectangle indices
            - nesting_groups: hierarchy of nested groups
            - overlapping_indices: indices of overlapping rectangles
            - isolated_indices: indices of isolated rectangles
            - areas: numpy array of rectangle areas
    """
    rng = np.random.default_rng(seed)
    x_min, x_max, y_min, y_max = bounds
    
    # Generate nested rectangles
    nested_rects, expected_nested, nesting_groups = generate_nested_rectangles(
        n_groups=n_nested_groups,
        max_depth=3,
        bounds=(x_min, x_max * 0.4, y_min, y_max),
        seed=seed
    )
    
    rectangles = list(nested_rects)
    
    # Generate overlapping rectangles in a different region
    overlap_start_idx = len(rectangles)
    overlap_center_x = x_max * 0.7
    overlap_center_y = y_max * 0.5
    
    for i in range(n_overlapping):
        size = rng.uniform(10, 25)
        offset_x = rng.uniform(-15, 15)
        offset_y = rng.uniform(-15, 15)
        
        x0 = overlap_center_x + offset_x
        y0 = overlap_center_y + offset_y
        rectangles.append((x0, y0, x0 + size, y0 + size))
    
    overlapping_indices = list(range(overlap_start_idx, len(rectangles)))
    
    # Generate isolated rectangles
    isolated_start_idx = len(rectangles)
    
    for i in range(n_isolated):
        size_x = rng.uniform(5, 15)
        size_y = rng.uniform(5, 15)
        # Place in bottom region to avoid others
        x0 = rng.uniform(x_min + i * 20, x_min + i * 20 + 15)
        y0 = rng.uniform(y_max * 0.85, y_max - size_y)
        rectangles.append((x0, y0, x0 + size_x, y0 + size_y))
    
    isolated_indices = list(range(isolated_start_idx, len(rectangles)))
    
    # Calculate areas
    areas = np.array([
        abs((r[2] - r[0]) * (r[3] - r[1])) for r in rectangles
    ])
    
    return {
        'rectangles': rectangles,
        'nested_indices': expected_nested,
        'nesting_groups': nesting_groups,
        'overlapping_indices': overlapping_indices,
        'isolated_indices': isolated_indices,
        'areas': areas
    }



if __name__ == '__main__':
    
    rect_list = [(2, 2, 17, 2), (2, 2, 17, 4), (2, 2, 17, 6), (2, 2, 17, 8), (2, 2, 17, 10), (2, 2, 17, 12), 
                 (2, 2, 17, 14), (2, 2, 17, 16), (2, 2, 17, 18), (2, 2, 17, 20), 
                 (2, 2, 17, 22), (2, 2, 17, 24), (2, 2, 17, 26), (2, 2, 17, 28)]
    
    
    
    overlap_groups, nested_indices = find_overlapping(rect_list)
    print(overlap_groups)
    print(nested_indices)
    print(max(rect_list))
    print(min(rect_list))