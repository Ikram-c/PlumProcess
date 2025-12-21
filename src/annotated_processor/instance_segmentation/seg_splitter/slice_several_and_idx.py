import networkx as nx
import itertools
import numpy as np
from collections import defaultdict
import json
import tempfile
import os
from pycocotools.coco import COCO


class IntervalUnionQuery:

    def __init__(self, L, y_coords):
        assert L != []
        self.N = 1
        while self.N < len(L):
            self.N *= 2
        self.c = [0] * (2 * self.N)
        self.s = [0] * (2 * self.N)
        self.w = [0] * (2 * self.N)
        self.overlaps = []
        self.y_coords = y_coords
        self.sweep_events = []
        self.active_rectangles_a = {}
        self.active_rectangles_b = {}

        for i, val in enumerate(L):
            self.w[self.N + i] = val
        for p in range(self.N - 1, 0, -1):
            self.w[p] = self.w[2 * p] + self.w[2 * p + 1]

    def modify_interval(self, i, k, offset, x_coord, rect_idx, is_from_list_a):
        for y in range(i, k):
            self._ensure_y_initialized(y)
            was_overlapping = bool(self.active_rectangles_a[y]) and bool(self.active_rectangles_b[y])
            old_sets = (set(self.active_rectangles_a[y]), set(self.active_rectangles_b[y]))
            self._update_target_set(y, is_from_list_a, offset, rect_idx)
            is_overlapping = bool(self.active_rectangles_a[y]) and bool(self.active_rectangles_b[y])
            new_sets = (set(self.active_rectangles_a[y]), set(self.active_rectangles_b[y]))
            self._maybe_append_sweep_event(was_overlapping, is_overlapping, old_sets, new_sets, x_coord, y)
        self._change(1, 0, self.N, i, k, offset)

    def _ensure_y_initialized(self, y):
        if y in self.active_rectangles_a:
            return
        self.active_rectangles_a[y] = set()
        self.active_rectangles_b[y] = set()

    def _update_target_set(self, y, is_from_list_a, offset, rect_idx):
        target_set = self.active_rectangles_a[y] if is_from_list_a else self.active_rectangles_b[y]
        if offset == 1:
            target_set.add(rect_idx)
            return
        target_set.discard(rect_idx)

    def _maybe_append_sweep_event(self, was_overlapping, is_overlapping, old_sets, new_sets, x_coord, y):
        changed_state = was_overlapping != is_overlapping
        sets_differ = is_overlapping and old_sets != new_sets
        if changed_state or sets_differ:
            self.sweep_events.append(
                (x_coord, y, set(self.active_rectangles_a[y]), set(self.active_rectangles_b[y])))

    def find_overlaps_between_lists(self):
        self.sweep_events.sort()
        active_regions = {}
        for x, y, rects_a, rects_b in self.sweep_events:
            self._process_existing_region(active_regions, x, y)
            self._maybe_start_new_region(active_regions, y, x, rects_a, rects_b)

    def _process_existing_region(self, active_regions, x, y):
        if y not in active_regions:
            return
        start_x, start_rects_a, start_rects_b = active_regions.pop(y)
        if x <= start_x:
            return
        y_low = self.y_coords[y]
        y_high = self.y_coords[y + 1]
        self.overlaps.append((start_x, x, y_low, y_high, start_rects_a, start_rects_b))

    def _maybe_start_new_region(self, active_regions, y, x, rects_a, rects_b):
        if rects_a and rects_b:
            active_regions[y] = (x, rects_a, rects_b)

    def _change(self, p, start, span, i, k, offset):
        if start + span <= i:
            return
        if k <= start:
            return
        if i <= start and start + span <= k:
            self.c[p] += offset
        else:
            mid = span // 2
            self._change(2 * p, start, mid, i, k, offset)
            self._change(2 * p + 1, start + mid, mid, i, k, offset)
        self._update_s(p)

    def _update_s(self, p):
        if self.c[p] != 0:
            self.s[p] = self.w[p]
            return
        if p >= self.N:
            self.s[p] = 0
            return
        self.s[p] = self.s[2 * p] + self.s[2 * p + 1]


class Event:

    def __init__(self, x, rectangle, is_start, rect_idx, is_from_list_a):
        self.x = x
        self.rectangle = rectangle
        self.is_start = is_start
        self.rect_idx = rect_idx
        self.is_from_list_a = is_from_list_a


def find_overlapping_between_lists(rectangles_a, rectangles_b):
    if _is_empty(rectangles_a) or _is_empty(rectangles_b):
        return None, None, []

    normalized_a = [_normalize_rect(r) for r in rectangles_a]
    normalized_b = [_normalize_rect(r) for r in rectangles_b]

    events = _create_events(normalized_a, normalized_b)
    if _is_empty(events):
        return None, None, []

    events.sort(key=lambda e: (e.x, e.is_start == False))

    y_coords = sorted(list(set(y for r in normalized_a + normalized_b for y in [r[1], r[3]])))
    if len(y_coords) < 2:
        return None, None, []

    y_intervals = [y_coords[i + 1] - y_coords[i] for i in range(len(y_coords) - 1)]
    y_mapping = {val: idx for idx, val in enumerate(y_coords)}

    query = IntervalUnionQuery(y_intervals, y_coords)
    _process_events(events, y_mapping, query)
    query.find_overlaps_between_lists()

    return _extract_results(query.overlaps)


def _is_empty(collection):
    return len(collection) == 0


def _normalize_rect(r):
    return (min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3]))


def _create_events(normalized_a, normalized_b):
    events = []
    for i, r in enumerate(normalized_a):
        if r[0] < r[2]:
            events.append(Event(r[0], r, True, i, True))
            events.append(Event(r[2], r, False, i, True))
    for i, r in enumerate(normalized_b):
        if r[0] < r[2]:
            events.append(Event(r[0], r, True, i, False))
            events.append(Event(r[2], r, False, i, False))
    return events


def _process_events(events, y_mapping, query):
    for event in events:
        y0 = y_mapping[event.rectangle[1]]
        y1 = y_mapping[event.rectangle[3]]
        if y0 == y1:
            continue
        offset = 1 if event.is_start else -1
        query.modify_interval(y0, y1, offset, event.x, event.rect_idx, event.is_from_list_a)


def _extract_results(overlaps):
    indices_a = set()
    indices_b = set()
    pairs = set()
    for _, _, _, _, rects_a, rects_b in overlaps:
        indices_a.update(rects_a)
        indices_b.update(rects_b)
        for a_idx in rects_a:
            for b_idx in rects_b:
                pairs.add((a_idx, b_idx))
    if _is_empty(indices_a):
        return None, None, []
    return sorted(list(indices_a)), sorted(list(indices_b)), sorted(list(pairs))


def is_contained(rect_b, rect_a):
    left_ok = rect_b[0] >= rect_a[0]
    right_ok = rect_b[2] <= rect_a[2]
    top_ok = rect_b[1] >= rect_a[1]
    bottom_ok = rect_b[3] <= rect_a[3]
    return left_ok and right_ok and top_ok and bottom_ok


def _has_overlap(rect_b, rect_a):
    h_overlap = rect_b[0] < rect_a[2] and rect_b[2] > rect_a[0]
    v_overlap = rect_b[1] < rect_a[3] and rect_b[3] > rect_a[1]
    return h_overlap and v_overlap


def classify_intersection(rect_b, rect_a):
    if is_contained(rect_b, rect_a):
        return _make_intersection_result('contained', [], [], [])

    if _has_overlap(rect_b, rect_a) == False:
        return _make_intersection_result('none', [], [], [])

    crosses_left = rect_b[0] < rect_a[0] and rect_b[2] > rect_a[0]
    crosses_right = rect_b[0] < rect_a[2] and rect_b[2] > rect_a[2]
    crosses_top = rect_b[1] < rect_a[1] and rect_b[3] > rect_a[1]
    crosses_bottom = rect_b[1] < rect_a[3] and rect_b[3] > rect_a[3]

    boundaries_crossed = []
    vertical_boundaries = []
    horizontal_boundaries = []

    _add_boundary_if_crossed(crosses_left, 'left', boundaries_crossed, vertical_boundaries)
    _add_boundary_if_crossed(crosses_right, 'right', boundaries_crossed, vertical_boundaries)
    _add_boundary_if_crossed(crosses_top, 'top', boundaries_crossed, horizontal_boundaries)
    _add_boundary_if_crossed(crosses_bottom, 'bottom', boundaries_crossed, horizontal_boundaries)

    crosses_vertical = crosses_left or crosses_right
    crosses_horizontal = crosses_top or crosses_bottom
    intersection_type = _determine_intersection_type(crosses_vertical, crosses_horizontal)

    return _make_intersection_result(intersection_type, boundaries_crossed, horizontal_boundaries, vertical_boundaries)


def _add_boundary_if_crossed(crossed, name, boundaries_crossed, directional_list):
    if crossed:
        boundaries_crossed.append(name)
        directional_list.append(name)


def _determine_intersection_type(crosses_vertical, crosses_horizontal):
    if crosses_vertical and crosses_horizontal:
        return 'both'
    if crosses_vertical:
        return 'vertical'
    if crosses_horizontal:
        return 'horizontal'
    return 'none'


def _make_intersection_result(itype, boundaries, horizontal, vertical):
    return {
        'type': itype,
        'boundaries_crossed': boundaries,
        'horizontal_boundaries': horizontal,
        'vertical_boundaries': vertical
    }


def analyze_b_intersections(rectangles_a, rectangles_b, b_idx_to_ann_id=None):
    _, _, overlap_pairs = find_overlapping_between_lists(rectangles_a, rectangles_b)
    if _is_empty(overlap_pairs):
        return {}

    normalized_a = [_normalize_rect(r) for r in rectangles_a]
    normalized_b = [_normalize_rect(r) for r in rectangles_b]

    grouped_by_b = defaultdict(list)
    for a_idx, b_idx in overlap_pairs:
        grouped_by_b[b_idx].append(a_idx)

    result = {}
    for b_idx, a_indices in grouped_by_b.items():
        rect_b = normalized_b[b_idx]
        intersections = _get_intersections_for_b(rect_b, a_indices, normalized_a)
        _add_to_result_if_valid(result, b_idx, intersections, b_idx_to_ann_id)

    return result


def _get_intersections_for_b(rect_b, a_indices, normalized_a):
    intersections = []
    for a_idx in a_indices:
        rect_a = normalized_a[a_idx]
        intersection_info = classify_intersection(rect_b, rect_a)
        _maybe_add_intersection(intersections, a_idx, intersection_info)
    return intersections


def _maybe_add_intersection(intersections, a_idx, intersection_info):
    excluded_types = {'contained', 'none'}
    if intersection_info['type'] in excluded_types:
        return
    intersections.append({
        'rect_a_index': a_idx,
        'intersection_type': intersection_info['type'],
        'boundaries_crossed': intersection_info['boundaries_crossed'],
        'horizontal_boundaries': intersection_info['horizontal_boundaries'],
        'vertical_boundaries': intersection_info['vertical_boundaries']
    })


def _add_to_result_if_valid(result, b_idx, intersections, b_idx_to_ann_id):
    if _is_empty(intersections):
        return
    b_key = b_idx_to_ann_id[b_idx] if b_idx_to_ann_id else b_idx
    result[b_key] = {
        'intersecting_rectangles_a': intersections,
        'intersection_count': len(intersections)
    }


def load_coco_data_for_processing(coco_api, img_ids):
    rectangles_b = []
    b_idx_to_ann_id = {}
    current_idx = 0

    for img_id in img_ids:
        annIds = coco_api.getAnnIds(imgIds=img_id)
        anns = coco_api.loadAnns(annIds)
        for ann in anns:
            bbox = ann['bbox']
            r = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            rectangles_b.append(r)
            b_idx_to_ann_id[current_idx] = ann['id']
            current_idx += 1

    return rectangles_b, b_idx_to_ann_id


def _create_dummy_coco_data():
    return {
        "images": [{"id": 1, "width": 200, "height": 200}],
        "annotations": [
            {"id": 101, "image_id": 1, "bbox": [60, 60, 20, 20], "category_id": 1, "iscrowd": 0, "area": 400},
            {"id": 102, "image_id": 1, "bbox": [30, 60, 30, 20], "category_id": 1, "iscrowd": 0, "area": 600},
            {"id": 103, "image_id": 1, "bbox": [60, 30, 20, 30], "category_id": 1, "iscrowd": 0, "area": 600},
            {"id": 104, "image_id": 1, "bbox": [30, 30, 30, 30], "category_id": 1, "iscrowd": 0, "area": 900},
            {"id": 105, "image_id": 1, "bbox": [150, 150, 20, 20], "category_id": 1, "iscrowd": 0, "area": 400},
        ],
        "categories": [{"id": 1, "name": "test"}]
    }


def _run_example():
    dummy_coco_data = _create_dummy_coco_data()

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(dummy_coco_data, f)
        tmp_ann_file = f.name

    coco_api = COCO(tmp_ann_file)
    img_ids_to_process = [1]
    rectangles_b, b_idx_to_ann_id = load_coco_data_for_processing(coco_api, img_ids_to_process)

    print(f"Loaded {len(rectangles_b)} annotations from COCO.")
    print(f"Rectangles B: {rectangles_b}\n")

    rectangles_a = [[50, 50, 100, 100]]
    print(f"Rectangles A: {rectangles_a}\n")

    intersection_results = analyze_b_intersections(rectangles_a, rectangles_b, b_idx_to_ann_id)

    print("--- Intersection Analysis (B's Perspective) ---")
    print(json.dumps(intersection_results, indent=2))

    os.remove(tmp_ann_file)


if __name__ == "__main__":
    _run_example()