import json
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.coco_bloom_filter import COCOAnnotationBloomFilter


class COCODatasetExtender:

    def __init__(
            self,
            existing_coco_path: str,
            bloom_capacity_multiplier: float = 1.3
    ):
        self.coco_path = existing_coco_path
        self.multiplier = bloom_capacity_multiplier
        self.coco_data = self._load_coco_data(existing_coco_path)
        self._extract_existing_ids()
        self._calculate_max_ids()
        self._initialize_bloom_filter()

    def _load_coco_data(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return json.load(f)

    def _extract_existing_ids(self) -> None:
        self.existing_image_ids = {img['id'] for img in self.coco_data['images']}
        self.existing_annotation_ids = {
            ann['id'] for ann in self.coco_data['annotations']
        }
        self.existing_category_ids = {
            cat['id'] for cat in self.coco_data['categories']
        }

    def _calculate_max_ids(self) -> None:
        self.max_image_id = self._get_max_id(self.existing_image_ids)
        self.max_annotation_id = self._get_max_id(self.existing_annotation_ids)
        self.max_category_id = self._get_max_id(self.existing_category_ids)

    def _get_max_id(self, id_set: Set[int]) -> int:
        has_ids = len(id_set) > 0
        if has_ids:
            return max(id_set)
        return 0

    def _initialize_bloom_filter(self) -> None:
        expected_additions = int(len(self.existing_annotation_ids) * 0.5)
        bloom_capacity = int(
            (len(self.existing_annotation_ids) + expected_additions) * self.multiplier
        )
        self.annotation_bloom = COCOAnnotationBloomFilter(
            capacity=bloom_capacity,
            false_positive_rate=0.001
        )
        self._populate_bloom_filter()

    def _populate_bloom_filter(self) -> None:
        for ann_id in self.existing_annotation_ids:
            self.annotation_bloom.add(ann_id)

    def check_annotation_id(self, ann_id: int) -> Tuple[bool, Optional[int]]:
        is_definitely_new = self.annotation_bloom.definitely_new(ann_id)
        if is_definitely_new:
            return True, None
        return self._handle_potential_collision(ann_id)

    def _handle_potential_collision(self, ann_id: int) -> Tuple[bool, Optional[int]]:
        is_real_collision = ann_id in self.existing_annotation_ids
        if is_real_collision:
            suggested = self.max_annotation_id + 1
            self.max_annotation_id += 1
            return False, suggested
        return True, None

    def add_annotations(
            self,
            new_annotations: List[Dict[str, Any]],
            auto_remap_ids: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        processed = []
        stats = self._create_annotation_stats(len(new_annotations))

        for ann in new_annotations:
            result = self._process_single_annotation(ann, auto_remap_ids, stats)
            processed.append(result)

        return processed, stats

    def _create_annotation_stats(self, total: int) -> Dict[str, Any]:
        return {
            'total': total,
            'collisions_detected': 0,
            'false_positives': 0,
            'ids_remapped': 0,
            'remapping': {}
        }

    def _process_single_annotation(
            self,
            ann: Dict[str, Any],
            auto_remap_ids: bool,
            stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        original_id = ann['id']
        is_unique, suggested_id = self.check_annotation_id(original_id)

        if is_unique:
            return self._add_unique_annotation(ann, original_id)
        return self._handle_collision(ann, original_id, suggested_id, auto_remap_ids, stats)

    def _add_unique_annotation(
            self,
            ann: Dict[str, Any],
            original_id: int
    ) -> Dict[str, Any]:
        self.annotation_bloom.add(original_id)
        self.existing_annotation_ids.add(original_id)
        return ann

    def _handle_collision(
            self,
            ann: Dict[str, Any],
            original_id: int,
            suggested_id: Optional[int],
            auto_remap_ids: bool,
            stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        stats['collisions_detected'] += 1

        can_remap = auto_remap_ids and suggested_id is not None
        if can_remap:
            return self._remap_annotation(ann, original_id, suggested_id, stats)
        raise ValueError(f"Annotation ID collision: {original_id} already exists")

    def _remap_annotation(
            self,
            ann: Dict[str, Any],
            original_id: int,
            suggested_id: int,
            stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        new_ann = ann.copy()
        new_ann['id'] = suggested_id
        self.annotation_bloom.add(suggested_id)
        self.existing_annotation_ids.add(suggested_id)
        stats['ids_remapped'] += 1
        stats['remapping'][original_id] = suggested_id
        warnings.warn(f"Annotation ID collision: {original_id} -> {suggested_id}")
        return new_ann

    def merge_and_save(
            self,
            new_coco_path: str,
            output_path: str,
            auto_remap_ids: bool = True
    ) -> Dict[str, Any]:
        new_coco = self._load_coco_data(new_coco_path)
        processed_annotations, ann_stats = self.add_annotations(
            new_coco['annotations'],
            auto_remap_ids=auto_remap_ids
        )

        merged_data = self.coco_data.copy()
        merged_data['annotations'].extend(processed_annotations)

        image_id_map = self._merge_images(new_coco['images'], merged_data)
        self._update_annotation_image_refs(processed_annotations, image_id_map)

        category_id_map = self._merge_categories(new_coco['categories'], merged_data)
        self._update_annotation_category_refs(processed_annotations, category_id_map)

        self._save_merged_data(merged_data, output_path)

        return self._compile_merge_stats(
            ann_stats, new_coco, image_id_map, category_id_map, output_path
        )

    def _merge_images(
            self,
            new_images: List[Dict[str, Any]],
            merged_data: Dict[str, Any]
    ) -> Dict[int, int]:
        image_id_map = {}
        for img in new_images:
            is_collision = img['id'] in self.existing_image_ids
            if is_collision:
                new_id = self.max_image_id + 1
                image_id_map[img['id']] = new_id
                img['id'] = new_id
                self.max_image_id += 1
            self.existing_image_ids.add(img['id'])
            merged_data['images'].append(img)
        return image_id_map

    def _update_annotation_image_refs(
            self,
            annotations: List[Dict[str, Any]],
            image_id_map: Dict[int, int]
    ) -> None:
        has_remapping = len(image_id_map) > 0
        if has_remapping:
            for ann in annotations:
                needs_update = ann['image_id'] in image_id_map
                if needs_update:
                    ann['image_id'] = image_id_map[ann['image_id']]

    def _merge_categories(
            self,
            new_categories: List[Dict[str, Any]],
            merged_data: Dict[str, Any]
    ) -> Dict[int, int]:
        category_name_map = {cat['name']: cat['id']
                             for cat in merged_data['categories']}
        category_id_map = {}

        for cat in new_categories:
            exists = cat['name'] in category_name_map
            if exists:
                category_id_map[cat['id']] = category_name_map[cat['name']]
                continue
            new_id = self.max_category_id + 1
            category_id_map[cat['id']] = new_id
            cat['id'] = new_id
            merged_data['categories'].append(cat)
            category_name_map[cat['name']] = new_id
            self.max_category_id += 1

        return category_id_map

    def _update_annotation_category_refs(
            self,
            annotations: List[Dict[str, Any]],
            category_id_map: Dict[int, int]
    ) -> None:
        for ann in annotations:
            needs_update = ann['category_id'] in category_id_map
            if needs_update:
                ann['category_id'] = category_id_map[ann['category_id']]

    def _save_merged_data(
            self,
            merged_data: Dict[str, Any],
            output_path: str
    ) -> None:
        with open(output_path, 'w') as f:
            json.dump(merged_data, f, indent=2)

    def _compile_merge_stats(
            self,
            ann_stats: Dict[str, Any],
            new_coco: Dict[str, Any],
            image_id_map: Dict[int, int],
            category_id_map: Dict[int, int],
            output_path: str
    ) -> Dict[str, Any]:
        return {
            'annotations': ann_stats,
            'images_added': len(new_coco['images']),
            'images_remapped': len(image_id_map),
            'categories_added': self._count_new_categories(new_coco['categories']),
            'bloom_filter_stats': self.annotation_bloom.get_stats(),
            'output_path': output_path
        }

    def _count_new_categories(self, categories: List[Dict[str, Any]]) -> int:
        existing_names = {cat['name'] for cat in self.coco_data['categories']}
        count = 0
        for cat in categories:
            is_existing = cat['name'] in existing_names
            if is_existing == False:
                count += 1
        return count