from typing import Any, Dict, List


def validate_coco_ids(coco_data: Dict[str, Any]) -> Dict[str, Any]:
    image_ids = [img['id'] for img in coco_data['images']]
    ann_ids = [ann['id'] for ann in coco_data['annotations']]
    cat_ids = [cat['id'] for cat in coco_data['categories']]

    return {
        'images_unique': len(image_ids) == len(set(image_ids)),
        'annotations_unique': len(ann_ids) == len(set(ann_ids)),
        'categories_unique': len(cat_ids) == len(set(cat_ids)),
        'num_images': len(image_ids),
        'num_annotations': len(ann_ids),
        'num_categories': len(cat_ids)
    }


def get_id_summary(coco_data: Dict[str, Any]) -> Dict[str, Any]:
    image_ids = [img['id'] for img in coco_data['images']]
    ann_ids = [ann['id'] for ann in coco_data['annotations']]
    cat_ids = [cat['id'] for cat in coco_data['categories']]

    return {
        'images': _summarize_ids(image_ids),
        'annotations': _summarize_ids(ann_ids),
        'categories': _summarize_ids(cat_ids)
    }


def _summarize_ids(id_list: List[int]) -> Dict[str, Any]:
    has_ids = len(id_list) > 0
    if has_ids:
        return {
            'count': len(id_list),
            'min_id': min(id_list),
            'max_id': max(id_list)
        }
    return {
        'count': 0,
        'min_id': None,
        'max_id': None
    }


def find_duplicate_ids(coco_data: Dict[str, Any]) -> Dict[str, List[int]]:
    image_ids = [img['id'] for img in coco_data['images']]
    ann_ids = [ann['id'] for ann in coco_data['annotations']]
    cat_ids = [cat['id'] for cat in coco_data['categories']]

    return {
        'duplicate_image_ids': _find_duplicates(image_ids),
        'duplicate_annotation_ids': _find_duplicates(ann_ids),
        'duplicate_category_ids': _find_duplicates(cat_ids)
    }


def _find_duplicates(id_list: List[int]) -> List[int]:
    seen = set()
    duplicates = set()
    for item in id_list:
        is_duplicate = item in seen
        if is_duplicate:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)


def get_category_distribution(coco_data: Dict[str, Any]) -> Dict[int, int]:
    distribution = {}
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        current_count = distribution.get(cat_id, 0)
        distribution[cat_id] = current_count + 1
    return distribution


def get_annotations_per_image(coco_data: Dict[str, Any]) -> Dict[int, int]:
    distribution = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        current_count = distribution.get(img_id, 0)
        distribution[img_id] = current_count + 1
    return distribution


def filter_annotations_by_category(
        coco_data: Dict[str, Any],
        category_ids: List[int]
) -> List[Dict[str, Any]]:
    category_set = set(category_ids)
    filtered = []
    for ann in coco_data['annotations']:
        is_match = ann['category_id'] in category_set
        if is_match:
            filtered.append(ann)
    return filtered


def filter_annotations_by_image(
        coco_data: Dict[str, Any],
        image_ids: List[int]
) -> List[Dict[str, Any]]:
    image_set = set(image_ids)
    filtered = []
    for ann in coco_data['annotations']:
        is_match = ann['image_id'] in image_set
        if is_match:
            filtered.append(ann)
    return filtered


def get_category_name_map(coco_data: Dict[str, Any]) -> Dict[int, str]:
    name_map = {}
    for cat in coco_data['categories']:
        name_map[cat['id']] = cat['name']
    return name_map


def get_image_filename_map(coco_data: Dict[str, Any]) -> Dict[int, str]:
    filename_map = {}
    for img in coco_data['images']:
        filename_map[img['id']] = img['file_name']
    return filename_map


def get_images_without_annotations(coco_data: Dict[str, Any]) -> List[int]:
    annotated_images = set()
    for ann in coco_data['annotations']:
        annotated_images.add(ann['image_id'])

    all_images = {img['id'] for img in coco_data['images']}
    unannotated = all_images - annotated_images
    return list(unannotated)


def get_annotation_area_stats(coco_data: Dict[str, Any]) -> Dict[str, float]:
    areas = [ann['area'] for ann in coco_data['annotations']]

    has_areas = len(areas) > 0
    if has_areas:
        return {
            'count': len(areas),
            'min_area': min(areas),
            'max_area': max(areas),
            'mean_area': sum(areas) / len(areas)
        }
    return {
        'count': 0,
        'min_area': None,
        'max_area': None,
        'mean_area': None
    }