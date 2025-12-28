-- Parse raw COCO annotation JSON inline
SELECT
    JSONExtractUInt(raw_json, 'id') AS annotation_id,
    JSONExtractUInt(raw_json, 'image_id') AS image_id,
    JSONExtract(raw_json, 'bbox', 'Array(Float64)') AS bbox,
    JSONExtract(raw_json, 'segmentation', 'Array(Array(Float64))') AS polygons,
    JSONExtractFloat(raw_json, 'area') AS area,
    JSONExtractBool(raw_json, 'iscrowd') AS is_crowd
FROM raw_coco_landing;

-- Extract individual bbox components
SELECT
    JSONExtractFloat(annotation_json, 'bbox', 1) AS x,
    JSONExtractFloat(annotation_json, 'bbox', 2) AS y,
    JSONExtractFloat(annotation_json, 'bbox', 3) AS width,
    JSONExtractFloat(annotation_json, 'bbox', 4) AS height
FROM annotations_raw;

-- Check array length for segmentation
SELECT JSONLength(raw_json, 'segmentation') AS num_polygons
FROM raw_coco;