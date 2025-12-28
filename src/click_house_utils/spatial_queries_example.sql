-- Convert flat COCO coordinates to Point array for geo functions
SELECT 
    annotation_id,
    arrayMap(
        polygon -> arrayMap(
            i -> (polygon[i], polygon[i + 1]),
            arrayFilter(i -> i % 2 = 1, range(1, length(polygon) + 1))
        ),
        segmentation
    ) AS polygon_points
FROM annotations
WHERE length(segmentation) > 0 AND length(segmentation[1]) > 0;

-- Check if a point is inside a polygon (polygon must be constant)
SELECT pointInPolygon(
    (125.0, 240.0),  -- Query point
    [(100.0, 200.0), (150.0, 200.0), (150.0, 280.0), (100.0, 280.0)]  -- Polygon
) AS is_inside;

-- Calculate polygon area (Cartesian coordinates for image pixels)
SELECT polygonAreaCartesian([[(0, 0), (100, 0), (100, 100), (0, 100)]]) AS area;

-- Bounding box intersection query (highly optimized with minmax index)
SELECT * FROM annotations
WHERE bbox.x + bbox.width >= 100   -- query_min_x
  AND bbox.x <= 500                -- query_max_x  
  AND bbox.y + bbox.height >= 100  -- query_min_y
  AND bbox.y <= 400;               -- query_max_y