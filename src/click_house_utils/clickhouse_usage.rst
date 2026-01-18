ClickHouse® is an open-source column-oriented database management system that allows generating analytical data reports in real-time.
It was investigated to solve the problem of querying a COCO segmentation JSON to get various properties.


When looking over the codebase, it was found that it could also be used to run spatial queries such as polygon overlap detection.
Spatial Querying:
https://github.com/ClickHouse/ClickHouse/pull/10678/files
ClickHouse provides native geo types (Point, Ring, Polygon, MultiPolygon) GitHub and functions for spatial operations: