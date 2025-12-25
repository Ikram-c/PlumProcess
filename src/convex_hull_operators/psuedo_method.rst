graham scan for the individual polygons to get self intersections
gjk_nesterov to find overlapping polygon pairs so we don't need to process each one to find intersections
-----------------
use a mod of our sweep tree using the x0,y0,xn-1,yn-1,xn,yn method 
if we use the bounding box intersection analyser, we can identify boom here is where the overlap happens i.e. the region 
this is where we need to use a connected components labelling model like pycle to figure out which annotation gets priority
-----------------------------------------------------
In the connected components labelling, we take the overlapping polygon pairs and a buffer around them
then run it. 
Get the outline contour of object 1 and object 2
turn them into coordinate points using our efd model
replace the coordinates within the overlap region for object 1 with the new coordinates
replace the coordinates within the overlap region for object 2 with the new coordinates
update the bounding box region

----------------------------------------------------
since the labelling part is for the segmentation models -> for the geometry section we should leave it as a mark in the dictionary

BBoxes need to be updated if convex hull expands the polygon -> add a check just in case as an optional optimizer

the dictionary to be edited is the original COCO JSON -> add more sub keys -> easiest implementation which allows cross compatibility

extra keys:
"convex_hull_coordinates" -> list[[float,float],[float,float]] -> [[x0,y0],[x1,y1]]
"self_intersection" -> bool -> True or False
"overlap" -> bool -> True or False
"overlap_pair" -> list[float,float,float] -> overlapping pairs where the floats are the annotation ids of the other overlapping objects
"overlap_region" -> list[[list],[list]] -> within each nested list is x0,y0,x1,y1 of the overlapping bbox region
                                        -> the list should be sorted by the overlap_pair order 
                                        ->i.e. if the first float in overlap_pair is anno_id 2, overlap region[0] should be of that area



sticking to bbox for initial detection allows us to ignore several overlaps at different points between two objects
    -> computationally more efficient 
    -> the reason we don't make the adjustments to the polygon "off the bat" is that we could have a bad human annotator, 
    -> this is why a connected component analyser is honestly the best trade off and it'll give us better polygons anyway
    


---------------
Nesterov accelerated distance measurement between non intersecting polygons is needed to get the relative distribution of segments in an image 
    -> stat analysis
    -> necessary for the train test splitter process

    