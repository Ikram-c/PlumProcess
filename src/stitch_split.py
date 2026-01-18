# This script will help speed the processing up by checking the resolutions and assessing whether each individual image needs to be assessed before
# stitching or splitting
# 
# check the lookup table for the preset the user defines in the config -> for dummy case -> YOLO -> 640x640

# define the original dataset path
# two flows -> unannotated -> annotated
# if unannotated -> process and then decide
# if annotated -> do a max width and a max height from the JSON
#                   -> if all values below minimum width and height -> gonna be splitting
#              -> do a minimum width and minimum height from JSON
#                   -> if all values above max -> gonna be stitching
                