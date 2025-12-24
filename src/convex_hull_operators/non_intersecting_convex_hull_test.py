import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random

def generate_non_intersecting_plot(seed_value=101):
    # Set seed for reproducibility
    np.random.seed(seed_value)
    random.seed(seed_value)

    fig, ax = plt.subplots(figsize=(10, 5))
    # Set plot limits to create distinct zones
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    
    def random_color():
        return (random.random(), random.random(), random.random())

    # Define centers for each shape to ensure separation
    # Zone 1: x in [0, 4] -> center at (2, 3)
    # Zone 2: x in [4, 8] -> center at (6, 3)
    # Zone 3: x in [8, 12] -> center at (10, 3)
    centers = [(2, 3), (6, 3), (10, 3)]
    
    # --- 1. Rectangle (Zone 1) ---
    cx, cy = centers[0]
    rect_w = np.random.uniform(1, 2)
    rect_h = np.random.uniform(1, 2)
    rect_angle = np.random.uniform(0, 360)
    
    # Vertices relative to center
    v_rel = np.array([
        [-rect_w/2, -rect_h/2],
        [rect_w/2, -rect_h/2],
        [rect_w/2, rect_h/2],
        [-rect_w/2, rect_h/2]
    ])
    # Rotate
    theta = np.radians(rect_angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    v_rot = v_rel.dot(R.T)
    # Translate to center
    v_final = v_rot + np.array([cx, cy])
    
    rect_poly = patches.Polygon(v_final, closed=True, linewidth=2, edgecolor='black', facecolor=random_color(), alpha=0.6, label='Rectangle')
    ax.add_patch(rect_poly)

    # --- 2. Right Triangle (Zone 2) ---
    cx, cy = centers[1]
    tri_base = np.random.uniform(1.5, 2.5)
    tri_height = np.random.uniform(1.5, 2.5)
    
    # Vertices with right angle at origin
    v_tri = np.array([[0, 0], [tri_base, 0], [0, tri_height]])
    # Calculate centroid
    centroid = np.mean(v_tri, axis=0)
    # Center the triangle at (0,0)
    v_tri_centered = v_tri - centroid
    
    # Rotate
    tri_angle = np.random.uniform(0, 360)
    theta = np.radians(tri_angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    v_tri_rot = v_tri_centered.dot(R.T)
    
    # Translate to zone center
    v_tri_final = v_tri_rot + np.array([cx, cy])
    
    tri_poly = patches.Polygon(v_tri_final, closed=True, linewidth=2, edgecolor='black', facecolor=random_color(), alpha=0.6, label='Triangle')
    ax.add_patch(tri_poly)

    # --- 3. Hexagon (Zone 3) ---
    cx, cy = centers[2]
    hex_radius = np.random.uniform(0.8, 1.5)
    hex_orientation = np.radians(np.random.uniform(0, 360))
    
    # RegularPolygon is naturally defined by its center
    hexagon = patches.RegularPolygon((cx, cy), numVertices=6, radius=hex_radius, orientation=hex_orientation,
                                     linewidth=2, edgecolor='black', facecolor=random_color(), alpha=0.6, label='Hexagon')
    ax.add_patch(hexagon)

    # Finalize Plot
    ax.legend(loc='upper right')
    ax.set_title(f"Non-Intersecting Shapes (Seed: {seed_value})")
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.show()

if __name__ == "__main__":
    generate_non_intersecting_plot(seed_value=101)