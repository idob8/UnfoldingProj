from Mesh import Mesh
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
#from scipy.spatial import ConvexHull
import heapq
from Tree import Tree 
from Unfolder import Unfolder, Polygon2D

vertices = [
    [1, 1, 1],
    [-1, -1, 1],
    [-1, 1, -1],
    [1, -1, -1]
]

faces = [
    [0, 1, 2],
    [0, 1, 3],
    [0, 2, 3],
    [1, 2, 3]
]


# # Define polygons as dictionaries of vertices
# vertices_2d_triangle1 = {0: (0, 0), 1: (4, 0), 2: (2, 3)}  # Triangle 1
# vertices_2d_triangle2 = {0: (2, 3), 1: (6, 0), 2: (4, 0)}  # Triangle 2, sharing edge (2, 3)-(4, 0)

# # Create Polygon2D instance and add polygons
# interPoly = Polygon2D()
# interPoly.add_polygon(1, vertices_2d_triangle1)
# interPoly.add_polygon(2, vertices_2d_triangle2)
# # Check for intersection
# collision = interPoly.check_face_collision(1, 2)
# print(f"Polygons intersect: {collision}") 

mesh = Mesh()
mesh.read_off("/Users/meravkeidar/OneDrive/Technion/semester4/DGP/DigitalGeometryProcessing/HW2/hw2_data/sphere_s0.off")

# mesh = Mesh(vertices= vertices, faces= faces)
print("Visualizing original 3D mesh:")
mesh.visualize()
# unfolder = Unfolder(mesh)
# tree = unfolder.steepest_edge_unfolder()
# unfolder.unfold_mesh_along_tree(tree)
# unfolder.polygon_2d.visualize_polygons()
# collisions = unfolder.polygon_2d.detect_all_collisions()
# print(f"number of collisions {len(collisions)} ")

# mesh.conformalized_mean_curvature_flow(1,0.1)
mesh.edge_normalizing_flow(1,0.1)
print("Visualizing transformed 3D mesh:")
mesh.visualize()
# unfolder = Unfolder(mesh)
# tree = unfolder.steepest_edge_unfolder()
# unfolder.unfold_mesh_along_tree(tree)
# unfolder.polygon_2d.visualize_polygons()
# collisions = unfolder.polygon_2d.detect_all_collisions()
# print(f"number of collisions {len(collisions)} ")


# for i in range(4):
#     mesh.assign_face_weights_random()
#     unfolder = Unfolder(mesh)
#     tree = unfolder.find_approximated_mst()
#     unfolder.unfold_mesh_along_tree(tree)
#     print("Visualizing flattened 2D mesh:")
#     unfolder.visualize_unfolded_mesh()



