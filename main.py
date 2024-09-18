from Mesh import Mesh
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
#from scipy.spatial import ConvexHull
import heapq
from Tree import Tree 
from Unfolder import Unfolder

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

mesh = Mesh()
mesh.read_off(r"C:\Users\user\OneDrive\Merav\Technion\DGP\data\phands.off")

print("Visualizing original 3D mesh:")
mesh.visualize()


for i in range(4):
    mesh.assign_face_weights_random()
    unfolder = Unfolder(mesh)
    tree = unfolder.find_approximated_mst()
    unfolder.unfold_mesh_along_tree(tree)
    print("Visualizing flattened 2D mesh:")
    unfolder.visualize_unfolded_mesh()