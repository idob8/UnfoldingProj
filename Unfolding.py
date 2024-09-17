from Mesh import Mesh
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
#from scipy.spatial import ConvexHull
import heapq

class Tree:
    def __init__(self):
        self.edges = []  # List to store edges in the MST
        self.nodes = set()  # Set to store nodes included in the MST
        self.root = None  # Root node of the tree
        self.children = {}  # Dictionary to store children of each node
    
    def add_edge(self, u, v, weight):
        self.edges.append((u, v, weight))
        self.nodes.add(u)
        self.nodes.add(v)
        
        # Set the first added node as the root
        if self.root is None:
            self.root = u
        
        # Add v as a child of u, or u as a child of v, based on the edge direction
        if u not in self.children:
            self.children[u] = []
        self.children[u].append(v)
        
        if v not in self.children:
            self.children[v] = []
        self.children[v].append(u)

    def get_edges(self):
        return self.edges
    
    def get_nodes(self):
        return self.nodes
    
    def get_root(self):
        return self.root
    
    def get_children(self, node):
        if node in self.children:
            return self.children[node]
        else:
            return []



def prim_mst_directed(vertices, edges, edge_weights):
    # Initialize the MST
    mst = Tree()
    visited = set()
    min_heap = []
    # Arbitrarily start from the first vertex
    start_vertex = vertices[0]
    visited.add(start_vertex)
    # Add all edges from the start_vertex to the heap
    for (u, v), weight in edge_weights:
        if u == start_vertex:
            heapq.heappush(min_heap, (weight, u, v))
    
    while min_heap:
        weight, u, v = heapq.heappop(min_heap)
        
        if v not in visited:
            visited.add(v)
            mst.add_edge(u, v, weight)
            
            # Add all edges from the new vertex to the heap
            for (x, y), edge_weight in edge_weights:
                if x == v and y not in visited:
                    heapq.heappush(min_heap, (edge_weight, x, y))
    
    return mst


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

mesh = Mesh(vertices, faces)

tree = Tree()
tree.add_edge(0, 1, 1.0)
tree.add_edge(0, 2, 1.0)
tree.add_edge(0, 3, 1.0)

mesh.unfold_mesh_along_tree(tree)

print("Visualizing original 3D mesh:")
mesh.visualize_mesh()

print("Visualizing flattened 2D mesh:")
mesh.visualize_flattened_mesh()