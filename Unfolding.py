from Mesh import Mesh
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import networkx as nx
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


def flatten_faces(centers, mst):
    G = nx.Graph()
    for weight, u, v in mst.get_edges():
        G.add_edge(u, v, weight=weight)
    
    pos = nx.spring_layout(G, seed=42)
    return pos

def draw_flattened_mesh(mesh, flattened_positions):
    plt.figure(figsize=(10, 8))
    
    # Draw edges of the mesh
    for face in mesh.faces:
        for i in range(len(face)):
            v1 = face[i]
            v2 = face[(i + 1) % len(face)]
            plt.plot([mesh.vertices[v1][0], mesh.vertices[v2][0]], [mesh.vertices[v1][1], mesh.vertices[v2][1]], 'k-')

    plt.title('Flattened 2D Mesh')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()


# Example usage
vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
faces = [(0, 1, 2), (1, 2, 3)]

#mesh = Mesh(vertices, faces)

mesh = Mesh()
mesh.read_off(r"C:\Users\ido.b\Documents\Technion\semester H\DigitalGeometryProcessing\HW2\hw2_data\sphere_s0.off")
# Example weights matrix between faces


# Compute MST
dual_vertices, dual_edges = mesh.get_dual_directed_graph()
mst = prim_mst_directed(dual_vertices, dual_edges,mesh.calculate_edge_weights())
print(mst.edges)
print(mst.nodes)
# Flatten faces using the MST
flattened_positions = flatten_faces(mesh.face_centers, mst)

# Draw flattened mesh
draw_flattened_mesh(mesh, flattened_positions)

