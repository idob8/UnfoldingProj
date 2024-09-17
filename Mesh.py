import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
import numpy as np

from collections import defaultdict

class Tree:
    def __init__(self):
        self.edges = []
        self.nodes = set()
        self.root = None
        self.children = defaultdict(list)
    
    def add_edge(self, u, v, weight):
        self.edges.append((u, v, weight))
        self.nodes.update([u, v])
        
        if self.root is None:
            self.root = u
        
        self.children[u].append(v)
        self.children[v].append(u)

    def get_edges(self):
        return self.edges
    
    def get_nodes(self):
        return self.nodes
    
    def get_root(self):
        return self.root
    
    def get_children(self, node):
        return self.children[node]



class Polygon:
    def __init__(self, vertex_indices, weight=1.0, center=None, normal=None):
        self.vertex_indices = vertex_indices  
        self.weight = weight  
        self.center = center
        self.normal = normal

    def __iter__(self):
        # iteration over vertex_indices
        return iter(self.vertex_indices)
    
    def __len__(self):
        # Return the number of vertex indices
        return len(self.vertex_indices)


class Mesh:
    def __init__(self, vertices=None, faces=None):
        # vertices is a list of 3D points (x, y, z)
        # faces is a list of tuples, each tuple contains indices of the vertices that form a face
        self.vertices = vertices
        self.faces = [Polygon(face) for face in faces] if faces else []  # List of Polygon objects
        self.claculate_face_centers()
        self.claculate_face_normals()
        self.polygons_2d = {}

    def read_off(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if lines[0].strip() != 'OFF':
                raise ValueError("Not a valid OFF file")
            n_verts, n_faces, _ = map(int, lines[1].strip().split())
            self.vertices = [list(map(float, line.strip().split())) for line in lines[2:2+n_verts]]
            self.faces = [Polygon(list(map(int, line.strip().split()[1:]))) for line in lines[2+n_verts:2+n_verts+n_faces]]
        self.calculate_face_centers()

    def write_off(self, file_path):
        with open(file_path, 'w') as file:
            file.write("OFF\n")
            file.write(f"{len(self.vertices)} {len(self.faces)} 0\n")
            for vertex in self.vertices:
                file.write(f"{' '.join(map(str, vertex))}\n")
            for face in self.faces:
                file.write(f"{len(face)} {' '.join(map(str, face.vertex_indices))}\n")

    def claculate_face_centers(self):
        for face in self.faces:
            coords = np.array([self.vertices[i] for i in face])
            face.center = np.mean(coords, axis=0)

    def claculate_face_normals(self):
        for face in self.faces:
           v0, v1, v2 = [np.array(self.vertices[i]) for i in face.vertex_indices[:3]]
           face.normal = np.cross(v1 - v0, v2 - v0)
    
    def calculate_face_adjacency_matrix(self):
        num_faces = len(self.faces)
        adjacency_matrix = np.zeros((num_faces, num_faces), dtype=int)
        edge_to_faces = defaultdict(set)
        for i, face in enumerate(self.faces):
            edges = [(min(face.vertex_indices[j], face.vertex_indices[(j+1) % len(face)]), 
                      max(face.vertex_indices[j], face.vertex_indices[(j+1) % len(face)])) 
                     for j in range(len(face))]
            for edge in edges:
                edge_to_faces[edge].add(i)
        
        for adjacent_faces in edge_to_faces.values():
            if len(adjacent_faces) == 2:
                face1, face2 = adjacent_faces
                adjacency_matrix[face1, face2] = adjacency_matrix[face2, face1] = 1
        
        return adjacency_matrix
    
    def calculate_edge_weights(self):
        adjacency_matrix = self.calculate_face_adjacency_matrix()
        edge_weights = []
        for i in range(len(self.faces)):
            for j in range(i+1, len(self.faces)):
                if adjacency_matrix[i, j]:
                    weight = self.faces[i].weight / self.faces[j].weight
                    edge_weights.append(((i, j), weight))
        return edge_weights

    def visualize_mesh(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for face in self.faces:
            indices = face.vertex_indices
            for i in range(len(indices)):
                start = self.vertices[indices[i]]
                end = self.vertices[indices[(i+1) % len(indices)]]
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'k-')

        vertices = np.array(self.vertices)
        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

        plt.show()

    def unfold_mesh_along_tree(self, tree):
        self.polygons_2d = {}
        root = tree.get_root()
        self.unfold_face(root, None, tree)

    def unfold_face(self, face_index, parent_index, tree):
        if face_index in self.polygons_2d:
            return

        face = self.faces[face_index]
        
        if parent_index is None:
            # This is the root face. Place it at the origin.
            a, b, c = face.vertex_indices[:3]
            va, vb, vc = self.vertices[a], self.vertices[b], self.vertices[c]
            
            # Place first two vertices along x-axis
            x1 = np.linalg.norm(np.array(vb) - np.array(va))
            self.polygons_2d[face_index] = [(0, 0), (x1, 0)]
            
            # Calculate position of third vertex
            vec1 = np.array(vb) - np.array(va)
            vec2 = np.array(vc) - np.array(va)
            x2 = np.dot(vec1, vec2) / np.linalg.norm(vec1)
            y2 = np.sqrt(np.dot(vec2, vec2) - x2*x2)
            self.polygons_2d[face_index].append((x2, y2))
        else:
            # This face shares an edge with its parent. Use that information to place it.
            parent_face = self.faces[parent_index]
            shared_edge = list(set(face.vertex_indices) & set(parent_face.vertex_indices))
            
            if len(shared_edge) != 2:
                raise ValueError(f"Faces {face_index} and {parent_index} don't share an edge")
            
            # Get 2D coordinates of shared edge in parent face
            parent_2d = self.polygons_2d[parent_index]
            edge_2d = [parent_2d[parent_face.vertex_indices.index(v)] for v in shared_edge]
            
            # Find the vertex of this face that's not in the shared edge
            new_vertex = [v for v in face.vertex_indices if v not in shared_edge][0]
            
            # Calculate 2D position of new vertex
            v0, v1 = [np.array(self.vertices[v]) for v in shared_edge]
            v2 = np.array(self.vertices[new_vertex])
            
            edge_vec = v1 - v0
            edge_len = np.linalg.norm(edge_vec)
            
            proj = np.dot(v2 - v0, edge_vec) / edge_len
            height = np.sqrt(np.dot(v2 - v0, v2 - v0) - proj*proj)
            
            edge_2d_vec = np.array(edge_2d[1]) - np.array(edge_2d[0])
            normal = np.array([-edge_2d_vec[1], edge_2d_vec[0]])
            normal = normal / np.linalg.norm(normal)
            
            new_point = np.array(edge_2d[0]) + (proj/edge_len)*edge_2d_vec + height*normal
            
            # Store the 2D coordinates
            self.polygons_2d[face_index] = edge_2d + [tuple(new_point)]

        # Recursively unfold children
        for child in tree.get_children(face_index):
            if child != parent_index:
                self.unfold_face(child, face_index, tree)

    def visualize_flattened_mesh(self):
        fig, ax = plt.subplots()
        for face_index, vertices_2d in self.polygons_2d.items():
            polygon = plt.Polygon(vertices_2d, fill=None, edgecolor='r')
            ax.add_patch(polygon)
            centroid = np.mean(vertices_2d, axis=0)
            ax.text(centroid[0], centroid[1], str(face_index), ha='center', va='center')

        ax.autoscale()
        ax.set_aspect('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Flattened Mesh')
        plt.show()
