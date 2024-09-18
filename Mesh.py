import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from collections import defaultdict
from Tree import Tree
import heapq
import random
xzero = 0.0001 

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
        self.initilize_mesh()

    def read_off(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if lines[0].strip() != 'OFF':
                raise ValueError("Not a valid OFF file")
            n_verts, n_faces, _ = map(int, lines[1].strip().split())
            self.vertices = [list(map(float, line.strip().split())) for line in lines[2:2+n_verts]]
            self.faces = [Polygon(list(map(int, line.strip().split()[1:]))) for line in lines[2+n_verts:2+n_verts+n_faces]]
        self.initilize_mesh()
        
    def initilize_mesh(self):
        self.calculate_face_centers()
        self.calculate_face_normals()
        self.face_vertices = [[self.vertices[idx] for idx in face.vertex_indices] for face in self.faces]
        self.poly3d = Poly3DCollection(self.face_vertices, facecolors='white', edgecolors='black', alpha=0.8)
    
    def write_off(self, file_path):
        with open(file_path, 'w') as file:
            file.write("OFF\n")
            file.write(f"{len(self.vertices)} {len(self.faces)} 0\n")
            for vertex in self.vertices:
                file.write(f"{' '.join(map(str, vertex))}\n")
            for face in self.faces:
                file.write(f"{len(face)} {' '.join(map(str, face.vertex_indices))}\n")


    def calculate_face_centers(self):
        for face in self.faces:
            coords = np.array([self.vertices[i] for i in face])
            face.center = np.mean(coords, axis=0)

    def calculate_face_normals(self):
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
    
    def calculate_dual_edge_weights(self):
        adjacency_matrix = self.calculate_face_adjacency_matrix()
        edge_weights = []
        for i in range(len(self.faces)):
            for j in range(i+1, len(self.faces)):
                if adjacency_matrix[i, j]:
                    weight_i = self.faces[i].weight if self.faces[i].weight != 0.0 else xzero
                    weight_j = self.faces[j].weight if self.faces[j].weight != 0.0 else xzero
                    weight = weight_i / weight_j
                    edge_weights.append(((i, j), weight))
        return edge_weights

   
    def visualize(self):
       fig = plt.figure(figsize=(10, 10))
       ax = fig.add_subplot(111, projection='3d')
       ax.add_collection3d(self.poly3d)
       ax.set_xlabel('X')
       ax.set_ylabel('Y')
       ax.set_zlabel('Z')
       # Auto scaling the axes
       ax.auto_scale_xyz([v[0] for v in self.vertices],
                          [v[1] for v in self.vertices],
                          [v[2] for v in self.vertices])

       plt.title('Mesh Visualization')
       plt.show()

    def assign_face_weights_by_area(self):
        for face in self.faces:
            coords = np.array([self.vertices[i] for i in face.vertex_indices])
            if len(coords) >= 3:
                v0, v1, v2 = coords[0], coords[1], coords[2]
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                face.weight = area if area != 0 else xzero  
            else:
                face.weight = xzero

    def assign_face_weights_random(self):
        for face in self.faces:
            weight = random.random()
            face.weight = weight if weight != 0 else xzero

    def assign_face_weights_custom(self, custom_weights):
        if custom_weights is None or len(custom_weights) != len(self.faces):
            raise ValueError("Custom weights must be provided and match the number of faces.")
        for face, weight in zip(self.faces, custom_weights):
            face.weight = weight if weight != 0 else xzero 
