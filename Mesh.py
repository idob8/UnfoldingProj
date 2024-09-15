import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
import numpy as np

from collections import defaultdict

class Mesh:
    def __init__(self, vertices=None, faces=None):
        # vertices is a list of 3D points (x, y, z)
        # faces is a list of tuples, each tuple contains indices of the vertices that form a face
        self.vertices = vertices  # [(x, y, z), (x, y, z), ...]
        self.faces = faces        # [(v1, v2, v3), (v4, v5, v6), ...]
        self.face_weights = None if vertices == None else np.ones(len(self.faces))
        self.face_centers = None if vertices == None else self.claculate_face_centers()

    def read_off(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            # Ensure the file starts with 'OFF'
            if lines[0].strip() != 'OFF':
                raise ValueError("Not a valid OFF file")
            
            # Read number of vertices, faces, and edges
            n_verts, n_faces, _ = map(int, lines[1].strip().split())
            
            # Read vertices
            self.vertices = []
            for i in range(2, 2 + n_verts):
                self.vertices.append(list(map(float, lines[i].strip().split())))
            
            # Read faces
            self.faces = []
            for i in range(2 + n_verts, 2 + n_verts + n_faces):
                face_data = list(map(int, lines[i].strip().split()))
                # First number is the number of vertices in the face, ignore it
                self.faces.append(face_data[1:])
            
            self.face_weights = np.ones(len(self.faces))
            self.face_centers = self.claculate_face_centers()
            

    def write_off(self, file_path):
        with open(file_path, 'w') as file:
            # Write the header
            file.write("OFF\n")
            
            # Write the number of vertices, faces, and edges (set edges to 0)
            file.write(f"{len(self.vertices)} {len(self.faces)} 0\n")
            
            # Write the vertices
            for vertex in self.vertices:
                file.write(f"{' '.join(map(str, vertex))}\n")
            
            # Write the faces
            for face in self.faces:
                file.write(f"{len(face)} {' '.join(map(str, face))}\n")

    def claculate_face_centers(self):
        centers = []
        for face in self.faces:
            coords = np.array([self.vertices[i] for i in face])
            center = np.mean(coords, axis=0)
            centers.append(center)
        return np.array(centers)
    
    def calculate_face_adjacency_matrix(self):
        num_faces = len(self.faces)
        adjacency_matrix = np.zeros((num_faces, num_faces), dtype=int)
        edge_to_faces = defaultdict(set)
        for i, face in enumerate(self.faces):
            edges = [(min(face[j], face[(j+1) % len(face)]), max(face[j], face[(j+1) % len(face)])) for j in range(len(face))]
            for edge in edges:
                edge_to_faces[edge].add(i)
        
        for edge, adjacent_faces in edge_to_faces.items():
            if len(adjacent_faces) == 2:
                face1, face2 = adjacent_faces
                adjacency_matrix[face1, face2] = 1
                adjacency_matrix[face2, face1] = 1
        
        return adjacency_matrix
    
    def get_dual_directed_graph(self):
        adjacency_matrix = self.calculate_face_adjacency_matrix()
        num_faces = len(adjacency_matrix)
        vertices = list(range(num_faces))
        edges = []
        for i in range(num_faces):
            for j in range(num_faces):
                if adjacency_matrix[i, j] == 1:
                    edges.append((i, j))
        
        return vertices, edges
    
    def calculate_edge_weights(self):
        # Ensure the face weights are not zero to avoid division by zero
        if np.any(self.face_weights == 0):
            raise ValueError("Face weights must be non-zero to calculate edge weights.")
        vertices, edges = self.get_dual_directed_graph()
        edge_weights = []
        for i, j in edges:
            weight = self.face_weights[i] / self.face_weights[j]
            edge_weights.append(((i, j), weight))
        
        return edge_weights

    def visualize_mesh(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create a list of lines (each line is an edge between two vertices)
        edges = []
        for face in self.faces:
            # Loop through vertices in each face and create edges between consecutive vertices
            for i in range(len(face)):
                edge = [self.vertices[face[i]], self.vertices[face[(i + 1) % len(face)]]]  # Wrap around to create the last edge
                edges.append(edge)

        # Add the edges to the plot
        edge_collection = Line3DCollection(edges, colors='b', linewidths=1)
        ax.add_collection3d(edge_collection)

        # Auto-scaling the axes for better visualization
        ax.auto_scale_xyz(
            [v[0] for v in self.vertices],
            [v[1] for v in self.vertices],
            [v[2] for v in self.vertices]
        )

        # Labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

        

    