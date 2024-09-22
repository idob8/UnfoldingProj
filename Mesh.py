import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from collections import defaultdict
from Tree import Tree
import heapq
import random
from scipy.sparse import csr_matrix,lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import eigs
xzero = 0.0001 

def compute_triangle_area(coords):
    v1, v2, v3 = coords
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

class Face:
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
        self.vertices = np.array(vertices) if vertices is not None else None
        self.faces = [Face(face) for face in faces] if faces else []
        if self.vertices is not None and self.faces:
            self.update_properties()

    def read_off(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if lines[0].strip() != 'OFF':
                raise ValueError("Not a valid OFF file")
            n_verts, n_faces, _ = map(int, lines[1].strip().split())
            self.vertices = np.loadtxt(lines[2:2+n_verts], dtype=float)
            self.faces = [Face(list(map(int, line.strip().split()[1:]))) for line in lines[2+n_verts:2+n_verts+n_faces]]
        self.update_properties()
        

    def update_properties(self):
        self.face_vertices = [[self.vertices[idx] for idx in face.vertex_indices] for face in self.faces]
        self.poly3d = Poly3DCollection(self.face_vertices, facecolors='white', edgecolors='black', alpha=0.8)
        self.calculate_face_centers()
        self.calculate_face_normals()
        self.calculate_genus()


    def write_off(self, file_path):
        with open(file_path, 'w') as file:
            file.write("OFF\n")
            file.write(f"{len(self.vertices)} {len(self.faces)} 0\n")
            for vertex in self.vertices:
                file.write(f"{' '.join(map(str, vertex))}\n")
            for face in self.faces:
                file.write(f"{len(face)} {' '.join(map(str, face.vertex_indices))}\n")

    def calculate_genus(self):
        V = len(self.vertices)
        F = len(self.faces)
        # Calculate the number of edges
        edges = set()
        for face in self.faces:
            for i in range(len(face)):
                edge = tuple(sorted([face.vertex_indices[i], face.vertex_indices[(i+1) % len(face.vertex_indices)]]))
                edges.add(edge)
        E = len(edges)
        # Euler characteristic
        euler_characteristic = V - E + F
        self.genus = (2 - euler_characteristic) // 2
        
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
       # Set equal aspect ratio
       x = self.vertices[:,0]
       y = self.vertices[:,1]
       z = self.vertices[:,2]
       max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
       mid_x = (x.max()+x.min()) * 0.5
       mid_y = (y.max()+y.min()) * 0.5
       mid_z = (z.max()+z.min()) * 0.5
       ax.set_xlim(mid_x - max_range, mid_x + max_range)
       ax.set_ylim(mid_y - max_range, mid_y + max_range)
       ax.set_zlim(mid_z - max_range, mid_z + max_range)
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


    def conformalized_mean_curvature_flow(self, n_iterations, step_factor):
        """
        Apply Conformalized Mean Curvature Flow to the mesh.
        :param n_iterations: Number of iterations to perform
        :param step_factor: Step factor (t in the equation)
        """
        if self.genus != 0:
            print(f"Warning: This mesh has genus {self.genus}. The cMCF algorithm is designed for genus-zero surfaces.")
        L = self._compute_stiffness_matrix()
        for i in range(n_iterations):
            M = self._compute_mass_matrix()
            # Solve the equation: (M - t*L) * V_{n+1} = M * V_n
            A = M - step_factor * L
            b = M @ self.vertices
            new_vertices = spsolve(A, b)
            self.vertices = new_vertices
            self.shift_and_normalize()
        self.update_properties()
    
    def shift_and_normalize(self):
        center = np.mean(self.vertices, axis=0)
        self.vertices -= center
        scale = np.max(np.linalg.norm(self.vertices, axis=1))
        self.vertices /= scale

    def _compute_mass_matrix(self):
        num_vertices = len(self.vertices)
        row, col, data = [], [], []
        for face in self.faces:
            v_indices = face.vertex_indices
            coords = np.array([self.vertices[i] for i in v_indices])
            area = compute_triangle_area(coords)
            for i in v_indices:
                row.append(i)
                col.append(i)
                data.append(area / 3)  # Barycentric mass distribution
        return csr_matrix((data, (row, col)), shape=(num_vertices, num_vertices))

    def _compute_stiffness_matrix(self):
        num_vertices = len(self.vertices)
        row, col, data = [], [], []
        for face in self.faces:
            v_indices = face.vertex_indices
            coords = np.array([self.vertices[i] for i in v_indices])
            cot_weights = self._compute_cotangent_weights(coords)
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                weight = cot_weights[k]
                if np.isfinite(weight) and abs(weight) > 1e-10:  # Avoid very small weights
                    row.extend([v_indices[i], v_indices[j], v_indices[i], v_indices[j]])
                    col.extend([v_indices[j], v_indices[i], v_indices[i], v_indices[j]])
                    data.extend([weight, weight, -weight, -weight])
        return csr_matrix((data, (row, col)), shape=(num_vertices, num_vertices))
    
    def _compute_cotangent_weights(self, coords):
        # Compute cotangent weights for a triangle
        v1, v2, v3 = coords
        e1, e2, e3 = v2 - v3, v3 - v1, v1 - v2
        def safe_cot(x, y):
            cross = np.cross(x, y)
            norm_cross = np.linalg.norm(cross)
            if norm_cross < 1e-10:
                return 0.0
            return np.clip(np.dot(x, y) / norm_cross, -1e3, 1e3)
        return [
            0.5 * safe_cot(e2, -e3),
            0.5 * safe_cot(e3, -e1),
            0.5 * safe_cot(e1, -e2)
        ]
        
    