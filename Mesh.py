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
from scipy.spatial.transform import Rotation
xzero = 0.0001 

def compute_triangle_area(coords):
    v1, v2, v3 = coords
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

class Face:
    def __init__(self, vertex_indices, weight=1.0, center=None, normal=None, area =None):
        self.vertex_indices = vertex_indices  
        self.weight = weight  
        self.center = center
        self.normal = normal
        self.area = area

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
        self.calculate_face_areas()
        self.calculate_genus()
        self.calculate_cot_matrix()

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
           normal = np.cross(v1 - v0, v2 - v0)
           norm = np.linalg.norm(normal)
           if norm > 0:
               face.normal = normal / norm
           else:
               print(f"Warning: Degenerate face detected at indices {face.vertex_indices}")
               face.normal = np.array([0.0, 0.0, 0.0])
    
    def calculate_face_areas(self):
        for face in self.faces:
            coords = [np.array(self.vertices[i]) for i in face.vertex_indices[:3]]
            face.area = compute_triangle_area(coords)
    
    def calculate_vertex_normals(self):
        vertex_normals = np.zeros_like(self.vertices)
        for face in self.faces:
            face_normal = face.normal
            for vertex_idx in face.vertex_indices:
                vertex_normals[vertex_idx] += face_normal
        vertex_normals = (vertex_normals.T / np.linalg.norm(vertex_normals, axis=1)).T  # Normalize 
        return vertex_normals
    
    def get_vertex_masses(self):
        vertex_masses = np.zeros(self.vertices.shape[0])
        for face in self.faces:
            for vertex_idx in face.vertex_indices:
                vertex_masses[vertex_idx] += face.area / 3
        return vertex_masses

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


    def mean_curvature_flow(self, n_iterations, step_factor):
        """
        Apply Mean Curvature Flow to the mesh.
        :param n_iterations: Number of iterations to perform
        :param step_factor: Step factor (t in the equation)
        """
        if self.genus != 0:
            print(f"Warning: This mesh has genus {self.genus}. The MCF algorithm is designed for genus-zero surfaces.")
        L =  self.cot_matrix
        for i in range(n_iterations):
            M = self._compute_mass_matrix()
            # Solve the equation: (M - t*L) * V_{n+1} = M * V_n
            A = M - step_factor * L
            b = M @ self.vertices
            new_vertices = spsolve(A, b)
            if np.any(np.isnan(new_vertices)):
                print("Warning: NaN values detected, skipping this iteration of MCF")
                continue
            self.vertices = new_vertices
            self.vertices = new_vertices
            self.shift_and_normalize()
        self.update_properties()
    
    def shift_and_normalize(self):
        center = np.mean(self.vertices, axis=0)
        self.vertices -= center
        scale = np.max(np.linalg.norm(self.vertices, axis=1))
        if scale > 0:
            self.vertices /= scale
        else:
            print("Warning: Scale is zero, skipping normalization")

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

    def calculate_cot_matrix(self):
        num_vertices = len(self.vertices)
        row, col, data = [], [], []
        for face in self.faces:
            v_indices = face.vertex_indices
            coords = np.array([self.vertices[i] for i in v_indices])
            cot_weights = self.calculate_cotangent_weights(coords)
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                weight = cot_weights[k]
                if np.isfinite(weight) and abs(weight) > 1e-10:  # Avoid very small weights
                    row.extend([v_indices[i], v_indices[j], v_indices[i], v_indices[j]])
                    col.extend([v_indices[j], v_indices[i], v_indices[i], v_indices[j]])
                    data.extend([weight, weight, -weight, -weight])
        self.cot_matrix = csr_matrix((data, (row, col)), shape=(num_vertices, num_vertices))
        self.cot_matrix.setdiag(self.cot_matrix.diagonal() + 1e-8)  # Add small regularization term
    
    def calculate_cotangent_weights(self, coords):
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

    def setup_edge_normals(self):
        self.edge_normals = [{} for _ in range(len(self.vertices))]
        
        for face in self.faces:
            for i in range(3):
                v1, v2 = face.vertex_indices[i], face.vertex_indices[(i+1) % 3]
                if v2 not in self.edge_normals[v1]:
                    self.edge_normals[v1][v2] = np.zeros(3)
                    self.edge_normals[v2][v1] = np.zeros(3)
                
                self.edge_normals[v1][v2] += face.normal * face.area
                self.edge_normals[v2][v1] += face.normal * face.area
        
        for i in range(len(self.vertices)):
            for j in self.edge_normals[i]:
                self.edge_normals[i][j] = self.edge_normals[i][j] / np.linalg.norm(self.edge_normals[i][j])

    def edge_normalizing_rhs(self, step_size):
        # Sets up the right-hand side of the equation for the flow.
        vertex_normals = self.calculate_vertex_normals()
        vertex_masses = self.get_vertex_masses()
        rhs = np.zeros_like(self.vertices)

        for i in range(len(self.vertices)):
            for j in self.edge_normals[i]:
                # Calculate the target normal as the average of vertex normals
                target_normal = (vertex_normals[i] / vertex_masses[i] + 
                                 vertex_normals[j] / vertex_masses[j])
                target_normal /= np.linalg.norm(target_normal)
                # Current edge normal
                current_normal = self.edge_normals[i][j]
                # Calculate the rotation to align current_normal with target_normal
                rotation_axis = np.cross(current_normal, target_normal)
                rotation_angle = np.arccos(np.clip(np.dot(current_normal, target_normal), -1.0, 1.0))
                rotation = Rotation.from_rotvec(step_size * rotation_angle * rotation_axis)
                # Apply rotation to the edge
                edge = self.vertices[j] - self.vertices[i]
                rotated_edge = rotation.apply(edge)
                # Update the right-hand side
                cot_weight = self.cot_matrix[i, j]
                rhs[i] += cot_weight * rotated_edge
                rhs[j] -= cot_weight * rotated_edge
        return rhs
    
    def calculate_angle_deficits(self):
        angle_deficits = np.zeros(len(self.vertices))
        for i, vertex in enumerate(self.vertices):
            adjacent_faces = [face for face in self.faces if i in face.vertex_indices]
            total_angle = 0
            for face in adjacent_faces:
                other_vertices = [self.vertices[j] for j in face.vertex_indices if j != i]
                v1, v2 = other_vertices
                e1 = v1 - vertex
                e2 = v2 - vertex
                angle = np.arccos(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
                total_angle += angle
            
            # The angle deficit is 2π minus the total angle
            angle_deficits[i] = 2 * np.pi - total_angle
        
        return angle_deficits

    def calculate_total_squared_angle_deficit(self):
        angle_deficits = self.calculate_angle_deficits()
        return np.sum(angle_deficits**2)

    def edge_normal_alignment_flow(self, n_iterations, step_size):
        for iteration in range(n_iterations):
            self.setup_edge_normals()
            # Initialize the right-hand side of our system
            rhs = self.edge_normalizing_rhs(step_size=step_size)
            # Solve the system to get updated vertex positions
            new_vertices = spsolve(self.cot_matrix, rhs)
            # Check for NaN values
            if np.any(np.isnan(new_vertices)):
                print(f"Warning: NaN values detected in iteration {iteration}, stopping ENAF")
                break
            self.vertices = new_vertices.reshape(-1, 3)
            # Normalize the mesh
            self.shift_and_normalize()
            self.update_properties()
        