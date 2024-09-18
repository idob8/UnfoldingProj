import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from Tree import Tree
import heapq
from Mesh import Mesh

class Unfolder:
    def __init__(self,mesh):
        self.mesh = mesh
        self.polygons_2d = {}
    
    def find_approximated_mst(self):
        edge_weights = self.mesh.calculate_dual_edge_weights()
        dual_graph = defaultdict(list)
        for (u, v), weight in edge_weights:
            dual_graph[u].append((v, weight))
            dual_graph[v].append((u, 1/weight))
        # Initialize the MST
        mst = Tree()
        visited = set()
        # Start with the first face
        start_node = 0
        visited.add(start_node)
        mst.root = start_node

        pq = [(weight, start_node, neighbor) for neighbor, weight in dual_graph[start_node]]
        heapq.heapify(pq)

        while pq:
            weight, u, v = heapq.heappop(pq)
            if v not in visited:
                visited.add(v)
                mst.add_edge(u, v, weight)
                for neighbor, w in dual_graph[v]:
                    if neighbor not in visited:
                        heapq.heappush(pq, (w, v, neighbor))
        return mst

   
    def unfold_mesh_along_tree(self, tree):
        root = tree.get_root()
        self.unfold_face(root, None, tree)

    def unfold_face(self, face_index, parent_index, tree):
        if face_index in self.polygons_2d:
            return
        face = self.mesh.faces[face_index]
        if parent_index is None:
            # This is the root face. Place it at the origin.
            a, b, c = face.vertex_indices[:3]
            va, vb, vc = self.mesh.vertices[a], self.mesh.vertices[b], self.mesh.vertices[c]
            
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
            parent_face = self.mesh.faces[parent_index]
            shared_edge = list(set(face.vertex_indices) & set(parent_face.vertex_indices))
            
            if len(shared_edge) != 2:
                raise ValueError(f"Faces {face_index} and {parent_index} don't share an edge")
            
            # Get 2D coordinates of shared edge in parent face
            parent_2d = self.polygons_2d[parent_index]
            edge_2d = [parent_2d[parent_face.vertex_indices.index(v)] for v in shared_edge]
            
            # Find the vertex of this face that's not in the shared edge
            new_vertex = [v for v in face.vertex_indices if v not in shared_edge][0]
            
            # Calculate 2D position of new vertex
            v0, v1 = [np.array(self.mesh.vertices[v]) for v in shared_edge]
            v2 = np.array(self.mesh.vertices[new_vertex])
            
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

    def visualize_unfolded_mesh(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        for face_index, vertices_2d in self.polygons_2d.items():
            polygon = plt.Polygon(vertices_2d, fill=None, edgecolor='black')
            ax.add_patch(polygon)
            # centroid = np.mean(vertices_2d, axis=0)
            # ax.text(centroid[0], centroid[1], str(face_index), ha='center', va='center')

        ax.autoscale()
        ax.set_aspect('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Unfolded Mesh')
        plt.show()
