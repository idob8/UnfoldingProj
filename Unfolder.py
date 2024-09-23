import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from Tree import Tree
import heapq
from Mesh import Mesh
import PolygonCollision
from PolygonCollision.shape import Shape
import random
from math import sqrt

class Polygon2D:
    def __init__(self):
        self.polygons = {}
        self.bounding_boxes = {}

    def add_polygon(self, face_index, vertices):
        self.polygons[face_index] = vertices
        self.compute_bounding_box(face_index)

    def compute_bounding_box(self, face_index):
        vertices = list(self.polygons[face_index].values())
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        self.bounding_boxes[face_index] = (
            np.array([min_x, min_y]),
            np.array([max_x, max_y])
        )

    def bounding_boxes_intersect(self, face1_index, face2_index):
        (min1_x, min1_y), (max1_x, max1_y) = self.bounding_boxes[face1_index]
        (min2_x, min2_y), (max2_x, max2_y) = self.bounding_boxes[face2_index]
        
        return (min1_x <= max2_x and max1_x >= min2_x and
                min1_y <= max2_y and max1_y >= min2_y)
        
    def check_face_collision(self, face1_index, face2_index):
        if face1_index == face2_index:
            return False
        if not self.bounding_boxes_intersect(face1_index, face2_index):
            return False
        
        poly1 = list(self.polygons[face1_index].values())
        poly2 = list(self.polygons[face2_index].values())
        
        # Check if polygons share an edge
        # edges1 = set(frozenset([tuple(poly1[i]), tuple(poly1[(i+1)%len(poly1)])]) for i in range(len(poly1)))
        # edges2 = set(frozenset([tuple(poly2[i]), tuple(poly2[(i+1)%len(poly2)])]) for i in range(len(poly2)))
        # shared_edges = edges1.intersection(edges2)

        # if shared_edges:
            # If they share an edge, check if they overlap
        return self.check_overlap(poly1, poly2)
    
        # If no shared edge, use Separating Axis Theorem
        # return self.separating_axis_theorem(poly1, poly2)
    
    def check_overlap(self, poly1, poly2):
        for point in poly1:
            if self.point_inside_polygon(point, poly2):
                return True
        # for point in poly2:
        #     if self.point_inside_polygon(point, poly1):
        #         return True
        return False

    def point_inside_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def separating_axis_theorem(self, poly1, poly2):
        for polygon in [poly1, poly2]:
            for i in range(len(polygon)):
                edge = np.array(polygon[i]) - np.array(polygon[i-1])
                normal = np.array([-edge[1], edge[0]])

                min_p1, max_p1 = float('inf'), float('-inf')
                min_p2, max_p2 = float('inf'), float('-inf')

                for vertex in poly1:
                    projection = np.dot(normal, vertex)
                    min_p1 = min(min_p1, projection)
                    max_p1 = max(max_p1, projection)

                for vertex in poly2:
                    projection = np.dot(normal, vertex)
                    min_p2 = min(min_p2, projection)
                    max_p2 = max(max_p2, projection)

                if max_p1 < min_p2 or max_p2 < min_p1:
                    return False  # Separating axis found, no collision

        return True  # No separating axis found, collision exists
    
    def sort_polygons(self):
        sorted_polygons = sorted(self.polygons.keys(), key=lambda idx: self.bounding_boxes[idx][1][0])
        return sorted_polygons

    def detect_all_collisions(self):
        collisions = []
        face_indices = list(self.sort_polygons())
        for i, face1_index in enumerate(face_indices):
            for face2_index in face_indices[i+1:]:
                if self.check_face_collision(face1_index, face2_index):
                    collisions.append((face1_index, face2_index))
        return collisions

    def visualize_polygons(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        for face_index, vertices_2d in self.polygons.items():
            polygon = plt.Polygon(list(vertices_2d.values()), fill=None, edgecolor='black')
            ax.add_patch(polygon)
        ax.autoscale()
        ax.set_aspect('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Unfolded Mesh')
        plt.show()

class Unfolder:
    def __init__(self,mesh):
        self.mesh = mesh 
        self.polygon_2d = Polygon2D()
    
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
        if face_index in self.polygon_2d.polygons:
            return
        face = self.mesh.faces[face_index]
        if parent_index is None:
            # This is the root face. Place it at the origin.
            a, b, c = face.vertex_indices[:3]
            va, vb, vc = self.mesh.vertices[a], self.mesh.vertices[b], self.mesh.vertices[c]
            
            # Place first two vertices along x-axis
            x1 = np.linalg.norm(np.array(vb) - np.array(va))
            vertices_2d = {a: (0, 0), b: (x1, 0)}
            
            vec1 = np.array(vb) - np.array(va)
            vec2 = np.array(vc) - np.array(va)
            x2 = np.dot(vec1, vec2) / np.linalg.norm(vec1)
            y2 = np.sqrt(np.dot(vec2, vec2) - x2*x2)
            vertices_2d[c] = (x2, y2)
        else:
            # This face shares an edge with its parent. Use that information to place it.
            parent_face = self.mesh.faces[parent_index]
            shared_edge = list(set(face.vertex_indices) & set(parent_face.vertex_indices))
            
            if len(shared_edge) != 2:
                raise ValueError(f"Faces {face_index} and {parent_index} don't share an edge")
            
            # Get 2D coordinates of shared edge in parent face
            parent_2d = self.polygon_2d.polygons[parent_index]
            edge_2d = [parent_2d[v] for v in shared_edge]
            
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

            vertices_2d = {shared_edge[0]: edge_2d[0], 
                           shared_edge[1]: edge_2d[1], 
                           new_vertex: tuple(new_point)}
            
        self.polygon_2d.add_polygon(face_index, vertices_2d)
        # Recursively unfold children
        
        for child in tree.get_children(face_index):
            if child != parent_index:
                self.unfold_face(child, face_index, tree)


    def steepest_edge_unfolder(self):
        # Initialize empty list T
        cut_tree = []
        # Generate random normalized 3D vector c
        c = np.array([random.uniform(-1, 1) for _ in range(3)])
        c = c / np.linalg.norm(c)  # Normalize the vector
        # Find the top vertex with respect to c
        top_vertex_index = max(range(len(self.mesh.vertices)), key=lambda i: np.dot(self.mesh.vertices[i], c))
        # Process each vertex except the top
        for i, vertex in enumerate(self.mesh.vertices):
            if i == top_vertex_index:
                continue
            # Find the edge with highest dot product with c
            max_dot_product = float('-inf')
            steepest_edge = None
            for face in self.mesh.faces:
                if i in face.vertex_indices:
                    for v in face.vertex_indices:
                        if v != i:
                            edge = np.array(self.mesh.vertices[v]) - np.array(vertex)
                            edge_normalized = edge / np.linalg.norm(edge)
                            dot_product = np.dot(edge_normalized, c)
                            if dot_product > max_dot_product:
                                max_dot_product = dot_product
                                steepest_edge = set([i, v])
            if steepest_edge:
                cut_tree.append(steepest_edge)
        return self.cut_tree_to_unfold_tree(cut_tree)

    def cut_tree_to_unfold_tree(self, cut_edges):
        # Helper function to get neighboring faces
        def get_neighbors(face_idx):
            neighbors = []
            #shared_edge = list(set(face.vertex_indices) & set(parent_face.vertex_indices))
            for other_face_idx, other_face in enumerate(self.mesh.faces):
                shared_edge = set(self.mesh.faces[face_idx].vertex_indices) & set(other_face.vertex_indices)
                if (len(shared_edge) == 2 and shared_edge not in cut_edges):
                    neighbors.append(other_face_idx)
            return list(set(neighbors))

        # Create adjacency list
        adjacency_list = {i: get_neighbors(i) for i in range(len(self.mesh.faces))}

        # Initialize tree and tracking variables
        face_tree = Tree()
        inserted_faces = set()
        lost_nodes = set()

        # Start with a random root face
        root_face = random.choice(range(len(self.mesh.faces)))
        face_tree.root = root_face
        inserted_faces.add(root_face)

        # Queue for BFS
        queue = [root_face]

        # Main tree construction
        while queue:
            current_face = queue.pop(0)
            for neighbor in adjacency_list[current_face]:
                if neighbor not in inserted_faces:
                    #face_tree.create_node(str(neighbor), neighbor, parent=current_face)
                    face_tree.add_edge(current_face, neighbor, 1)
                    inserted_faces.add(neighbor)
                    queue.append(neighbor)

        # Identify lost nodes
        lost_nodes = set(range(len(self.mesh.faces))) - inserted_faces

        # Connect lost nodes
        for lost_node in lost_nodes:
            possible_parents = [node for node in adjacency_list[lost_node] if node in inserted_faces]
            if possible_parents:
                parent = random.choice(possible_parents)
                #face_tree.create_node(str(lost_node), lost_node, parent=parent)
                face_tree.add_edge(parent, lost_node, 1)
                inserted_faces.add(lost_node)

        return face_tree
    
