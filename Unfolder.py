import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from Tree import Tree
import heapq
from Mesh import Mesh
import random

class Mesh2D:
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
    
    def visualize(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        for _, vertices_2d in self.polygons.items():
            polygon = plt.Polygon(list(vertices_2d.values()), fill=None, edgecolor='black')
            ax.add_patch(polygon)
            # centroid = np.mean(vertices_2d, axis=0)
            # ax.text(centroid[0], centroid[1], str(face_index), ha='center', va='center')

        ax.autoscale()
        ax.set_aspect('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Unfolded Mesh')
        plt.show()

    def count_collisions(self):
        collision_count = 0
        for i in range(len(self.polygons)):
            for j in range(i + 1, len(self.polygons)):
                if(self.faces_overlap(i, j)):
                    collision_count += 1            
        return collision_count
    
    def orientation(self, p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-8:  
            return 0  # Collinear
        return 1 if val > 0 else -1

    def on_segment(self, p, q, r):
        return (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and
                min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))

    def line_intersect(self, p1, q1, p2, q2):
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)

        # General case
        if o1 != o2 and o3 != o4:
            return True
        # Special Cases
        if o1 == 0 and self.on_segment(p1, q1, p2): return True
        if o2 == 0 and self.on_segment(p1, q1, q2): return True
        if o3 == 0 and self.on_segment(p2, q2, p1): return True
        if o4 == 0 and self.on_segment(p2, q2, q1): return True

        return False

    def point_inside_triangle(self, p, triangle):
        a, b, c = triangle
        area = abs(self.orientation(a, b, c))
        area1 = abs(self.orientation(p, b, c))
        area2 = abs(self.orientation(a, p, c))
        area3 = abs(self.orientation(a, b, p))
        return abs(area - (area1 + area2 + area3)) < 1e-5
        
    def bounding_boxes_intersect(self, face_index1, face_index2):
            (min1_x, min1_y), (max1_x, max1_y) = self.bounding_boxes[face_index1]
            (min2_x, min2_y), (max2_x, max2_y) = self.bounding_boxes[face_index2]
            
            return (min1_x <= max2_x and max1_x >= min2_x and
                    min1_y <= max2_y and max1_y >= min2_y)

    def shared_edge_overlap(self, face_index1, face_index2, shared_vertices):
        # Get the shared edge vertices
        v1, v2 = shared_vertices

        # Get the non-shared vertex for each triangle
        p1 = list(set(self.polygons[face_index1].keys()) - set(shared_vertices))[0]
        p2 = list(set(self.polygons[face_index2].keys()) - set(shared_vertices))[0]

        # Check if the line segment (p1, p2) intersects with the shared edge (v1, v2) -
        # if they do -  each third point is on the other side of the shared edge - the faces do not collide
        # if they dont -  both third points are on the same side of the shared edge - the faces collide
        return not (self.line_intersect(self.polygons[face_index1][p1], self.polygons[face_index2][p2],
                                   self.polygons[face_index1][v1], self.polygons[face_index1][v2]))

        # # Get the shared edge vertices
        # v1, v2 = shared_vertices
        
        # # Get the non-shared vertex for each triangle
        # p1 = list(set(self.polygons[face_index1].keys()) - set(shared_vertices))[0]
        # p2 = list(set(self.polygons[face_index2].keys()) - set(shared_vertices))[0]
        
        # # Get 2D coordinates
        # v1_coord = self.polygons[face_index1][v1]
        # v2_coord = self.polygons[face_index1][v2]
        # p1_coord = self.polygons[face_index1][p1]
        # p2_coord = self.polygons[face_index2][p2]
        
        # # Use the orientation function to determine if p1 and p2 are on the same side of the edge
        # o1 = self.orientation(v1_coord, v2_coord, p1_coord)
        # o2 = self.orientation(v1_coord, v2_coord, p2_coord)
        
        # # If orient1 and orient2 have the same sign, the points are on the same side
        # return o1 * o2 > 0

    def faces_overlap(self, face_index1, face_index2):
        if(not self.bounding_boxes_intersect(face_index1, face_index2)): return False

        # Check if triangles share an edge
        shared_vertices_3d = (set(self.polygons[face_index1].keys()) & set(self.polygons[face_index2].keys()))
        if(len(shared_vertices_3d) == 2):
            return self.shared_edge_overlap(face_index1, face_index2, shared_vertices_3d)
        
        #faces completly overlap
        if(len(shared_vertices_3d) == 3):
            return True
        
        face1_2d = list(self.polygons[face_index1].values())
        face2_2d = list(self.polygons[face_index2].values())
        for i in range(3):
            for j in range(3):
                if self.line_intersect(face1_2d[i], face1_2d[(i+1)%3], face2_2d[j], face2_2d[(j+1)%3]):
                    return True

        #check if one triangle is completely in another 
        face1_in_face2 = all(self.point_inside_triangle(point, face2_2d) for point in face1_2d)
        face2_in_face1 = all(self.point_inside_triangle(point, face1_2d) for point in face2_2d)

        return face1_in_face2 or face2_in_face1

class Unfolder:
    def __init__(self,mesh):
        self.mesh = mesh
        self.mesh_2d = Mesh2D()
    
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
        if face_index in self.mesh_2d.polygons:
            return
        face = self.mesh.faces[face_index]
        if parent_index is None:
            # This is the root face. Place it at the origin.
            a, b, c = face.vertex_indices[:3]
            va, vb, vc = self.mesh.vertices[a], self.mesh.vertices[b], self.mesh.vertices[c]
            
            # Place first two vertices along x-axis and calculate position of third vertex
            x1 = np.linalg.norm(np.array(vb) - np.array(va))
            vec1 = np.array(vb) - np.array(va)
            vec2 = np.array(vc) - np.array(va)
            x2 = np.dot(vec1, vec2) / np.linalg.norm(vec1)
            y2 = np.sqrt(np.dot(vec2, vec2) - x2*x2)
            self.mesh_2d.add_polygon(face_index, {a: (0, 0), b: (x1, 0), c: (x2, y2)})

        else:
            # This face shares an edge with its parent. Use that information to place it.
            parent_face = self.mesh.faces[parent_index]
            shared_edge = list(set(face.vertex_indices) & set(parent_face.vertex_indices))
            
            if len(shared_edge) != 2:
                raise ValueError(f"Faces {face_index} and {parent_index} don't share an edge")
            
            # Get 2D coordinates of shared edge in parent face
            parent_2d = self.mesh_2d.polygons[parent_index]
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
            
            # Store the 2D coordinates
            self.mesh_2d.add_polygon(face_index, {shared_edge[0]: edge_2d[0], 
                                                  shared_edge[1]: edge_2d[1], new_vertex: tuple(new_point)})
            
            if self.mesh_2d.faces_overlap(face_index, parent_index):
                new_point = np.array(edge_2d[0]) + (proj/edge_len)*edge_2d_vec - height*normal
                self.mesh_2d.add_polygon(face_index, {shared_edge[0]: edge_2d[0], 
                                            shared_edge[1]: edge_2d[1], new_vertex: tuple(new_point)})

        # Recursively unfold children
        for child in tree.get_children(face_index):
            if child != parent_index:
                self.unfold_face(child, face_index, tree)

    def steepest_edge_unfolder(self):
        ## Initialize empty list T
        cut_tree = []

        ## Generate random normalized 3D vector c
        c = np.array([random.uniform(-1, 1) for _ in range(3)])
        c = c / np.linalg.norm(c)  # Normalize the vector

        ## Find the top vertex with respect to c
        top_vertex_index = max(range(len(self.mesh.vertices)), key=lambda i: np.dot(self.mesh.vertices[i], c))

        ## Process each vertex except the top
        for i, vertex in enumerate(self.mesh.vertices):
            if i == top_vertex_index:
                continue

            ## Find the edge with highest dot product with c
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
