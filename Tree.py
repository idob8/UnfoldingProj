import heapq
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
