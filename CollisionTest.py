import unittest
import numpy as np
from Unfolder import Mesh2D, epsilon

class TestMesh2DFacesOverlap(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh2D()

    def test_no_overlap(self):
        self.mesh.add_polygon(0, {1: np.array([0, 0]), 2: np.array([1, 0]), 3: np.array([0, 1])})
        self.mesh.add_polygon(1, {4: np.array([2, 2]), 5: np.array([3, 2]), 6: np.array([2, 3])})
        self.assertFalse(self.mesh.faces_overlap(0, 1))

    def test_shared_edge_no_overlap(self):
        self.mesh.add_polygon(0, {1: np.array([0, 0]), 2: np.array([1, 0]), 3: np.array([0, 1])})
        self.mesh.add_polygon(1, {2: np.array([1, 0]), 3: np.array([0, 1]), 4: np.array([1, 1])})
        self.assertFalse(self.mesh.faces_overlap(0, 1))

    def test_shared_edge_overlap(self):
        self.mesh.add_polygon(0, {1: np.array([0, 0]), 2: np.array([1, 0]), 3: np.array([0, 1])})
        self.mesh.add_polygon(1, {2: np.array([1, 0]), 3: np.array([0, 1]), 4: np.array([0.5, 0.5])})
        self.assertTrue(self.mesh.faces_overlap(0, 1))

    def test_partial_overlap(self):
        self.mesh.add_polygon(0, {1: np.array([0, 0]), 2: np.array([2, 0]), 3: np.array([1, 2])})
        self.mesh.add_polygon(1, {4: np.array([1, 1]), 5: np.array([3, 1]), 6: np.array([2, 3])})
        self.assertTrue(self.mesh.faces_overlap(0, 1))

    def test_complete_overlap(self):
        self.mesh.add_polygon(0, {1: np.array([0, 0]), 2: np.array([1, 0]), 3: np.array([0, 1])})
        self.mesh.add_polygon(1, {4: np.array([0, 0]), 5: np.array([1, 0]), 6: np.array([0, 1])})
        self.assertTrue(self.mesh.faces_overlap(0, 1))

    def test_point_touch(self):
        self.mesh.add_polygon(0, {1: np.array([0, 0]), 2: np.array([1, 0]), 3: np.array([0, 1])})
        self.mesh.add_polygon(1, {3: np.array([0, 1]), 4: np.array([1, 1]), 5: np.array([1, 2])})
        self.assertFalse(self.mesh.faces_overlap(0, 1))

    def test_one_inside_other(self):
        self.mesh.add_polygon(0, {1: np.array([0, 0]), 2: np.array([3, 0]), 3: np.array([1.5, 3])})
        self.mesh.add_polygon(1, {4: np.array([1, 1]), 5: np.array([2, 1]), 6: np.array([1.5, 2])})
        self.assertTrue(self.mesh.faces_overlap(0, 1))

    def test_almost_touching(self):
        self.mesh.add_polygon(0, {1: np.array([0, 0]), 2: np.array([1, 0]), 3: np.array([0, 1])})
        self.mesh.add_polygon(1, {4: np.array([1 + epsilon/2, 0]), 5: np.array([2, 0]), 6: np.array([1, 1])})
        self.assertFalse(self.mesh.faces_overlap(0, 1))

if __name__ == '__main__':
    unittest.main()