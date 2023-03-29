import unittest
import sys
import io
import numpy as np

sys.path.append("..")
from src.Data import Point

class TestPoint(unittest.TestCase):

    def setUp(self):
        self.coords = (1., 0., 0.)
        self.point = Point(self.coords)

    def test_init(self):
        self.assertTrue(np.all(np.array(self.point.coords)== np.array(self.coords)))
        self.assertEqual(self.point.dim, len(self.coords))
        self.assertTrue(np.all(np.array(self.point.basis)==np.array(self.point._get_canonical_basis())))

    def test_change_basis(self):
        new_basis = [np.array([0., 0., 1.]), np.array([-1., 0., 0.]), np.array([0., 1., 0.])]
        self.point.change_basis(new_basis)
        expected_coords = np.array([0., -1., 0.])
        for i in range(len(expected_coords)):
            self.assertAlmostEqual(float(self.point.coords[i]),float(expected_coords[i]))
        self.assertEqual(self.point.basis, new_basis)


if __name__ == '__main__':
    unittest.main()