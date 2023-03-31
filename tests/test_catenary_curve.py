import unittest
import sys
import numpy as np

sys.path.append("..")
from src.models import _3D_CatenaryCurve

class TestCloud(unittest.TestCase):

    def setUp(self):
        basis=np.identity(3)
        self.cat=_3D_CatenaryCurve(basis[0],basis[1],basis[2],
                                   0,1,0,(0,0,1))

    def test_generate_points(self):
        x,y,z=self.cat.generate_points(100)
        self.assertTrue(len(x)==len(y))
        self.assertTrue(len(x)==len(z))
        self.assertTrue(len(x)==100)


if __name__ == '__main__':
    unittest.main()