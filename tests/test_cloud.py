import unittest
import sys

sys.path.append("..")
from src.Data import Point,Cloud

class TestCloud(unittest.TestCase):

    def setUp(self):
        self.points = [Point((0.,0.,1.)),Point((0.,0.,0.)),Point((1.,1.,1.))]
        self.length = 3
        self.cloud=Cloud([(0.,0.,1.),(0.,0.,0.),(1.,1.,1.)])

    def test_get_point(self):
        self.assertTrue(self.cloud.get_point(1).coords[1]==0.)

    def test_add_point(self):
        self.cloud.add_point((0.,0.,1.))
        self.assertTrue(self.cloud.length==4)

    def test_remove_point(self):
        self.cloud.remove_point(0)
        self.assertTrue(self.cloud.length==2)


if __name__ == '__main__':
    unittest.main()