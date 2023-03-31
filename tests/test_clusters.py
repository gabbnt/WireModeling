import unittest
import sys

sys.path.append("..")
from src.Data import Point,Cloud
from src.Clusters import Clusters


class TestClusters(unittest.TestCase):

    def setUp(self):
        self.points = [Point((1.,2.,3.)),Point((2.,4.,1.)),Point((3.,6.,0.)),
                       Point((4.,6.,1.)),Point((5.,10.,2.)),Point((1.,2.,7.)),
                       Point((2.,4.,5.)),Point((3.,6.,7.)),Point((4.,8.,6.)),
                       Point((5.,10.,7.))]
        self.length = 10
        self.cluster=Clusters([(1.,2.,3.),(2.,5.,1.),(3.,6.,0.),(4.,6.,1.),(5.,10.,2.),
                               (1.,2.,7.),(2.,4.,5.),(3.,6.,7.),(4.,8.,6.),(5.,10.,7.)])

    def test_get_point(self):
        self.assertTrue(self.cluster.get_point(1).coords[2]>=0.)

    def test_add_point(self):
        self.cluster.add_point((0.,0.,1.))
        self.assertTrue(self.cluster.length>=0)

    def test_remove_point(self):
        self.cluster.remove_point(1)
        self.assertTrue(self.cluster.length>=0)
    def test_print(self):
        self.cluster.print()

    def test_planes(self):
        for cluster in self.cluster.what_clusters():
            print(self.cluster.find_2D_plane(cluster))

if __name__ == '__main__':
    unittest.main()
