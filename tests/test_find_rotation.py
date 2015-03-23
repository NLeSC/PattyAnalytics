import numpy as np
from patty.registration import find_rotation
import pcl

from numpy.testing import assert_almost_equal, assert_equal
from nose.tools import assert_true
import unittest

class TestFindRotation(unittest.TestCase):
    def setUp(self):
        self.rotation = np.array([[ 0.69353487, -0.33259093,  0.63905606],
                                  [ 0.63905606,  0.69353487, -0.33259093],
                                  [-0.33259093,  0.63905606,  0.69353487]] )

        data = np.random.random_sample( [200,3] ) * [5.0, 10.0, 1.0] - [2.5, 5, 0.5]
        self.pc1 = pcl.PointCloudXYZRGB( data.astype(np.float32) )

        self.pc2 = pcl.PointCloudXYZRGB( data.astype(np.float32) )
        self.pc2.rotate( self.rotation )

    def test_perfectly_aligned(self):
        rotation = find_rotation( self.pc1, self.pc1 ) 
        assert_almost_equal( rotation, np.eye(3) )

    def test_rotated(self):
        rotation = find_rotation( self.pc1, self.pc2 )
        w1,v1 = np.linalg.eig( rotation[0:2,0:2] )
        w2,v2 = np.linalg.eig( self.rotation[0:2,0:2] )
        # assert_almost_equal( rotation, self.rotation )
