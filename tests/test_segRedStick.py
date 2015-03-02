import unittest
import numpy as np
import pcl
from patty.segmentation.segRedStick import getRedMask
from numpy.testing import assert_almost_equal

class TestSegRedStick(unittest.TestCase):
    def test_centeredLineOnXaxis_correctResult(self):
        #Arrange
        ar = np.asarray([[0,0,0,210,25,30], [0,0,0,0,0,150], [0,0,0,0,150,70]], dtype=np.float32)
        pc = pcl.PointCloudXYZRGB(ar)
        
        expected = 1

        #Act
        reds = getRedMask(pc)

        #Assert
        assert_almost_equal(sum(reds), expected)
