import unittest
import numpy as np
from patty_segmentation.pointCloudMeasurer import PointCloudMeasurer
from numpy.testing import assert_almost_equal





class TestPointCloudMeasurer(unittest.TestCase):
    def test_centeredLineOnXaxis_correctResult(self):
        #Arrange
        line = np.array([[-5.,0,0], [0,0,0], [5.,0,0]])
        expected = 10
        
        #Act
        length = PointCloudMeasurer().measureLength(line)
        
        #Assert
        assert_almost_equal(expected, length)
        
    
    def test_centeredLineOnYaxis_correctResult(self):
        #Arrange
        line = np.array([[0, -5.,0], [0,0,0], [0, 5.,0]])
        expected = 10
        
        #Act
        length = PointCloudMeasurer().measureLength(line)
        
        #Assert
        assert_almost_equal(expected, length)
        
    
