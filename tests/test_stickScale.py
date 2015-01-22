import pcl
import unittest
import numpy as np
from patty.registration.stickScale import getStickScale
from numpy.testing import assert_almost_equal

class TestStickScale(unittest.TestCase):
    def test_centeredLineOnXaxis_correctResult(self):
        #Arrange
        pc = pcl.load('tests/testdata/teststicks.ply')
        ar = np.asarray(pc)
        expectedMedian = 4.6638692276102685
        expectedMin = 4.1675386526169405
        expectedMax = 4.6813818282215154
        
        #Act
        median, mini, maxi = getStickScale(ar)
        
        #Assert
        assert_almost_equal(median, expectedMedian)
        assert_almost_equal(mini, expectedMin)
        assert_almost_equal(maxi, expectedMax)
     
    
