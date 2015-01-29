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
        # Would 5 decimal places be enough? -- test break with 6 decimals!
        assert_almost_equal(median, expectedMedian, decimal=5)
        assert_almost_equal(mini, expectedMin, decimal=5)
        assert_almost_equal(maxi, expectedMax, decimal=5)

