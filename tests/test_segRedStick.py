import unittest
import numpy as np
from patty.segmentation.segRedStick import getReds
from numpy.testing import assert_almost_equal

class TestSegRedStick(unittest.TestCase):
    def test_centeredLineOnXaxis_correctResult(self):
        #Arrange
        ar = np.asarray([[0,0,0,210,25,30], [0,0,0,0,0,150], [0,0,0,0,150,70]], dtype=np.float32)
        expected = 1

        #Act
        reds = getReds(ar)

        #Assert
        assert_almost_equal(len(reds), expected)

    def test_centeredLineOnXaxisInt_correctResult(self):
        #Arrange
        # Data type of point cloud should not make any difference to the result
        ar = np.asarray([[0,0,0,210,25,30], [0,0,0,0,0,150], [0,0,0,0,150,70]])
        expected = 1

        #Act
        reds = getReds(ar)

        #Assert
        assert_almost_equal(len(reds), expected)
