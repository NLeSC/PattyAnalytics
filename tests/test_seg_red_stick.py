import numpy as np
import pcl
from patty.segmentation.segRedStick import get_red_mask
from numpy.testing import assert_almost_equal


def test_centered_line_on_x_axis():
    '''Test get_red_mask function from patty.segmentation.segRedStick'''
    # Arrange
    ar = np.asarray([[0, 0, 0, 210, 25, 30],
                     [0, 0, 0, 0, 0, 150],
                     [0, 0, 0, 0, 150, 70]], dtype=np.float32)
    pc = pcl.PointCloudXYZRGB(ar)

    expected = 1

    # Act
    reds = get_red_mask(pc)

    # Assert
    assert_almost_equal(sum(reds), expected)
