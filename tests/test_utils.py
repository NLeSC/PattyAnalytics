import numpy as np
from pcl import PointCloud

from patty.registration import downsample_random

from nose.tools import assert_equal, assert_raises


def test_downsample_random():
    pc = PointCloud(10)
    a = np.asarray(pc)
    a[:] = np.random.randn(*a.shape)

    assert_raises(ValueError, downsample_random, pc, 0)
    assert_raises(ValueError, downsample_random, pc, 2)

    assert_equal(len(downsample_random(pc, .39)), 4)
