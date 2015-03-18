import numpy as np
from pcl import PointCloud

from patty.registration import downsample

from nose.tools import assert_equal, assert_raises


def test_downsample():
    pc = PointCloud(10)
    a = np.asarray(pc)
    a[:] = np.random.randn(*a.shape)

    assert_raises(ValueError, downsample, pc, 0)
    assert_raises(ValueError, downsample, pc, 2)

    assert_equal(len(downsample(pc, .39)), 4)
