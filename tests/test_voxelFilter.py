import logging
import pcl
import numpy as np
import patty
from helpers import make_tri_pyramid_with_base

from nose.tools import assert_greater

logging.basicConfig(level=logging.INFO)


def test_filter():
    '''
    Test Voxel Grid Filter functionality
    '''
    side = 10
    delta = 0.05
    offset = [-5, -5, 0]
    points, footprint = make_tri_pyramid_with_base(side, delta, offset)

    pc = pcl.PointCloudXYZRGB(points.astype(np.float32))
    patty.register(pc)

    # Build Voxel Grid Filter
    vgf = pc.make_voxel_grid_filter()

    # Filter with Voxel size 1
    vgf.set_leaf_size(1, 1, 1)
    pc2 = vgf.filter()

    # Filter with Voxel size 0.1
    vgf.set_leaf_size(0.1, 0.1, 0.1)
    pc3 = vgf.filter()

    # Filter with Voxel size 10
    vgf.set_leaf_size(10, 10, 10)
    pc4 = vgf.filter()

    assert_greater(len(pc), len(pc2))
    assert_greater(len(pc3), len(pc2))
    assert_greater(len(pc), len(pc4))
