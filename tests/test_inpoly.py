import logging
import pcl
import numpy as np

import patty

from helpers import make_tri_pyramid_with_base

from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal

logging.basicConfig(level=logging.INFO)


def test_in_poly():
    '''
    Test point cloud / footprint intersection functionality provided
    by patty.registration.intersect_polygon2d()
    '''
    side = 10
    delta = 0.05
    offset = [-5, -5, 0]
    points, footprint = make_tri_pyramid_with_base(side, delta, offset)

    pc = pcl.PointCloudXYZRGB(points.astype(np.float32))
    patty.register(pc)

    pcIn = patty.registration.intersect_polygon2d(pc, footprint)

    assert_true(len(pc) >= len(pcIn))
    assert_true(len(pcIn) > 0)

    # Asset bounding boxes match in X and Y dimension
    #  -- Because footprints do not have a Z dimension (flat as a pancake!)
    bbPC = patty.BoundingBox(points=np.asarray(pcIn))
    bbFP = patty.BoundingBox(points=footprint)
    assert_array_almost_equal(
        bbPC.center[:2], bbFP.center[:2], 1, "Center mismatch")
    assert_array_almost_equal(
        bbPC.min[:2], bbFP.min[:2], 1, "Lower boundary mismatch")
    assert_array_almost_equal(
        bbPC.max[:2], bbFP.max[:2], 1, "Upper boundary mismatch")
    assert_array_almost_equal(
        bbPC.size[:2], bbFP.size[:2], 1, "Size mismatch")
