import patty
from helpers import __makeTriPyramidWithBase__

from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal
import unittest
import logging
import pcl
import numpy as np

logging.basicConfig(level=logging.INFO)


class TestInPoly(unittest.TestCase):

    def setUp(self):
        side = 10
        delta = 0.05
        offset = [-5, -5, 0]
        points, footprint = __makeTriPyramidWithBase__(side, delta, offset)

        self.pc = pcl.PointCloudXYZRGB(points.astype(np.float32))
        patty.register(self.pc)
        self.footprint = footprint

    def testSynthData(self):
        '''
        Test point cloud / footprint intersection functionality provided
        by patty.registration.registration.intersect_polgyon2d()
        '''
        pcIn = patty.registration.intersect_polgyon2d(self.pc, self.footprint)

        assert_true(len(self.pc) >= len(pcIn))
        assert_true(len(pcIn) > 0)

        # Asset bounding boxes match in X and Y dimension
        #  -- Because footprints do not have a Z dimension (flat as a pancake!)
        bbPC = patty.BoundingBox(points=np.asarray(pcIn))
        bbFP = patty.BoundingBox(points=self.footprint)
        assert_array_almost_equal(
            bbPC.center[:2], bbFP.center[:2], 1, "Center mismatch")
        assert_array_almost_equal(
            bbPC.min[:2], bbFP.min[:2], 1, "Lower boundary mismatch")
        assert_array_almost_equal(
            bbPC.max[:2], bbFP.max[:2], 1, "Upper boundary mismatch")
        assert_array_almost_equal(
            bbPC.size[:2], bbFP.size[:2], 1, "Size mismatch")
