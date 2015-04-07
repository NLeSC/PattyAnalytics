import os.path
import shutil
from tempfile import mkdtemp

import numpy as np
import pcl

from patty import utils
from patty.registration import (downsample_voxel)
from patty.utils import clone

from helpers import make_tri_pyramid_with_base
from nose.tools import (assert_equal, assert_greater, assert_less,
                        assert_true, assert_false)
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_array_less)
from sklearn.utils.extmath import cartesian
import unittest



class TestRegistrationPipeline(unittest.TestCase):

    def setUp(self):
        self.useLocal = False

        if self.useLocal:
            self.tempdir = tempdir = '.'
        else:
            self.tempdir = tempdir = mkdtemp(prefix='patty-analytics')

        self.drivemapLas = os.path.join(tempdir, 'testDriveMap.las')
        self.sourcelas = os.path.join(tempdir, 'testSource.las')
        self.footprint_csv = os.path.join(tempdir, 'testFootprint.csv')
        self.foutlas = os.path.join(tempdir, 'testOutput.las')

        self.min = -10
        self.max = 10
        self.num_rows = 1000

        # Create plane with a pyramid
        dm_pct = 0.5
        dm_rows = np.round(self.num_rows * dm_pct)
        dm_min = self.min * dm_pct
        dm_max = self.max * dm_pct

        delta = dm_max / dm_rows
        shape_side = dm_max - dm_min

        dm_offset = [0, 0, 0]
        self.dense_obj_offset = [3, 2, -(1 + shape_side / 2)]

        # make drivemap
        plane_row = np.linspace(
            start=self.min, stop=self.max, num=self.num_rows)
        plane_points = cartesian((plane_row, plane_row, [0]))

        shape_points, footprint = make_tri_pyramid_with_base(
            shape_side, delta, dm_offset)
        np.savetxt(self.footprint_csv, footprint, fmt='%.3f', delimiter=',')

        dm_points = np.vstack([plane_points, shape_points])
        plane_grid = np.zeros((dm_points.shape[0], 6), dtype=np.float32)
        plane_grid[:, 0:3] = dm_points

        self.drivemap_pc = pcl.PointCloudXYZRGB(plane_grid)
        self.drivemap_pc = downsample_voxel(self.drivemap_pc,
                                            voxel_size=delta * 20)
        # utils.set_registration(self.drivemap_pc)
        utils.save(self.drivemap_pc, self.drivemapLas)

        # Create a simple pyramid
        dense_grid = np.zeros((shape_points.shape[0], 6), dtype=np.float32)
        dense_grid[:, 0:3] = shape_points + self.dense_obj_offset

        self.source_pc = pcl.PointCloudXYZRGB(dense_grid)
        self.source_pc = downsample_voxel(self.source_pc, voxel_size=delta * 5)
        utils.save(self.source_pc, self.sourcelas)

    def tearDown(self):
        if not self.useLocal:
            shutil.rmtree(self.tempdir, ignore_errors=True)

    def test_pipeline(self):
        pass
        # # TODO: should just use shutil to run the registration.py script, and
        # # load the result

        # os.system( './scripts/registration.py -u testupfile.json'
        #     " " + self.sourcelas +
        #     " " + self.drivemapLas +
        #     " " + self.footprint_csv +
        #     " " + self.foutlas )

        # goal   = utils.load( self.sourcelas)
        # actual = np.asarray( start )

        # result = utils.load( self.foutlas )
        # target = np.asarray( result )

        # array_in_margin(target.min(axis=0), actual.min(axis=0), [1, 1, 1],
        #                 "Lower bound of registered cloud does not"
        #                 " match expectation")
        # array_in_margin(target.max(axis=0), actual.max(axis=0), [2.5, 5.5, 2],
        #                 "Upper bound of registered cloud does not"
        #                 " match expectation")
        # array_in_margin(target.mean(axis=0), actual.mean(axis=0), [1, 1, 1],
        #                 "Middle point of registered cloud does not"
        #                 " match expectation")

