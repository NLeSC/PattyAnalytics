import numpy as np
import pcl

from patty import center_boundingbox, register, save, load
from patty.registration import (point_in_polygon2d, downsample_voxel,
                                scale_points, intersect_polgyon2d,
                                get_pointcloud_boundaries)
from patty.utils import BoundingBox
from scripts.registration import registrationPipeline

from helpers import __makeTriPyramidWithBase__
from nose.tools import assert_true
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_array_less)
from sklearn.utils.extmath import cartesian
import unittest


class TestPolygon(unittest.TestCase):

    def setUp(self):
        self.poly = [[0., 0.], [1., 0.], [0.4, 0.4], [1., 1.], [0., 1.]]
        self.points = [[0., 0.], [0.5, 0.2], [1.1, 1.1], [0.2, 1.1]]

    def testInPolygon(self):
        ''' Test whether the point_in_polygon2d behaves as expected. '''
        in_polygon = point_in_polygon2d(self.points, self.poly)
        assert_array_equal(in_polygon, [False, True, False, False],
                           "points expected in polygon not matched")

    def testScalePolygon(self):
        ''' Test whether scaling up the polygon works '''
        newpoly = scale_points(self.poly, 1.3)
        self.assertEqual(len(newpoly), len(self.poly),
                         "number of polygon points is altered when scaling")
        assert_array_equal(self.poly[0], [.0, .0],
                           err_msg="original polygon is altered when scaling")
        assert_array_less(newpoly[0], self.poly[0],
                          err_msg="small polygon points do not shrink when"
                                  " scaling up")
        assert_array_less(self.poly[3], newpoly[3],
                          err_msg="large polygon points do not grow when "
                                  "scaling up")
        in_scaled_polygon = point_in_polygon2d(self.points, newpoly)
        assert_true(np.all(in_scaled_polygon),
                    "not all points are in polygon when scaling up")


class TestCutoutPointCloud(unittest.TestCase):

    def setUp(self):
        self.footprint = [[0., 0.], [1., 0.], [0.4, 0.4], [1., 1.], [0., 1.]]
        self.offset = [-0.01, -0.01, -0.01]
        points = np.array([[0., 0.], [0.5, 0.2], [1.1, 1.1], [0.2, 1.1]])
        data = np.zeros((4, 6), dtype=np.float32)
        data[:, :2] = points
        self.pc = pcl.PointCloudXYZRGB(data)
        register(self.pc, offset=self.offset)

    def testCutOutFromFootprint(self):
        ''' Test whether a cutout from a pointcloud gets the right points '''
        pc_fp = intersect_polgyon2d(self.pc, self.footprint)
        self.assertEqual(pc_fp.size, 1,
                         "number of points expected in polygon not matched")
        assert_array_almost_equal(pc_fp[0], [0.5, 0.2, 0., 0., 0., 0.],
                                  err_msg="point that should be matched was "
                                          "modified")
        assert_array_equal(pc_fp.offset, self.offset,
                           err_msg="offset changed by intersection with "
                                   "polygon")


class TestCenter(unittest.TestCase):

    def setUp(self):
        data = np.array(
            [[1, 1, 1, 1, 1, 1], [3, 3, 3, 1, 1, 1]], dtype=np.float32)
        self.pc = pcl.PointCloudXYZRGB(data)

    def testCenter(self):
        ''' test whether pointcloud can be centered around zero '''
        # Baseline: original center
        bb = BoundingBox(points=np.asarray(self.pc))
        assert_array_equal(bb.center, [2., 2., 2.],
                           "original bounding box center"
                           " is not center of input")

        # New center
        center_boundingbox(self.pc)
        bb_new = BoundingBox(points=np.asarray(self.pc))
        assert_array_equal(bb_new.center, np.zeros(3),
                           err_msg="after centering, BoundingBox center not "
                                   "in origin")
        assert_array_equal(self.pc.offset, bb.center,
                           err_msg="offset of centering operation not equal to"
                                   " original center")
        assert_array_equal(bb.size, bb_new.size,
                           err_msg="bounding box size changed due to "
                                   "translation")


class TestBoundary(unittest.TestCase):

    def setUp(self):
        self.num_rows = 50
        self.max = 0.1
        self.num_points = self.num_rows * self.num_rows
        grid = np.zeros((self.num_points, 6))
        row = np.linspace(start=0.0, stop=self.max, num=self.num_rows)
        grid[:, 0:2] = cartesian((row, row))
        self.pc = pcl.PointCloudXYZRGB(grid.astype(np.float32))
        register(self.pc)
        self.footprint_boundary = np.array(
            [[0.0, 0.0], [0.0, self.max],
             [self.max, self.max], [self.max, 0.0]])

    def testBoundaries(self):
        boundary = get_pointcloud_boundaries(self.pc)
        self.assertEqual(self.pc.size, self.num_points)
        self.assertLess(boundary.size, self.num_points)
        self.assertGreater(boundary.size, 0)

        small_footprint = scale_points(self.footprint_boundary, 0.9)
        large_footprint = scale_points(self.footprint_boundary, 1.1)

        self.assertEqual(np.sum(point_in_polygon2d(boundary,
                                                   small_footprint)), 0)
        self.assertEqual(np.sum(point_in_polygon2d(boundary,
                                                   large_footprint)),
                         boundary.size)
        self.assertGreater(np.sum(point_in_polygon2d(self.pc,
                                                     small_footprint)), 0)
        self.assertEqual(np.sum(point_in_polygon2d(self.pc, large_footprint)),
                         self.pc.size)

    def testBoundariesTooSmallRadius(self):
        boundary = get_pointcloud_boundaries(
            self.pc, search_radius=0.0001, normal_search_radius=0.0001)
        self.assertEqual(boundary.size, 0)


class TestRegistrationPipeline(unittest.TestCase):

    def setUp(self):
        self.drivemapLas = './testDriveMap.las'
        self.sourceLas = './testSource.las'
        self.footprintCsv = './testFootprint.csv'
        self.foutLas = './testOutput.las'

        self.min = -10
        self.max = 10
        self.num_rows = 50

        # Create plane with a pyramid
        dmPct = 0.5
        dmRows = np.round(self.num_rows * dmPct)
        dmMin = self.min * dmPct
        dmMax = self.max * dmPct

        delta = dmMax / dmRows
        shapeSide = dmMax - dmMin

        dmOffset = [0, 0, 0]
        denseObjOffset = [3, 2, -(1 + shapeSide / 2)]

        plane_row = np.linspace(
            start=self.min, stop=self.max, num=self.num_rows)
        planePoints = cartesian((plane_row, plane_row, 0))

        shapePoints, footprint = __makeTriPyramidWithBase__(
            shapeSide, delta, dmOffset)
        allPoints = np.vstack([planePoints, shapePoints])

        planeGrid = np.zeros((allPoints.shape[0], 6))
        planeGrid[:, 0:3] = allPoints

        self.dmPC = pcl.PointCloudXYZRGB(planeGrid.astype(np.float32))
        self.dmPC = downsample_voxel(self.dmPC, voxel_size=delta)
        register(self.dmPC)
        save(self.dmPC, self.drivemapLas)

        # Create a simple pyramid
        denseObjPoints, _ = __makeTriPyramidWithBase__(
            shapeSide, delta / 20, denseObjOffset)
        denseGrid = np.zeros((denseObjPoints.shape[0], 6))
        denseGrid[:, 0:3] = denseObjPoints

        self.sourcePC = pcl.PointCloudXYZRGB(denseGrid.astype(np.float32))
        self.sourcePC = downsample_voxel(self.sourcePC, voxel_size=delta / 4)
        register(self.sourcePC)
        save(self.sourcePC, self.sourceLas)

        np.savetxt(self.footprintCsv, footprint, fmt='%.3f', delimiter=',')

    def testPipeline(self):
        # Register box on surface
        registrationPipeline(self.sourceLas, self.drivemapLas,
                             self.footprintCsv, self.foutLas)
        registeredPC = load(self.foutLas)

        target = np.asarray(self.sourcePC)
        actual = np.asarray(registeredPC)

        assert_array_almost_equal(target.min(axis=0), actual.min(axis=0), 6,
                                  "Lower bound of registered cloud does not"
                                  " match expectation")
        assert_array_almost_equal(target.max(axis=0), actual.max(axis=0), 6,
                                  "Upper bound of registered cloud does not"
                                  " match expectation")
        assert_array_almost_equal(target.mean(axis=0), actual.mean(axis=0), 6,
                                  "Middle point of registered cloud does not"
                                  " match expectation")
