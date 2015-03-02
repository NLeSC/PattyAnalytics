import unittest
import pcl
# import os.path
from patty import conversions
from patty.registration import registration
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy as np
from patty.utils import BoundingBox
from sklearn.utils.extmath import cartesian

class TestPolygon(unittest.TestCase):
    def setUp(self):
        self.poly = [[0., 0.], [1., 0.], [0.4, 0.4], [1., 1.], [0., 1.]]
        self.points = [[0.,0.], [0.5, 0.2], [1.1, 1.1], [0.2, 1.1]]
    
    def testInPolygon(self):
        ''' Test whether the point_in_polygon2d behaves as expected. '''
        in_polygon = registration.point_in_polygon2d(self.points, self.poly)
        assert_array_equal(in_polygon, [False, True, False, False], "points expected in polygon not matched")
        
    def testScalePolygon(self):
        ''' Test whether scaling up the polygon works '''
        newpoly = registration.scale_points(self.poly, 1.3)
        self.assertEqual(len(newpoly), len(self.poly), "number of polygon points is altered after scaling")
        assert_array_equal(self.poly[0], [.0, .0], err_msg="original polygon is altered after scaling")
        assert np.all(newpoly[0] < self.poly[0]), "small polygon points are do not become smaller after scaling up"
        assert np.all(newpoly[3] > self.poly[3]), "large polygon points are do not become larger after scaling up"
        in_scaled_polygon = registration.point_in_polygon2d(self.points, newpoly)
        assert np.all(in_scaled_polygon), "not all points are in polygon after scaling up"
    

class TestCutoutPointCloud(unittest.TestCase):
    def setUp(self):
        self.footprint = [[0., 0.], [1., 0.], [0.4, 0.4], [1., 1.], [0., 1.]]
        self.offset = [-0.01, -0.01, -0.01]
        points = np.array([[0.,0.], [0.5, 0.2], [1.1, 1.1], [0.2, 1.1]])
        data = np.zeros((4,6),dtype=np.float32)
        data[:,:2] = points
        self.pc = pcl.PointCloudXYZRGB(data)
        conversions.register(self.pc, offset=self.offset)
        
    def testCutOutFromFootprint(self):
        ''' Test whether a cutout from a pointcloud gets the right points '''
        pc_fp = registration.intersect_polgyon2d(self.pc, self.footprint)
        self.assertEqual(pc_fp.size, 1, "number of points expected in polygon not matched")
        assert_array_almost_equal(pc_fp[0], [0.5, 0.2, 0., 0., 0., 0.], err_msg="point that should be matched was modified")
        assert_array_equal(pc_fp.offset, self.offset, err_msg="offset changed by intersection with polygon")

class TestCenter(unittest.TestCase):
    def setUp(self):
        data = np.array([[1,1,1,1,1,1],[3,3,3,1,1,1]],dtype=np.float32)
        self.pc = pcl.PointCloudXYZRGB(data)
        
    def testCenter(self):
        ''' test whether pointcloud can be centered around zero '''
        # Baseline: original center
        bb = BoundingBox(points=np.asarray(self.pc))
        assert_array_equal(bb.center,[2.,2.,2.], err_msg="original bounding box center is not center of input")
        
        # New center
        registration.center_boundingbox(self.pc)
        bb_new = BoundingBox(points=np.asarray(self.pc))
        assert_array_equal(bb_new.center, np.zeros(3), err_msg="after centering, bounding box center is not in origin")
        assert_array_equal(self.pc.offset, bb.center, err_msg="offset of centering operation is not equal to original center")
        assert_array_equal(bb.size, bb_new.size, err_msg="bounding box size changed due to translation")

class TestBoundary(unittest.TestCase):
    def setUp(self):
        self.num_rows = 50
        self.max = 0.1
        self.num_points = self.num_rows * self.num_rows
        grid = np.zeros((self.num_points, 6))
        row = np.linspace(start=0.0, stop=self.max, num=self.num_rows)
        grid[:,0:2] = cartesian((row, row))
        self.pc = pcl.PointCloudXYZRGB(grid.astype(np.float32))
        conversions.register(self.pc)
        self.footprint_boundary = np.array([[0.0, 0.0], [0.0, self.max], [self.max, self.max], [self.max, 0.0]])
        
    def testBoundaries(self):
        boundary = registration.get_pointcloud_boundaries(self.pc)
        self.assertEqual(self.pc.size, self.num_points)
        self.assertLess(boundary.size, self.num_points)
        self.assertGreater(boundary.size, 0)
        
        small_footprint = registration.scale_points(self.footprint_boundary, 0.9)
        large_footprint = registration.scale_points(self.footprint_boundary, 1.1)
        
        self.assertEqual(np.sum(registration.point_in_polygon2d(boundary, small_footprint)), 0)
        self.assertEqual(np.sum(registration.point_in_polygon2d(boundary, large_footprint)), boundary.size)
        self.assertGreater(np.sum(registration.point_in_polygon2d(self.pc, small_footprint)), 0)
        self.assertEqual(np.sum(registration.point_in_polygon2d(self.pc, large_footprint)), self.pc.size)
    
    def testBoundariesTooSmallRadius(self):
        boundary = registration.get_pointcloud_boundaries(self.pc, search_radius=0.0001, normal_search_radius=0.0001)
        self.assertEqual(boundary.size, 0)
    
if __name__ == "__main__":
    unittest.main()

# Commented out for slowness
# class TestRegistrationSite20(unittest.TestCase):
#     def testRegistrationFromFootprint(self):
#         fname = 'data/footprints/site20.pcd'
#         frefname = 'data/footprints/20.las'
#         fp_name = 'data/footprints/20.las_footprint.csv'
#         assert os.path.exists(fname)
#         assert os.path.exists(fp_name)
#         assert os.path.exists(frefname)
#         drivemap = conversions.loadLas(frefname)
#         footprint = conversions.loadCsvPolygon(fp_name)
#         # Shift footprint by (-1.579, 0.525) -- value estimated manually
#         footprint[:,0] += -1.579381346780
#         footprint[:,1] += 0.52519696509
#         pointcloud = pcl.load(fname,loadRGB=True)
#         conversions.register(pointcloud)
#         registration.register_from_footprint(pointcloud, footprint)
#         conversions.writeLas(pointcloud, 'tests/20.testscale.las')