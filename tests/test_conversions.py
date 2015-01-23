import unittest
import os
import os.path
import numpy as np
import pcl
from patty import conversions
from numpy.testing import assert_array_equal, assert_array_almost_equal
from tempfile import mktemp

class TestReadLas(unittest.TestCase):
    def testRead20(self):
        fname = 'data/footprints/20.las'
        assert os.path.exists(fname), "test file %s does not exist" % fname
        pc = conversions.loadLas(fname)

        xyz_array = np.asarray(pc)
        assert_array_almost_equal(xyz_array.min(axis=0), -xyz_array.max(axis=0), err_msg="loaded file bounding box is not centered around zero")

        rgb_array = pc.to_array()[:,3:6]
        assert np.all(rgb_array.min(axis=0) >= np.zeros(3)), "some colors are negative"
        assert np.all(rgb_array.max(axis=0) > np.zeros(3)), "all colors are zero"
        assert np.all(rgb_array.max(axis=0) <= np.array([255.0, 255.0, 255.0])), "some colors are larger than 255.0"

class TestWriteLas(unittest.TestCase):
    def setUp(self):
        data = np.array([[1,1,1,1,120,13],[3,3,3,1,2,3]],dtype=np.float32)
        self.pc = pcl.PointCloudXYZRGB(data)
        self.wkt = ("GEOGCS[\"WGS 84\",  DATUM[    \"WGS_1984\","
            "    SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],"
            "    TOWGS84[0,0,0,0,0,0,0],    AUTHORITY[\"EPSG\",\"6326\"]],"
            "  PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],"
            "  UNIT[\"DMSH\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9108\"]],"
            "  AXIS[\"Lat\",NORTH],  AXIS[\"Long\",EAST],"
            "  AUTHORITY[\"EPSG\",\"4326\"]]")
        conversions.register(self.pc, offset=[2., 1., 15.],crs_wkt=self.wkt)
        
    def testWriteRead(self):
        ''' Test writing a pointcloud to LAS and reading it in to yield the same data '''
        wfname = mktemp()
        conversions.writeLas(wfname, self.pc)
        
        assert os.path.exists(wfname), "temporary test file could not be written"
        pc_new = conversions.loadLas(wfname)
        os.remove(wfname)
        
        assert_array_almost_equal(np.asarray(self.pc) + self.pc.offset, np.asarray(pc_new) + pc_new.offset, err_msg='points differ')
        assert_array_equal(self.pc.to_array()[:,3:6], pc_new.to_array()[:,3:6], err_msg='colors differ')
        assert pc_new.crs_wkt == self.wkt, "Projections well-known text is not maintained"

class TestExtractMask(unittest.TestCase):
    def setUp(self):
        data = np.array([[1,1,1,1,120,13],[3,3,3,1,2,3]],dtype=np.float32)
        self.pc = pcl.PointCloudXYZRGB(data)
        self.offset = [2., 1., 15.]
        conversions.register(self.pc, offset=self.offset)
    
    def testExtractMask(self):
        pc_first = conversions.extract_mask(self.pc, [True, False])
        self.assertEquals(pc_first.size, 1)
        assert_array_equal(pc_first[0], [1., 1., 1., 1., 120., 13.])
        self.assertEquals(self.pc.size, 2)
        assert_array_equal(pc_first.offset, self.offset)
        
        pc_second = conversions.extract_mask(self.pc, [False, True])
        assert_array_equal(pc_second[0], [3.,3.,3.,1.,2.,3.])
        
        pc_empty = conversions.extract_mask(self.pc, [False, False])
        self.assertEquals(pc_empty.size, 0)
