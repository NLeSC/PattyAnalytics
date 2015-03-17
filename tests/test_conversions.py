import unittest
import os
import os.path
import numpy as np
import pcl
from patty import conversions
from tempfile import mktemp

from nose.tools import assert_equal, assert_true
from numpy.testing import assert_array_equal, assert_array_almost_equal


class TestReadLas(unittest.TestCase):

    def testRead20(self):
        fname = 'data/footprints/20.las'
        assert os.path.exists(fname), "test file %s does not exist" % fname
        pc = conversions.loadLas(fname)

        xyz_array = np.asarray(pc)
        minimum = xyz_array.min(axis=0)
        assert_array_almost_equal(minimum, 0,
                                  "bounding box not centered around zero")

        rgb_array = pc.to_array()[:, 3:6]
        assert_true(np.all(rgb_array.min(axis=0) >= 0),
                    "some colors are negative")
        assert_true(np.all(rgb_array.max(axis=0) > 0), "all colors are zero")
        assert_true(np.all(rgb_array.max(axis=0) <= 255),
                    "some colors are larger than 255.0")


WKT = """GEOGCS["WGS 84",
                DATUM["WGS_1984",
                      SPHEROID["WGS 84", 6378137, 298.257223563,
                               AUTHORITY["EPSG", "7030"]],
                               TOWGS84[0,0,0,0,0,0,0],
                               AUTHORITY["EPSG","6326"]],
                PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
                UNIT["DMSH",0.0174532925199433,
                     AUTHORITY["EPSG","9108"]],
                AXIS["Lat",NORTH],
                AXIS["Long",EAST],
                AUTHORITY["EPSG","4326"]]"""


class TestWriteLas(unittest.TestCase):

    def setUp(self):
        data = np.array(
            [[1, 1, 1, 1, 120, 13], [3, 3, 3, 1, 2, 3]], dtype=np.float32)
        self.pc = pcl.PointCloudXYZRGB(data)
        conversions.register(self.pc, offset=[2., 1., 15.], crs_wkt=WKT,
                             precision=[0.1, 0.1, 0.1])

    def testWriteRead(self):
        """Read-then-write should be idempotent."""
        wfname = mktemp()
        header = conversions.makeLasHeader(self.pc)
        assert_array_almost_equal(header.min, [1 + 2, 1 + 1, 1 + 15])
        assert_array_almost_equal(
            np.asarray(self.pc).min(axis=0) + self.pc.offset, header.min)
        assert_array_almost_equal(header.max, [3 + 2, 3 + 1, 3 + 15])
        assert_array_almost_equal(
            np.asarray(self.pc).max(axis=0) + self.pc.offset, header.max)
        conversions.writeLas(wfname, self.pc, header)

        assert_true(os.path.exists(wfname), "temporary test file not written")
        pc_new = conversions.loadLas(wfname)
        os.remove(wfname)

        assert_array_almost_equal(
            np.asarray(pc_new).min(axis=0) + pc_new.offset, header.min)
        assert_array_almost_equal(
            np.asarray(pc_new).max(axis=0) + pc_new.offset, header.max)
        assert_array_almost_equal(np.asarray(self.pc) + self.pc.offset,
                                  np.asarray(pc_new) + pc_new.offset,
                                  'points differ')
        assert_array_equal(self.pc.to_array()[:, 3:6],
                           pc_new.to_array()[:, 3:6],
                           'colors differ')
        assert_equal(pc_new.crs_wkt, WKT,
                     "Projections well-known text is not maintained")


class TestExtractMask(unittest.TestCase):

    def setUp(self):
        data = np.array(
            [[1, 1, 1, 1, 120, 13], [3, 3, 3, 1, 2, 3]], dtype=np.float32)
        self.pc = pcl.PointCloudXYZRGB(data)
        self.offset = [2., 1., 15.]
        conversions.register(self.pc, offset=self.offset)

    def testExtractMask(self):
        assert_array_equal(self.pc[0], [1., 1., 1., 1., 120., 13.],
                           "data not represented in pointcloud")
        pc_first = conversions.extract_mask(self.pc, [True, False])
        self.assertEquals(pc_first.size, 1)
        assert_array_equal(pc_first[0], [1., 1., 1., 1., 120., 13.],
                           "original point modified")
        self.assertEquals(self.pc.size, 2, "original pointcloud modified")
        assert_array_equal(pc_first.offset, self.offset)

        pc_second = conversions.extract_mask(self.pc, [False, True])
        assert_array_equal(pc_second[0], [3., 3., 3., 1., 2., 3.])

        pc_empty = conversions.extract_mask(self.pc, [False, False])
        self.assertEquals(pc_empty.size, 0)
