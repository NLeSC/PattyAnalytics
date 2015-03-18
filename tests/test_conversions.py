import unittest
import os
import os.path
import numpy as np
import pcl
from patty import conversions
from tempfile import mktemp

from nose.tools import assert_equal, assert_true
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_read_las():
    fname = 'data/footprints/20.las'
    pc = conversions.load_las(fname)

    xyz_array = np.asarray(pc)
    minimum = xyz_array.min(axis=0)
    assert_array_almost_equal(minimum, 0,
                              err_msg="bounding box not centered around zero")

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

    def get_header(self):
        return conversions.make_las_header(self.pc)

    def test_header(self):
        header = self.get_header()
        assert_array_almost_equal(header.max, [3 + 2, 3 + 1, 3 + 15])
        assert_array_almost_equal(header.min, [1 + 2, 1 + 1, 1 + 15])
        self.pointcloud_header(self.pc, header)

    def pointcloud_header(self, cloud, header):
        assert_array_almost_equal(
            np.asarray(cloud).min(axis=0) + cloud.offset, header.min)
        assert_array_almost_equal(
            np.asarray(cloud).max(axis=0) + cloud.offset, header.max)

    def test_write_read(self):
        """Write-then-read should be idempotent."""
        wfname = mktemp()
        header = self.get_header()
        conversions.write_las(wfname, self.pc, header=header)

        assert_true(os.path.exists(wfname), "temporary test file not written")
        pc_new = conversions.load_las(wfname)
        os.remove(wfname)
        
        self.pointcloud_header(pc_new, header)
        assert_array_almost_equal(np.asarray(self.pc) + self.pc.offset,
                                  np.asarray(pc_new) + pc_new.offset,
                                  err_msg='points differ')
        assert_array_equal(self.pc.to_array()[:, 3:6],
                           pc_new.to_array()[:, 3:6],
                           err_msg='colors differ')
        assert_equal(pc_new.crs_wkt, WKT,
                     "Projections well-known text is not maintained")


class TestExtractMask(unittest.TestCase):
    def setUp(self):
        data = np.array(
            [[1, 1, 1, 1, 120, 13], [3, 3, 3, 1, 2, 3]], dtype=np.float32)
        self.pc = pcl.PointCloudXYZRGB(data)
        self.offset = [2., 1., 15.]
        conversions.register(self.pc, offset=self.offset)

    def test_extract_mask(self):
        assert_array_equal(self.pc[0], [1., 1., 1., 1., 120., 13.],
                           err_msg="data not represented in pointcloud")
        pc_first = conversions.extract_mask(self.pc, [True, False])
        self.assertEquals(pc_first.size, 1)
        assert_array_equal(pc_first[0], [1., 1., 1., 1., 120., 13.],
                           err_msg="original point modified")
        self.assertEquals(self.pc.size, 2, "original pointcloud modified")
        assert_array_equal(pc_first.offset, self.offset)

        pc_second = conversions.extract_mask(self.pc, [False, True])
        assert_array_equal(pc_second[0], [3., 3., 3., 1., 2., 3.])

        pc_empty = conversions.extract_mask(self.pc, [False, False])
        self.assertEquals(pc_empty.size, 0)
