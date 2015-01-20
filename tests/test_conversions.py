import unittest
import pcl
import os
import os.path
import numpy as np
from patty_registration import conversions
from numpy.testing import assert_array_equal, assert_array_almost_equal
from tempfile import mktemp

class TestReadLas(unittest.TestCase):
    # def testReadRome(self):
    #     fname = 'tests/Rome-000062.las'
    def testRead162(self):
        fname = 'data/footprints/162.las'
        assert os.path.exists(fname)
        pc, offset, header = conversions.loadLas(fname)
        print offset

        xyz_array = pc.to_array()[:,0:3]
        assert_array_almost_equal(xyz_array.min(axis=0), -xyz_array.max(axis=0))

        rgb_array = pc.to_array()[:,3:6]
        assert np.all(rgb_array.min(axis=0) >= np.zeros(3))
        assert np.all(rgb_array.max(axis=0) > np.zeros(3))
        assert np.all(rgb_array.max(axis=0) <= np.array([255.0, 255.0, 255.0]))
#
# class TestWriteLas(unittest.TestCase):
#     # def testReadRome(self):
#     #     fname = 'tests/Rome-000062.las'
#     def testWrite162(self):
#         fname = 'data/footprints/162.las'
#         assert os.path.exists(fname)
#         pc, offset, header = conversions.loadLas(fname)
#         print offset
#
#         wfname = mktemp()
#         conversions.writeLas(wfname, pc, offset, header)
#
#         assert os.path.exists(wfname)
#         pc_new, offset_new, header_new = conversions.loadLas(wfname)
#         os.remove(wfname)
#
#         assert_array_almost_equal(pc.to_array(), pc_new.to_array())
#         assert_array_almost_equal(offset, offset_new)
#         assert_array_almost_equal(header.min, header_new.min)
#         assert_array_almost_equal(header.max, header_new.max)
#         assert_array_equal(header.offset, header_new.offset)
#         assert_array_equal(header.scale, header_new.scale)
