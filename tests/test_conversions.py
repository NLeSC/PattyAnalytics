import unittest
import pcl
import os
import os.path
import numpy as np
from patty_registration import conversions
from numpy.testing import assert_array_equal, assert_array_almost_equal
from tempfile import mktemp

# class TestReadLas(unittest.TestCase):
#     # def testReadRome(self):
#     #     fname = 'tests/Rome-000062.las'
#     def testRead162(self):
#         fname = 'data/footprints/162.las'
#         assert os.path.exists(fname)
#         pc, offset, header = conversions.loadLas(fname)
#         print offset
#
#         xyz_array = np.asarray(pc)
#         assert_array_almost_equal(xyz_array.min(axis=0), -xyz_array.max(axis=0))
#
#         rgb_array = pc.to_array()[:,3:6]
#         assert np.all(rgb_array.min(axis=0) >= np.zeros(3))
#         assert np.all(rgb_array.max(axis=0) > np.zeros(3))
#         assert np.all(rgb_array.max(axis=0) <= np.array([255.0, 255.0, 255.0]))
#
class TestWriteLas(unittest.TestCase):
    # def testReadRome(self):
    #     fname = 'tests/Rome-000062.las'
    def testWrite162(self):
        fname = 'data/footprints/162.las'
        assert os.path.exists(fname)
        pc, scale = conversions.loadLas(fname)
        print pc.offset
        
        wfname = mktemp()
        conversions.writeLas(wfname, pc, scale)
        
        assert os.path.exists(wfname)
        pc_new, scale_new = conversions.loadLas(wfname)
        os.remove(wfname)
        
        assert_array_almost_equal(np.asarray(pc), np.asarray(pc_new))
        assert_array_almost_equal(pc.offset, pc_new.offset)
