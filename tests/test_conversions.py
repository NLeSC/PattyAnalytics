import unittest
import pcl
import os.path
import numpy as np
from patty_registration import conversions
from numpy.testing import assert_array_equal, assert_array_almost_equal

class TestReadLas(unittest.TestCase):
    # def testReadRome(self):
    #     fname = 'tests/Rome-000062.las'
    def testRead162(self):
        fname = 'data/footprints/162.las'
        assert os.path.exists(fname)
        pc, offset, scale = conversions.loadLas(fname)
        print offset

        xyz_array = pc.to_array()[:,0:3]
        assert_array_almost_equal(xyz_array.min(axis=0), -xyz_array.max(axis=0))
        
        rgb_array = pc.to_array()[:,3:6]
        assert np.all(rgb_array.min(axis=0) >= np.zeros(3))
        assert np.all(rgb_array.max(axis=0) > np.zeros(3))
        assert np.all(rgb_array.max(axis=0) <= np.array([255.0, 255.0, 255.0]))
                
        assert np.all(offset > np.zeros(3))
        assert np.all(scale > np.zeros(3))