import unittest
import pcl
import os.path
import numpy as np
from patty_registration import conversions, registration
from numpy.testing import assert_array_equal, assert_array_almost_equal


class TestRegistrationScaleOffset(unittest.TestCase):
    def testRegistrationFromFootprint(self):
        fname = 'data/footprints/162.las'
        fp_name = 'data/footprints/162.las_footprint.csv'
        assert os.path.exists(fname)
        assert os.path.exists(fp_name)
        pc, offset, header_offset, scale = conversions.loadLas(fname)
        footprint = conversions.loadCsvPolygon(fp_name)
        
        registered_offset, registered_scale = registration.register_from_footprint(footprint, pc, offset)
        
        print registered_offset, registered_scale

class TestRegistrationSite20(unittest.TestCase):
    def testRegistrationFromFootprint20(self):
        fname = 'data/footprints/162.las'
        fp_name = 'data/footprints/162.las_footprint.csv'
        assert os.path.exists(fname)
        assert os.path.exists(fp_name)
        pc, offset, header_offset, scale = conversions.loadLas(fname)
        footprint = conversions.loadCsvPolygon(fp_name)
        
        registered_offset, registered_scale = registration.register_from_footprint(footprint, pc, offset)
        
        print registered_offset, registered_scale
