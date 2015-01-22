import unittest
import pcl
import os.path
import numpy as np
from patty.conversions import conversions
from patty.registration import registration
from patty.segmentation import dbscan
from numpy.testing import assert_array_equal, assert_array_almost_equal


# class TestRegistrationScaleOffset(unittest.TestCase):
#     def testRegistrationFromFootprint(self):
#         fname = 'data/footprints/162.las'
#         fp_name = 'data/footprints/162.las_footprint.csv'
#         assert os.path.exists(fname)
#         assert os.path.exists(fp_name)
#         pc, scale = conversions.loadLas(fname)
#         footprint = conversions.loadCsvPolygon(fp_name)
#
#         registered_offset, registered_scale = registration.register_from_footprint(footprint, pc)
#
#         print registered_offset, registered_scale
#

# class TestRegistrationSite20(unittest.TestCase):
#     def testRegistrationFromFootprint20(self):
#         fname = 'data/footprints/site20.pcd'
#         fp_name = 'data/footprints/20.las_footprint.csv'
#         foutname = 'data/footprints/20.out.las'
#
#         assert os.path.exists(fname)
#         assert os.path.exists(fp_name)
#         print "loading file"
#         pc = pcl.load(fname,loadRGB=True)
#         conversions.register(pc)
#         footprint = conversions.loadCsvPolygon(fp_name)
#         # translate
#         footprint[:,0] += -1.579381346780
#         footprint[:,1] += 0.52519696509
#
#         pc = register_from_footprint(pc, footprint)
#
#         print "Writing files"
#         conversions.writeLas(foutname, pc)

class TestRegistrationSite20(unittest.TestCase):
    def testRegistrationFromFootprint20(self):
        fname = 'data/footprints/site20.pcd'
        frefname = 'data/footprints/20.las'
        fp_name = 'data/footprints/20.las_footprint.csv'
        foutname = 'data/footprints/20.out.las'
        frefoutname = 'data/footprints/20.fp.out.las'

        assert os.path.exists(fname)
        assert os.path.exists(fp_name)
        print "loading file"
        pc = pcl.load(fname,loadRGB=True)
        conversions.register(pc)
        footprint = conversions.loadCsvPolygon(fp_name)
        # translate
        footprint[:,0] += -1.579381346780
        footprint[:,1] += 0.52519696509
        pc_ref = conversions.loadLas(frefname)
        pc_fp = registration.intersect_polgyon2d(pc_ref, footprint)
        conversions.copy_registration(pc_fp, pc_ref)
        
        pc = registration.register_from_reference(pc, pc_fp)
        
        print "Writing files"
        conversions.writeLas(foutname, pc)
        conversions.writeLas(frefoutname, pc_fp)
