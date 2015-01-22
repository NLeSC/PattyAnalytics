import unittest
import pcl
import os.path
from patty.conversions import conversions
from patty.registration import registration

class TestRegistrationScaleOffset(unittest.TestCase):
    def testRegistrationFromFootprint(self):
        fname = 'data/footprints/162.las'
        fp_name = 'data/footprints/162.las_footprint.csv'
        assert os.path.exists(fname)
        assert os.path.exists(fp_name)
        pc = conversions.loadLas(fname)
        footprint = conversions.loadCsvPolygon(fp_name)
        registration.register_from_footprint(pc, footprint)

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

        # Shift footprint by (-1.579, 0.525) -- value estimated manually
        footprint[:,0] += -1.579381346780
        footprint[:,1] += 0.52519696509
        pc_ref = conversions.loadLas(frefname)
        pc_fp = registration.intersect_polgyon2d(pc_ref, footprint)
        conversions.copy_registration(pc_fp, pc_ref)
        
        pc = registration.register_from_reference(pc, pc_fp)
        
        print "Writing files"
        conversions.writeLas(foutname, pc)
        conversions.writeLas(frefoutname, pc_fp)

