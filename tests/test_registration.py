import unittest
import pcl
import os.path
import numpy as np
from patty_registration import conversions, registration
from patty_segmentation import dbscan
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

class TestRegistrationSite20(unittest.TestCase):
    def testRegistrationFromFootprint20(self):
        fname = 'data/footprints/site20.pcd'
        fp_name = 'data/footprints/20.las_footprint.csv'
        foutname = 'data/footprints/20.out.las'
        fboundaryname = 'data/footprints/20.boundary.las'
        assert os.path.exists(fname)
        assert os.path.exists(fp_name)
        print "loading file"
        pc = pcl.load(fname,loadRGB=True)
        conversions.register(pc)
        footprint = conversions.loadCsvPolygon(fp_name)
        
        print "Selecting main cluster"
        clusters = dbscan.segment_dbscan(pc, .1, 250)
        pc_main = max(clusters, key=len)
        conversions.copy_registration(pc_main, pc)
                
        print "Discovering boundary"
        boundary = registration.get_pointcloud_boundaries(pc_main)
        conversions.copy_registration(boundary, pc_main)
        
        print "Finding rotation"
        pc_transform = np.matrix(registration.principal_axes_rotation(np.asarray(boundary)))
        fp_transform = np.matrix(registration.principal_axes_rotation(footprint))
        transform = np.linalg.inv(fp_transform) * pc_transform
        boundary.transform(transform)

        print "Registering pointcloud to footprint"
        registered_offset, registered_scale = registration.register_from_footprint(footprint, boundary)
        
        conversions.copy_registration(pc, boundary)
        pc.transform(transform)
        pc_array = np.asarray(pc)
        pc_array *= registered_scale
        
        print "Writing files"
        conversions.writeLas(fboundaryname, boundary)
        conversions.writeLas(foutname, pc)
        #
        #
        # print pc[0]
        # print pc[1]
        # print pc[2]
        # print pc[3]
        #
        # print registered_offset, registered_scale
        #
        # a = np.asarray(pc)
        # a *= registered_scale
        # a += registered_offset
        
        # outname = 'data/footprints/SITE_20_k1000_s2.out.ply'
        # pcl.save(pc, outname)
