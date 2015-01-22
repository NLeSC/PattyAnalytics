#!/usr/bin/env python

"""
Run the registration algorithm on PCD or ply files files.
"""

from __future__ import print_function
import argparse
import pcl
import pcl.registration
from pcl.boundaries import estimate_boundaries
import time
import sys
import numpy as np
from patty.conversions import conversions
from patty.conversions.conversions import copy_registration, extract_mask
from sklearn.decomposition import PCA
from patty.segmentation import dbscan
from matplotlib import path

def log(*args, **kwargs):
    print(time.strftime("[%H:%M:%S]"), *args, **kwargs)

def process_args():
    """ Parse arguments from the command-line using argparse """

    # Implemented registration functions
    funcs = {
        'icp': pcl.registration.icp,
        'gicp': pcl.registration.gicp,
        'icp_nl': pcl.registration.icp_nl,
        'ia_ransac': pcl.registration.ia_ransac
    }

    # For use in the argparser to select value from interval
    class Interval(object):
        def __init__(self, minimum, maximum):
            self._min = minimum
            self._max = maximum

        # for the 0.5 in Interval(0, 0.5)
        def __contains__(self, x):
            return self._min < x < self._max

        # make it iterable for pretty printing
        def __iter__(self):
            self._istate = 0
            return self
        def next(self):
            if self._istate == 0:
                self._istate = 1
                return self._min
            elif self._istate == 1:
                self._istate = 2
                return self._max
            else:
                raise StopIteration

    parser = argparse.ArgumentParser(description='Registration for two PLY point clouds')
    parser.add_argument('-f','--function', choices=funcs.keys(), help='Registration algorithm to run. Choose between gicp, icp, icp_nl, and ia_ransac.', default='gicp')
    parser.add_argument('-d','--downsample', metavar="downsample", nargs=1, type=float, help='Downsample to use one point per given voxel size. Suggested value: 0.005.', choices=Interval(0.0, 1.0) )
    parser.add_argument('source', metavar="SOURCE", help="Source LAS file")
    parser.add_argument('target', metavar="TARGET", help="Target LAS file to map source to")
    
    args = parser.parse_args()

    log("reading source", args.source)
    # source = pcl.load(args.source)
    source = conversions.loadLas(args.source)
    log("offset:", source.offset)
    log("reading target ", args.target)
    target = conversions.loadLas(args.target)
    log("offset:", target.offset)
    # target = pcl.load(args.target)
    
    algo = funcs[args.function]
    
    return source, target, algo, args.downsample

def print_output(algo, converged, transf, fitness):
    """ Print some output based on the algorithm output """
    
    log("converged:", converged, "- error:", fitness)
    log("rotation:")
    log("\n", transf[0:3,0:3])
    log("translation:", transf[3, 0:3])
    log("---------------")

def length_3d(pointcloud):
    xyz_array = np.asarray(pointcloud)
    return xyz_array.max(axis=0) - xyz_array.min(axis=0)

def scale(pointcloud, scale_factor):
    transf = np.identity(4, dtype=float)*scale_factor
    transf[3,3] = 1.0
    return pointcloud.transform(transf)

def downsample(pointcloud, voxel_size=0.01):
    old_len = len(pointcloud)
    pc_filter = source.make_voxel_grid_filter()
    pc_filter.set_leaf_size(voxel_size, voxel_size, voxel_size)
    filtered_pointcloud = pc_filter.filter()
    new_len = len(filtered_pointcloud)
    decrease_percent = (old_len - new_len)*100 / old_len
    log("number of points reduced from", old_len, "to", new_len, "(", decrease_percent, "% decrease)")
    return filtered_pointcloud

def register_offset_scale_from_ref(pc, ref_array, ref_offset=np.zeros(3)):
    ''' Returns a 3d-offset and uniform scale value from footprint.
    The scale is immediately applied to the pointcloud, the offset is set to the patty_registration.conversions.RegisteredPointCloud'''
    ref_min = ref_array.min(axis=0)
    ref_max = ref_array.max(axis=0)
    ref_center = (ref_min + ref_max) / 2.0 + ref_offset

    pc_array = np.asarray(pc)
    pc_min = pc_array.min(axis=0)
    pc_max = pc_array.max(axis=0)
    
    pc_size = pc_max - pc_min
    ref_size = ref_max - ref_min
    
    # Take the footprint as the real offset, and correct the z-offset
    # The z-offset of the footprint will be ground level, the z-offset of the
    # pointcloud will include the monuments height
    pc_registration_scale = np.mean(ref_size[0:1]/pc_size[0:1])

    pc_array *= pc_registration_scale
    pc_min *= pc_registration_scale
    pc_max *= pc_registration_scale
    
    conversions.register(pc, offset=ref_center - (pc_min + pc_max) / 2.0, precision=pc.precision * pc_registration_scale)

    return pc.offset, pc_registration_scale

def get_pointcloud_boundaries(pointcloud):
    boundary = estimate_boundaries(pointcloud, angle_threshold=0.1, search_radius=0.02, normal_search_radius=0.02)
    return extract_mask(pointcloud, boundary)

def principal_axes_rotation(data):
    pca = PCA(n_components=data.shape[1])
    pca.fit(data)
    transform = np.zeros((4,4))
    transform[:3,:3] = np.array(pca.components_)
    transform[3,3] = 1.0
    
    return np.matrix(transform)

def register_from_footprint(pc, footprint):
    log("Finding largest cluster")
    pc_main = dbscan.largest_dbscan_cluster(pc, .1, 250)
    conversions.copy_registration(pc_main, pc)
    
    log("Detecting boundary")
    boundary = get_pointcloud_boundaries(pc_main)
    conversions.copy_registration(boundary, pc_main)
    
    log("Finding rotation")
    pc_transform = principal_axes_rotation(np.asarray(boundary))
    fp_transform = principal_axes_rotation(footprint)
    transform = np.linalg.inv(fp_transform) * pc_transform
    boundary.transform(transform)

    log("Registering pointcloud to footprint")
    registered_offset, registered_scale = register_offset_scale_from_ref(boundary, footprint)
    conversions.copy_registration(pc, boundary)
    
    # rotate and scale up
    transform[:3,:3] *= registered_scale
    pc.transform(transform)
    
    return pc

def register_from_reference(pc, pc_ref):
    log("Finding largest cluster")
    pc_main = dbscan.largest_dbscan_cluster(pc, .1, 250)
    conversions.copy_registration(pc_main, pc)
    
    log("Finding rotation")
    pc_transform = principal_axes_rotation(np.asarray(pc_main))
    ref_transform = principal_axes_rotation(np.asarray(pc_ref))
    transform = np.linalg.inv(ref_transform) * pc_transform
    pc_main.transform(transform)

    log("Registering pointcloud to footprint")
    registered_offset, registered_scale = register_offset_scale_from_ref(pc_main, np.asarray(pc_ref), pc_ref.offset)
    conversions.copy_registration(pc, pc_main)
    
    # rotate and scale up
    transform[:3,:3] *= registered_scale
    pc.transform(transform)
    
    return pc

def point_in_convex_polygon(points, polygon):
    ''' WARNING: Only works for convex polygons '''
    mask = np.ones(len(points),dtype=np.bool)
    for i in xrange(len(polygon)):
        v1 = polygon[i - 1] - polygon[i]
        v2 = points - polygon[i - 1]
        
        is_left = v1[0]*v2[:,1] - v1[1]*v2[:,0] >= 0
        mask = mask & is_left
         
    return mask

def point_in_polygon2d(points, polygon):
    p = path.Path(polygon[:,:2])
    return np.array( [p.contains_point(point[:2]) for point in points] )        

def intersect_polgyon2d(pc, polygon):
    in_polygon = point_in_polygon2d(np.asarray(pc) + pc.offset, polygon)
    return extract_mask(pc, in_polygon)

def scale_points(polygon, factor):
    polygon = np.array(polygon,dtype=np.float64)
    offset = (polygon.max(axis=0) + polygon.min(axis=0)) / 2.0
    return ((polygon - offset) * factor) + offset

if __name__ == '__main__':
    source, target, algo, voxel_size = process_args()
    
    # choose the maximum size over all coordinates to determine scale
    # src_maxsize = max(length_3d(source))
    # tgt_maxsize = max(length_3d(target))
    
    # preprocess source and target
    # log("scaling source down by: ", src_maxsize)
    # source = scale(source, 1.0/src_maxsize)
    # log("scaling target down by: ", tgt_maxsize)
    # target = scale(target, 1.0/tgt_maxsize)

    if voxel_size is not None:
        log("downsampling source using voxel size", voxel_size)
        source = downsample(source, voxel_size=voxel_size)
        log("downsampling target using voxel size", voxel_size)
        target = downsample(target, voxel_size=voxel_size)

    log("------", algo.__name__, "-----")
    converged, transf, estimate, fitness = algo(source, target)
    print_output(algo, converged, transf, fitness)
