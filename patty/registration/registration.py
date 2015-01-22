#!/usr/bin/env python

"""
Registration algorithms and utility functions
"""

from __future__ import print_function
import pcl.registration
from pcl.boundaries import estimate_boundaries
import numpy as np
import logging
from patty import conversions
from sklearn.decomposition import PCA
from patty.segmentation import dbscan
from matplotlib import path

logging.basicConfig(level=logging.INFO)

def length_3d(pointcloud):
    xyz_array = np.asarray(pointcloud)
    return xyz_array.max(axis=0) - xyz_array.min(axis=0)

def scale(pointcloud, scale_factor):
    transf = np.identity(4, dtype=float)*scale_factor
    transf[3,3] = 1.0
    return pointcloud.transform(transf)

def downsample(pointcloud, voxel_size=0.01):
    ''' Downsample a pointcloud using a voxel grid filter.
    Arguments:
        pointcloud    Original pointcloud
        voxel_size    Grid spacing for the voxel grid
    Returns:
        filtered_pointcloud
    '''
    old_len = len(pointcloud)
    pc_filter = pointcloud.make_voxel_grid_filter()
    pc_filter.set_leaf_size(voxel_size, voxel_size, voxel_size)
    filtered_pointcloud = pc_filter.filter()
    new_len = len(filtered_pointcloud)
    decrease_percent = (old_len - new_len)*100 / old_len
    logging.info("number of points reduced from", old_len, "to", new_len, "(", decrease_percent, "% decrease)")
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

def get_pointcloud_boundaries(pointcloud, angle_threshold=0.1, search_radius=0.02, normal_search_radius=0.02):
    '''Find the boundary of a pointcloud.
    Arguments:
        pointcloud            Input pointcloud
        angle_threshold=0.1 
        search_radius=0.02
        normal_radius=0.02
    Returns:
        a pointcloud
    '''
    boundary = estimate_boundaries(pointcloud, angle_threshold=0.1, search_radius=0.02, normal_search_radius=0.02)
    return pointcloud.extract(np.where(boundary)[0])

def principal_axes_rotation(data):
    '''Find the 3 princial axis of the pointcloud, and the rotation to align it to the x,y, and z axis.

    Arguments:
        data    pointcloud
    Returns:
        transformation matrix
    '''
    pca = PCA(n_components=data.shape[1])
    pca.fit(data)
    transform = np.zeros((4,4))
    transform[:3,:3] = np.array(pca.components_)
    transform[3,3] = 1.0
    
    return np.matrix(transform)

def register_from_footprint(pc, footprint):
    '''Register a pointcloud by placing it in footprint. Applies dbscan first.
    Arguments:
        pc         pointcloud
        footprint  array of [x,y,z] describing the footprint
    Returns:
        the original pointcloud, but rotated/translated to the footprint
    '''
    logging.info("Finding largest cluster")
    pc_main = dbscan.largest_dbscan_cluster(pc, .1, 250)
    conversions.copy_registration(pc_main, pc)
    
    logging.info("Detecting boundary")
    boundary = get_pointcloud_boundaries(pc_main)
    conversions.copy_registration(boundary, pc_main)
    
    logging.info("Finding rotation")
    pc_transform = principal_axes_rotation(np.asarray(boundary))
    fp_transform = principal_axes_rotation(footprint)
    transform = np.linalg.inv(fp_transform) * pc_transform
    boundary.transform(transform)

    logging.info("Registering pointcloud to footprint")
    registered_offset, registered_scale = register_offset_scale_from_ref(boundary, footprint)
    conversions.copy_registration(pc, boundary)
    
    # rotate and scale up
    transform[:3,:3] *= registered_scale
    pc.transform(transform)
    
    return pc

def register_from_reference(pc, pc_ref):
    '''Register a pointcloud by aligning it with a reference pointcloud. Applies dbscan first.
    Arguments:
        pc         pointcloud
        footprint  array of [x,y,z] describing the footprint
    Returns:
        the original pointcloud, but rotated/translated to the footprint
    '''
    logging.info("Finding largest cluster")
    pc_main = dbscan.largest_dbscan_cluster(pc, .1, 250)
    conversions.copy_registration(pc_main, pc)
    
    logging.info("Finding rotation")
    pc_transform = principal_axes_rotation(np.asarray(pc_main))
    ref_transform = principal_axes_rotation(np.asarray(pc_ref))
    transform = np.linalg.inv(ref_transform) * pc_transform
    pc_main.transform(transform)

    logging.info("Registering pointcloud to footprint")
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
    intersection = pc.extract( np.where(in_polygon)[0] )
    return intersection

