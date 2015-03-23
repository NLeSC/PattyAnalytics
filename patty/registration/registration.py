#!/usr/bin/env python

"""
Registration algorithms and utility functions
"""

from __future__ import print_function
from pcl.boundaries import estimate_boundaries
import numpy as np
import logging
import json
from .. import copy_registration, is_registered, extract_mask, set_registration, BoundingBox
from ..segmentation import dbscan
from matplotlib import path
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)


def downsample_voxel(pointcloud, voxel_size=0.01):
    '''Downsample a pointcloud using a voxel grid filter.

    Arguments:
        pointcloud    Original pointcloud
        voxel_size    Grid spacing for the voxel grid
    Returns:
        filtered_pointcloud
    '''
    pc_filter = pointcloud.make_voxel_grid_filter()
    pc_filter.set_leaf_size(voxel_size, voxel_size, voxel_size)
    return pc_filter.filter()


def downsample_random(pc, fraction, random_seed=None):
    """Randomly downsample pointcloud to a fraction of its size.

    Returns a pointcloud of size fraction * len(pc), rounded to the nearest
    integer.

    Use random_seed=k for some integer k to get reproducible results.
    Arguments:
        pc : pcl.PointCloud
            Input pointcloud.
        fraction : float
            Fraction of points to include.
        random_seed : int, optional
            Seed to use in random number generator.

    Returns:
        pcl.Pointcloud
    """
    if not 0 < fraction <= 1:
        raise ValueError("Expected fraction in (0,1], got %r" % fraction)

    rng = np.random.RandomState(random_seed)

    k = max(int(round(fraction * len(pc))), 1)
    sample = rng.choice(len(pc), k, replace=False)
    new_pc = pc.extract(sample)
    if is_registered(pc):
        copy_registration(new_pc, pc)
    return new_pc

def get_pointcloud_boundaries(pointcloud, angle_threshold=0.1,
                              search_radius=None, normal_search_radius=None):
    '''Find the boundary of a pointcloud.

    Arguments:
        pointcloud : pcl.PointCloud

        angle_threshold : float

        search_radius : float defaults to 1 percent of pointcloud size as
                        determined by the diagonal of the boundingbox

        normal_search_radius : float defaults to search_radius

    Returns:
        a pointcloud
    '''

    if search_radius == None:
        bb = BoundingBox(points=pointcloud)
        logging.info( bb )
        logging.info( bb.diagonal )
        search_radius = 0.01 * bb.diagonal 

    if normal_search_radius == None:
        normal_search_radius = search_radius

    logging.info("Search radius from bounding box: %f" % search_radius )

    boundary = estimate_boundaries(pointcloud, angle_threshold=angle_threshold,
                                   search_radius=search_radius,
                                   normal_search_radius=normal_search_radius)

    logging.info("Found %d out of %d boundary points" 
                   % (np.count_nonzero(boundary),len(pointcloud)))

    return extract_mask(pointcloud, boundary)


def register_from_footprint(pc, footprint, allow_scaling=True, allow_rotation=True, allow_translation=True):
    '''Register a pointcloud by placing it in footprint.

    Applies dbscan to find the main object, and estimates its footprint
    by taking the pointcloud boundary.
    Then the pointcloud footprint is alinged with the reference footprint
    by rotating its pricipal axis, and translating it so the centers of mass
    coincide. 
    Finally, the pointcloud is scaled to have the same extent. The scale factor is
    is determined by the red meter sticks detection algorithm, or form the fit between
    pc and footprint if red meter sticks cant be found.

    Arguments:
        pc : pcl.PointCloud

        footprint : np.ndarray
            Array of [x,y,z] describing the footprint.

    Returns:
        The original pointcloud, rotated/translated to match the footprint.
    '''
    # find the footprint of the pointcloud: the boundary of its center object
    # (ie. largest object)
    logging.info("Finding largest cluster")
    pc_main = dbscan.largest_dbscan_cluster(pc, .1, 250)

    logging.info("Detecting boundary")
    boundary = get_pointcloud_boundaries(pc_main)

    if len(boundary) == len(pc_main) or len(boundary) == 0:
        log("Boundary information could not be retrieved")
        return None

    if allow_rotation:
        logging.info("Finding rotation")
        rot_center = boundary.center()
        rot_matrix = find_rotation(boundary, footprint)
        boundary.rotate(rot_matrix, origin=rot_center)
    else:
        rot_center = np.array( [0.0, 0.0, 0.0] )
        rot_matrix = np.eye(3)

    if allow_scaling:
        logging.info("Finding scale")
        footprint_bb = BoundingBox( footprint )
        boundary_bb = BoundingBox( boundary )
        scale = footprint_bb.size / boundary_bb.size
        # take the average scale factor for the x and y dimensions
        scale = np.mean( scale[0:2] )
    else:
        scale = 1.0

    if allow_translation:
        logging.info("Finding translation")
        fp_center = np.mean( footprint, axis=0 )
        translation = fp_center - rot_center
        boundary.translate( translation ) 
    else:
        translation = np.array( [0.0,0.0,0.0] )

    return rot_matrix, rot_center, scale, translation


def _find_rotation_helper(pointcloud):
    pca = PCA(n_components=3)
    pca.fit(np.asarray(pointcloud))

    rotation = np.array(pca.components_)

    # keep the up direction pointing (mostly) upwards
    if rotation[2,2] < 0.0:
        rotation[:,2] *= -1.0 # flip the whole vector

    # make sure the rotation is a proper rotation, ie det = +1
    if np.linalg.det( rotation ) < 0:
        rotation[:,1] *= -1.0

    return rotation 

def find_rotation(pc, ref):
    '''Find the transformation that rotates the principal axis of the
    pointcloud onto those of the reference.
    Keep the z-axis pointing upwards.

    Arguments:
        pc: pcl.PointCloud

        ref: pcl.PointCloud

    Returns:
        numpy array of shape [3,3], can be used to rotate pointclouds with pc.rotate()
    '''

    pc_transform  = _find_rotation_helper( pc )
    ref_transform = _find_rotation_helper( ref )

    return np.dot(np.linalg.inv(ref_transform), pc_transform)


def point_in_polygon2d(points, polygon):
    p = path.Path(np.asarray(polygon)[:, :2])
    return np.array([p.contains_point(point[:2]) for point in points],
                    dtype=np.bool)


def intersect_polygon2d(pc, polygon):
    in_polygon = point_in_polygon2d(np.asarray(pc) + pc.offset, polygon)
    return extract_mask(pc, in_polygon)

def is_upside_down(upfilepath, transform):
    '''Decides if a pointcloud is upside down using its relative up
    vector and the transformation (rotation only) matrix.

    Arguments:
        upfilepath path of the json file containing the relative up vector
        transform  2d array describing the rotation matrix
    Returns:
        true  pointcloud is upside down
        false pointcloud is right side up
    '''
    if upfilepath in (None, ''):
        return False

    try:
        with open(upfilepath) as upfile:
            dic = json.load(upfile)
    except:
        return False

    estimated_up = np.array(dic['estimatedUpDirection'])
    real_up = transform[:,2]

    return np.dot( estimated_up, real_up ) < 0
