#!/usr/bin/env python

"""
Registration algorithms and utility functions
"""

from __future__ import print_function
from pcl.boundaries import estimate_boundaries
import numpy as np
from .. import is_registered, extract_mask, BoundingBox, log, save
from ..segmentation import dbscan
from patty import conversions
from matplotlib import path
from sklearn.decomposition import PCA


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
        conversions.force_srs(new_pc, same_as=pc)
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
        log(bb)
        log(bb.diagonal)
        search_radius = 0.01 * bb.diagonal

    if normal_search_radius == None:
        normal_search_radius = search_radius

    log("Search radius from bounding box: %f" % search_radius)

    boundary = estimate_boundaries(pointcloud, angle_threshold=angle_threshold,
                                   search_radius=search_radius,
                                   normal_search_radius=normal_search_radius)

    log("Found %d out of %d boundary points"
             % (np.count_nonzero(boundary), len(pointcloud)))

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

        footprint : pcl.PointCloud
            Treated as array of [x,y] describing the footprint.

    Returns:
        The original pointcloud, rotated/translated to match the footprint.
    '''
    # find the footprint of the pointcloud: the boundary of its center object
    # (ie. largest object)
    log("Finding largest cluster")
    pc_main = dbscan.largest_dbscan_cluster(pc, .1, 250)

    log("Detecting boundary")
    boundary = get_pointcloud_boundaries(pc_main)


    # We are looking for the main axes to orient the bounding boxes,
    # however, high objects (pillars), the z directions will be the main axis,
    # turning the object to its side.
    # Prevent this by squashing the object to very small z-extent.
    log( "Squashing z axis" )
    avgz = boundary.center()[2]
    squash = np.array ( [[ 1.0, 0.0, 0.0, 0.0 ],
                         [ 0.0, 1.0, 0.0, 0.0 ],
                         [ 0.0, 0.0, 0.0, avgz],
                         [ 0.0, 0.0, 0.0, 1.0 ]] )
    boundary.transform( squash )

    # FIXME: debug output
    save( boundary, "object_boundary.las" )

    if len(boundary) == len(pc_main) or len(boundary) == 0:
        log("Boundary information could not be retrieved")
        return None

    if allow_rotation:
        log("Finding rotation")
        rot_center = boundary.center()
        rot_matrix = find_rotation(boundary, footprint)
        boundary.rotate(rot_matrix, origin=rot_center)
    else:
        log("Skipping rotation")
        rot_center = np.array([0.0, 0.0, 0.0])
        rot_matrix = np.eye(3)

    if allow_scaling:
        log("Finding scale")
        footprint_bb = BoundingBox(footprint)
        boundary_bb = BoundingBox(boundary)
        scale = footprint_bb.size / boundary_bb.size
        # take the average scale factor for the x and y dimensions
        scale = np.mean(scale[0:2])
    else:
        log("Skipping scale")
        scale = 1.0

    if allow_translation:
        log("Finding translation")
        translation = footprint.center() - rot_center
        boundary.translate(translation)
    else:
        log("Skipping translation")
        translation = np.array([0.0, 0.0, 0.0])

    return rot_matrix, rot_center, scale, translation


def _find_rotation_helper(pointcloud):
    pca = PCA(n_components=3)
    pca.fit(np.asarray(pointcloud))

    rotation = np.array(pca.components_)

    # keep the up direction pointing (mostly) upwards
    if rotation[2, 2] < 0.0:
        rotation[:, 2] *= -1.0  # flip the whole vector

    # make sure the rotation is a proper rotation, ie det = +1
    if np.linalg.det(rotation) < 0:
        rotation[:, 1] *= -1.0

    # Apply extra constraints

    # Z should point up
    newz = np.array( [0,0,1] )

    # X should point in x,y plane
    newx = rotation[:,0]
    newx[2] = 0.0
    newx /= np.dot( newx, newx ) ** 0.5

    # Y in x,y plane, perpendicular to x and z
    newy = np.cross( newz, newx )

    rotation[:,0] = newx
    rotation[:,1] = newy
    rotation[:,2] = newz

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

    pc_transform = _find_rotation_helper(pc)
    ref_transform = _find_rotation_helper(ref)

    return np.dot(np.linalg.inv(ref_transform), pc_transform)


def point_in_polygon2d(points, polygon):
    p = path.Path(np.asarray(polygon)[:, :2])
    return np.array([p.contains_point(point[:2]) for point in points],
                    dtype=np.bool)


def intersect_polygon2d(pc, polygon):
    in_polygon = point_in_polygon2d(np.asarray(pc), polygon)
    return extract_mask(pc, in_polygon)
