#!/usr/bin/env python

"""
Registration algorithms and utility functions
"""

from __future__ import print_function
from pcl import PointCloud
from pcl.boundaries import estimate_boundaries
import numpy as np
from .. import is_registered, extract_mask, BoundingBox, log, save
from ..segmentation import get_largest_dbscan_clusters
from patty import conversions

from sklearn.decomposition import PCA

from shapely.geometry.polygon import LinearRing
from shapely.geometry import Point




def downsample_voxel(pc, voxel_size=0.01):
    '''Downsample a pointcloud using a voxel grid filter.
    Resulting pointcloud has the same SRS and offset as the input.

    Arguments:
        pc         : pcl.PointCloud
                     Original pointcloud
        float      : voxel_size
                     Grid spacing for the voxel grid
    Returns:
        pc : pcl.PointCloud
             filtered pointcloud
    '''
    pc_filter = pc.make_voxel_grid_filter()
    pc_filter.set_leaf_size(voxel_size, voxel_size, voxel_size)
    newpc = pc_filter.filter()

    conversions.force_srs(newpc, same_as=pc)

    return newpc


def downsample_random(pc, fraction, random_seed=None):
    """Randomly downsample pointcloud to a fraction of its size.

    Returns a pointcloud of size fraction * len(pc), rounded to the nearest
    integer.  Resulting pointcloud has the same SRS and offset as the input.

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

    conversions.force_srs(new_pc, same_as=pc)

    return new_pc



def boundary_of_drivemap(drivemap, footprint, height=1.0, edge_width=0.25):
    '''
    Construct an object boundary using the manually recorded corner points.
    Do this by finding all the points in the drivemap along the footprint.
    Use the bottom 'height' meters of the drivemap (not trees).
    Resulting pointcloud has the same SRS and offset as the input.

    Arguments:
        drivemap   : pcl.PointCloud
        footprint  : pcl.PointCloud
        height     : Cut-off height, points more than this value above the
                     lowest point of the drivemap are considered trees,
                     and dropped. default 1 m.
        edge_width : Points belong to the boundary when they are within
                     this distance from the footprint. default 0.25

    Returns:
        boundary   : pcl.PointCloud
    '''

    # construct basemap as the bottom 'height' meters of the drivemap

    drivemap_array = np.asarray(drivemap)
    bb = BoundingBox(points=drivemap_array)
    basemap = extract_mask(drivemap, drivemap_array[:, 2] < bb.min[2] + height)

    # Cut band between +- edge_width around the footprint
    edge = LinearRing( np.asarray(footprint) ).buffer( edge_width )

    points = [point for point in basemap if edge.contains( Point(point) )]
    boundary = PointCloud( np.asarray( points, dtype=np.float32) )
    
    conversions.force_srs(boundary, same_as=basemap)

    return boundary


def boundary_via_lowest_points(pc, height_fraction=0.01):
    '''
    Construct an object boundary by taking the lowest (ie. min z coordinate)
    fraction of points.
    Resulting pointcloud has the same SRS and offset as the input.

    Arguments:
        pc               : pcl.PointCloud
        height_fraction  : float

    Returns:
        boundary   : pcl.PointCloud
    '''
    array = np.asarray(pc)
    bb = BoundingBox(points=array)
    maxh = bb.min[2] + (bb.max[2] - bb.min[2]) * height_fraction

    newpc = extract_mask(pc, array[:, 2] < maxh)
    
    conversions.force_srs(newpc, same_as=pc)

    return newpc


def boundary_of_center_object(pc,
                              downsample=None,
                              angle_threshold=0.1,
                              search_radius=0.1,
                              normal_search_radius=0.1):
    '''Find the boundary of the main object.
    First applies dbscan to find the main object,
    then estimates its footprint by taking the pointcloud boundary.
    Resulting pointcloud has the same SRS and offset as the input.

    Arguments:
        pointcloud : pcl.PointCloud

        downsample : If given, reduce the pointcloud to given percentage 
                     values should be in [0,1]

        angle_threshold : float defaults to 0.1

        search_radius : float defaults to 0.1

        normal_search_radius : float defaults to 0.1

    Returns:
        boundary : pcl.PointCloud
    '''

    if downsample is not None:
        log( ' - Downsampling factor:', downsample )
        pc = downsample_random(pc, downsample)
    else:
        log( ' - Not downsampling' )
    save( pc, 'downsampled.las' )

    # find largest cluster, it should be the main object
    log( ' - Starting dbscan' )
    mainobject = get_largest_dbscan_clusters(pc, 0.7, .15, 250) # if this doesnt work, try 2.0

    save( mainobject, 'mainobject.las' )
    log( ' - Finished dbscan' )

    boundary = estimate_boundaries(mainobject,
                                   angle_threshold=angle_threshold,
                                   search_radius=search_radius,
                                   normal_search_radius=normal_search_radius)

    boundary = extract_mask(mainobject, boundary)

    if len(boundary) == len(mainobject) or len(boundary) == 0:
        log( 'Cannot find boundary' )
        return None

    # project on the xy plane 
    points = np.asarray( boundary )
    zmin = np.min( points, axis=0 )[2]
    points[:,2] = zmin
    
    conversions.force_srs(boundary, same_as=pc)

    return boundary



def register_from_footprint(loose_pc, fixed_pc,
                            allow_scaling=True,
                            allow_rotation=True,
                            allow_translation=True):
    '''
    Align a pointcloud 'loose_pc' by placing it on top of
    'fixed_pc' as good as poosible. Done by aligning the 
    principle axis of both pointclouds.

    NOTE: Both pointclouds are assumed to be the footprint (or projection)
    on the xy plane, with basically zero extent along the z-axis.

    (allow_rotation=True)
        The pointcloud boundary is alinged with the footprint
        by rotating its pricipal axis in the (x,y) plane. 

    (allow_translation=True)
        Then, it is translated so the centers of mass coincide.

    (allow_scaling=True)
        Finally, the pointcloud is scaled to have the same extent.

    Arguments:
        loose_pc          : pcl.PointCloud
        fixed_pc          : pcl.PointCloud

        allow_scaling     : Bolean
        allow_rotation    : Bolean
        allow_translation : Bolean

    Returns:
        rot_matrix, rot_center, scale, translation : np.array()

    '''

    rot_center = loose_pc.center()

    if allow_rotation:
        log("Finding rotation")
        rot_matrix = find_rotation_xy(loose_pc, fixed_pc)
        loose_pc.rotate(rot_matrix, origin=rot_center)
    else:
        log("Skipping rotation")
        rot_matrix = np.eye(3)

    if allow_scaling:
        log("Finding scale")
        loose_bb = BoundingBox(loose_pc)
        fixed_bb = BoundingBox(fixed_pc)
        scale = fixed_bb.size[0:2] / loose_bb.size[0:2]

        # take the average scale factor for the x and y dimensions
        scale = np.mean(scale)
    else:
        log("Skipping scale")
        scale = 1.0

    if allow_translation:
        log("Finding translation")
        translation = fixed_pc.center() - rot_center
        loose_pc.translate(translation)
    else:
        log("Skipping translation")
        translation = np.array([0.0, 0.0, 0.0])

    return rot_matrix, rot_center, scale, translation


def _find_rotation_xy_helper(pointcloud):
    pca = PCA(n_components=2)

    points = np.asarray(pointcloud)
    print(points)
    pca.fit( points[:,0:2] )

    rotxy = np.array(pca.components_)

    # make sure the rotation is a proper rotation, ie det = +1
    if np.linalg.det(rotxy) < 0:
        rotxy[:, 1] *= -1.0

    # create a 3D rotation around the z-axis
    rotation = np.eye(3)
    rotation[0:2,0:2] = rotxy

    return rotation


def find_rotation_xy(pc, ref):
    '''Find the transformation that rotates the principal axis of the
    pointcloud onto those of the reference.
    Keep the z-axis pointing upwards.

    Arguments:
        pc: pcl.PointCloud

        ref: pcl.PointCloud

    Returns:
        numpy array of shape [3,3], can be used to rotate pointclouds with pc.rotate()
    '''

    pc_transform = _find_rotation_xy_helper(pc)
    ref_transform = _find_rotation_xy_helper(ref)

    return np.dot(np.linalg.inv(ref_transform), pc_transform)


def rotate_upwards(pc, up):
    '''
    Rotate the pointcloud in-place around its center, such that the
    'up' vector points along [0,0,1]

    Arguments:
        pc : pcl.PointCloud
        up : np.array([3]) 

    Returns:
        pc : pcl.PointCloud the input pointcloud, for convenience.

    '''

    newz = np.array( up )

    # Right-handed coordiante system:
    # np.cross(x,y) = z
    # np.cross(y,z) = x
    # np.cross(z,x) = y

    # normalize
    newz /= ( np.dot( newz, newz ) ) ** 0.5

    # find two orthogonal vectors to represent x and y,
    # randomly choose a vector, and take cross product. If we're unlucky,
    # this ones is parallel to z, so cross pruduct is zero.
    # In that case, try another one
    try:
        newx = np.cross( np.array([0,1,0]), newz )
        newx /= ( np.dot( newx, newx ) ) ** 0.5

        newy = np.cross( newz, newx )
        newy /= ( np.dot( newy, newy ) ) ** 0.5
    except:
        newy = np.cross( newz, np.array([1,0,0]) )
        newy /= ( np.dot( newy, newy ) ) ** 0.5

        newx = np.cross( newy, newz )
        newx /= ( np.dot( newx, newx ) ) ** 0.5

    rotation = np.zeros([3,3])
    rotation[0,0] = newx[0]
    rotation[1,0] = newx[1]
    rotation[2,0] = newx[2]

    rotation[0,1] = newy[0]
    rotation[1,1] = newy[1]
    rotation[2,1] = newy[2]

    rotation[0,2] = newz[0]
    rotation[1,2] = newz[1]
    rotation[2,2] = newz[2]

    rotation = np.linalg.inv(rotation)

    log( "Rotating pointcloud around origin, using:\n%s" % rotation )
    pc.rotate( rotation, origin=pc.center() )

    return pc 
