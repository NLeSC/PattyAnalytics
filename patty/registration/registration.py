#!/usr/bin/env python

"""
Registration algorithms and utility functions
"""

from __future__ import print_function
import numpy as np
from .. import BoundingBox, force_srs, extract_mask, clone
from .stickscale import get_stick_scale
from pcl.registration import gicp

from patty.utils import (
    log,
    save,
    downsample_voxel,
)

from patty.segmentation import (
    boundary_of_center_object,
    boundary_of_drivemap,
    boundary_of_lowest_points,
)

from sklearn.decomposition import PCA


def align_footprints(loose_pc, fixed_pc,
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
        log(" - Finding rotation")
        rot_matrix = find_rotation_xy(loose_pc, fixed_pc)
        loose_pc.rotate(rot_matrix, origin=rot_center)
    else:
        log(" - Skipping rotation")
        rot_matrix = np.eye(3)


    if allow_scaling:
        fixed_bb = BoundingBox(fixed_pc) # used 2x below
        loose_bb = BoundingBox(loose_pc)
        scale = fixed_bb.size[0:2] / loose_bb.size[0:2]

        # take the average scale factor for the x and y dimensions
        scale = np.mean(scale)
        loose_pc.scale(scale, origin=rot_center)
        log(" - Scale: %s" % scale )
    else:
        log(" - Skipping scale")
        scale = 1.0


    if allow_translation:
        translation = fixed_pc.center() - rot_center
        loose_pc.translate(translation)
        log(" - Translation: %s" % translation )
    else:
        log(" - Skipping translation")
        translation = np.array([0.0, 0.0, 0.0])

    return rot_matrix, rot_center, scale, translation


def estimate_pancake_up(pointcloud):
    '''
    Assuming a pancake like pointcloud, the up direction is the third PCA.
    '''
    pca = PCA(n_components=3)

    points = np.asarray(pointcloud)
    pca.fit( points[:,0:3] )

    return pca.components_[2]


def _find_rotation_xy_helper(pointcloud):
    pca = PCA(n_components=2)

    points = np.asarray(pointcloud)
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
        print("Alternative")
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
    pc.rotate( rotation, origin=pc.center() )

    return pc


def initial_registration(pointcloud, up, drivemap, initial_scale=None, trust_up=True):
    """
    Initial registration adds an spatial reference system to the pointcloud,
    and place the pointlcoud on top of the drivemap. The pointcloud is rotated
    so that the up vector points along [0,0,1], and scaled such that it has the
    right order of magnitude in size.

    Arguments:
        pointcloud : pcl.PointCloud
            The high-res object to register.

        up: np.array([3])
            Up direction for the pointcloud.
            If None, assume the object is pancake shaped, and chose the upvector such
            that it is perpendicullar to the pancake.

        drivemap : pcl.PointCloud
            A small part of the low-res drivemap on which to register.

        initial_scale : float
            if given, scale pointcloud using this value; estimate scale factor
            from bounding boxes.

        trust_up : Boolean, default to True
            True:  Assume the up vector is exact.
            False: Calculate 'up' as if it was None, but orient it such that
                   np.dot( up, pancake_up ) > 0

    NOTE: Modifies the input pointcloud in-place, and leaves
    it in a undefined state.

    """
    log( "Starting initial registration" )

    #####
    # set scale and offset of pointcloud, drivemap, and footprint
    # as the pointcloud is unregisterd, the coordinate system is undefined,
    # and we lose nothing if we just copy it

    if( hasattr(pointcloud, "offset") ):
        log( " - Dropping initial offset, was: %s" % pointcloud.offset)
    else:
        log( " - No initial offset" )
    force_srs(pointcloud, same_as=drivemap)
    log( " - New offset forced to: %s" % pointcloud.offset )

    if up is not None:
        log( " - Rotating the pointcloud so up points along [0,0,1]" )

        if trust_up:
            rotate_upwards(pointcloud, up)
            log( " - Using trusted up: %s" % up )
        else:
            pancake_up = estimate_pancake_up(pointcloud)
            if np.dot( up, pancake_up ) < 0.0:
                pancake_up *= -1.0
            log( " - Using estimated up: %s" % pancake_up )
            rotate_upwards(pointcloud, pancake_up)

    else:
        log( " - No upvector, skipping" )

    if initial_scale is None:
        bbDrivemap = BoundingBox( points=np.asarray( drivemap ) )
        bbObject   = BoundingBox( points=np.asarray( pointcloud ) )
        scale = bbDrivemap.size[0:2] / bbObject.size[0:2] # ignore z-direction

        # take the average scale factor for x and y dimensions
        scale = np.mean(scale)
    else:
        # use user provided scale
        scale = initial_scale

    log(" - Applying rough estimation of scale factor", scale )
    pointcloud.scale(scale)  # dont care about origin of scaling


def coarse_registration(pointcloud, drivemap, footprint, downsample=None):
    """
    Improve the initial registration.
    Find the proper scale by looking for the red meter sticks, and calculate
    and align the pointcloud's footprint.

    Arguments:
        pointcloud: pcl.PointCloud
                    The high-res object to register.

        drivemap:   pcl.PointCloud
                    A small part of the low-res drivemap on which to register

        footprint:  pcl.PointCloud
                    Pointlcloud containing the objects footprint

        downsample: float, default=None, no resampling
                    Downsample the high-res pointcloud before footprint calculation.
    """
    log( "Starting coarse registration" )

    ###
    # find redstick scale, and use it if possible
    log(" - Redstick scaling")

    allow_scaling = True

    scale, confidence = get_stick_scale(pointcloud)
    log(" - Red stick scale=%s confidence=%s" % (scale, confidence))

    if (confidence > 0.5):
        log(" - Applying red stick scale")
        pointcloud.scale(1.0 / scale)  # dont care about origin of scaling
        allow_scaling = False
    else:
        log(" - Not applying red stick scale, confidence too low")

    #####
    # find all the points in the drivemap along the footprint
    # use bottom two meters of drivemap (not trees)

    dm_boundary = boundary_of_drivemap(drivemap, footprint)
    dm_bb = BoundingBox( dm_boundary )

    # set footprint height from minimum value,
    # as trees, or high object make the pc.center() too high
    fixed_boundary = clone(footprint)
    fp_array = np.asarray( fixed_boundary )
    fp_array[:,2] = dm_bb.min[2]

    save( fixed_boundary, "fixed_bound.las" )

    #####
    # find all the boundary points of the pointcloud

    loose_boundary = boundary_of_center_object(pointcloud, downsample)
    if loose_boundary is None:
        log(" - boundary estimation failed, using lowest 30 percent of points" )
        loose_boundary = boundary_of_lowest_points(pointcloud, height_fraction=0.3 )

    ####
    # match the pointcloud boundary with the footprint boundary

    log( " - Aligning footprints:" )
    rot_matrix, rot_center, scale, translation = align_footprints(
        loose_boundary, fixed_boundary,
        allow_scaling=allow_scaling,
        allow_rotation=True,
        allow_translation=True)

    save( loose_boundary, "aligned_bound.las" )

    ####
    # Apply to the main pointcloud

    pointcloud.rotate(rot_matrix, origin=rot_center)
    pointcloud.scale(scale, origin=rot_center)
    pointcloud.translate(translation)
    rot_center += translation

    return rot_center


def _fine_registration_helper(pointcloud, drivemap, voxelsize=0.05, attempt=0):
    '''
    Perform ICP on pointcloud with drivemap, and return convergence indicator.
    Reject large translatoins.

    Returns:
        transf : np.array([4,4])
            transform
        success : Boolean
            if icp was successful
        fitness : float
            sort of sum of square differences, ie. smaller is better

    '''

    ####
    # Downsample to speed up
    # use voxel filter to keep evenly distributed spatial extent

    log( " - Downsampling with voxel filter: %s" % voxelsize )
    pc = downsample_voxel( pointcloud, voxelsize )


    ####
    # Clip to drivemap to prevent outliers confusing the ICP algorithm

    log( " - Clipping to drivemap" )
    bb = BoundingBox( drivemap )
    z = bb.center[2]
    extracted = extract_mask(pc, [bb.contains([point[0],point[1],z]) for point in pc])

    log( " - Remaining points: %s" % len(extracted) )

    ####
    # GICP

    converged, transf, estimate, fitness = gicp(extracted, drivemap)

    ####
    # Dont accept large translations

    translation = transf[0:3,3]
    if np.dot( translation, translation ) > 5 ** 2:
        log(" - Translation too large, considering it a failure." )
        converged = False
        fitness = 1e30
    else:
        log(" - Success, fitness: ", converged, fitness )

    force_srs( estimate, same_as=pointcloud )
    save( estimate, "attempt%s.las" % attempt )

    return transf, converged, fitness

def fine_registration(pointcloud, drivemap, center, voxelsize=0.05):
    '''
    Final registration step using ICP.

    Find the local optimal postion of the pointcloud on the drivemap; due to
    our coarse_registration algorithm, we have to try two orientations:
    original, and rotated by 180 degrees around the z-axis.

    Arguments:
        pointcloud: pcl.PointCloud
                    The high-res object to register.

        drivemap: pcl.PointCloud
                    A small part of the low-res drivemap on which to register

        center: np.array([3])
                    Vector giving the centerpoint of the pointcloud, used to do
                    the 180 degree rotations.

        voxelsize: float default : 0.05
                    Size in [m] of the voxel grid used for downsampling
    '''
    log( "Starting fine registration" )

    # for rotation around z-axis
    rot=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    # rot=np.array([[-1,0,0],[0,-1,0],[0,0,1]])

    ####
    # do a ICP step for 4 orientations

    transf  = {}
    success = {}
    fitness = {}
    for i in range(4):
        log( " - attempt: %s" % i )
        transf[i], success[i], fitness[i] = _fine_registration_helper(
                                             pointcloud,
                                             drivemap, attempt=i,
                                             voxelsize=voxelsize)
        pointcloud.rotate( rot, origin=center)

    ####
    # pick best

    best,value = min(fitness.iteritems(), key=lambda x:x[1])
    if success[ best ]:
        log( " - Best attempt: %s" % best )
        pointcloud.rotate( rot**best, origin=center).transform( transf[best] )
        return

    # ICP failed:
    # return the pointcloud with just footprints aligned
    # no use to undo a rotation, as any orientationi is equally likely.
    log( " - Unable to do fine registration" )

    return
