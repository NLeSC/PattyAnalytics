#!/usr/bin/env python

"""
Registration algorithms and utility functions
"""

from __future__ import print_function
import numpy as np
from .. import BoundingBox, log
from patty import utils

from sklearn.decomposition import PCA

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
        log(" - Finding rotation")
        rot_matrix = find_rotation_xy(loose_pc, fixed_pc)
        loose_pc.rotate(rot_matrix, origin=rot_center)
    else:
        log(" - Skipping rotation")
        rot_matrix = np.eye(3)

    if allow_scaling:
        log("Finding scale")
        loose_bb = BoundingBox(loose_pc)
        fixed_bb = BoundingBox(fixed_pc)
        scale = fixed_bb.size[0:2] / loose_bb.size[0:2]

        # take the average scale factor for the x and y dimensions
        scale = np.mean(scale)
    else:
        log(" - Skipping scale")
        scale = 1.0

    if allow_translation:
        log(" - Finding translation")
        translation = fixed_pc.center() - rot_center
        loose_pc.translate(translation)
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

    log( " - Rotating pointcloud around origin, using:\n%s" % rotation )
    pc.rotate( rotation, origin=pc.center() )

    return pc 
