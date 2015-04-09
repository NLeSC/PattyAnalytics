#!/usr/bin/env python2.7
"""Registration script.

Usage:
  registration.py [-h] [-d <sample>] [-U] [-u <upfile>] [-c <camfile>] <source> <drivemap> <footprint> <output>

Positional arguments:
  source       Source LAS file
  drivemap     Target LAS file to map source to
  footprint    Footprint for the source LAS file
  output       file to write output LAS to

Options:
  -d <sample>  Downsample source pointcloud to a percentage of number of points
               [default: 1.0].
  -U           Trust the upvector completely and dont estimate it in this script, too
  -u <upfile>  Json file containing the up vector relative to the pointcloud.
  -c <camfile> CSV file containing all the camera postionions.
  

"""

from __future__ import print_function
from docopt import docopt

from pcl.registration import icp
import numpy as np
import os
import json
from patty.utils import (load, save, log, BoundingBox)
from patty.srs import (set_srs, force_srs, same_srs)

from patty.segmentation import (
    boundary_of_center_object,
    boundary_of_drivemap,
    boundary_of_lowest_points,
    )

from patty.registration import (
    align_footprints,
    estimate_pancake_up,
    get_stick_scale,
    rotate_upwards,
    )




def initial_registration(pointcloud, up, drivemap, trust_up=False):
    """
    Initial registration adds an spatial reference system to the pointcloud,
    and place the pointlcoud on top of the drivemap. The pointcloud is rotated
    so that the up vector points along [0,0,1].

    Arguments:
        pointcloud : pcl.PointCloud
            The high-res object to register.

        up: np.array([3])
            Up direction for the pointcloud. 
            If None, assume the object is pancake shaped, and chose the upvector such
            that it is perpendicullar to the pancake. 

        drivemap : pcl.PointCloud
            A small part of the low-res drivemap on which to register.

        trust_up : Boolean
            True:  Assume the up vector is exact.
            False: Calculate 'up' as if it was None, but orient it such that
                   np.dot( up, pancake_up ) > 0

    NOTE: Modifies the input pointcloud in-place, and leaves
    it in a undefined state.

    """
    #####
    # set scale and offset of pointcloud, drivemap, and footprint
    # as the pointcloud is unregisterd, the coordinate system is undefined,
    # and we lose nothing if we just copy it

    if( hasattr(pointcloud, "offset") ):
        log( "Dropping initial offset, was: %s" % pointcloud.offset)
    else:
        log( "No initial offset" )
    force_srs(pointcloud, same_as=drivemap)
    log( "New offset forced to: %s" % pointcloud.offset )

    if up is not None:
        log( "Rotating the pointcloud so up points along [0,0,1]" )
        # rotate_upwards(pointcloud, up)

        if trust_up:
            rotate_upwards(pointcloud, up)
        else:
            pancake_up = estimate_pancake_up(pointcloud)
            if np.dot( up, pancake_up ) < 0.0:
                pancake_up *= -1.0
            rotate_upwards(pointcloud, pancake_up)

    else:
        log( "No upvector, skipping" )


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
    ###
    # find redstick scale, and use it if possible
    log("Redstick scaling")

    scale, confidence = get_stick_scale(pointcloud)
    log("Red stick scale=%s confidence=%s" % (scale, confidence))

    allow_scaling = True
    if (confidence > 0.5):
        log("Applying red stick scale")
        pointcloud.scale(1.0 / scale)  # dont care about origin of scaling
        allow_scaling = False
    else:
        log("Not applying red stick scale, confidence too low")
        bbDrivemap = BoundingBox( points=np.asarray( drivemap ) )
        bbObject   = BoundingBox( points=np.asarray( pointcloud ) )
        scale = bbDrivemap.size / bbObject.size

        # take the average scale factor for all dimensions
        scale = np.mean(scale)
        log("Applying rough estimation of scale factor", scale )
        pointcloud.scale(scale)  # dont care about origin of scaling

        allow_scaling = True

    #####
    # find all the points in the drivemap along the footprint
    # use bottom two meters of drivemap (not trees)

    fixed_boundary = boundary_of_drivemap(drivemap, footprint)
    save( fixed_boundary, "fixed_bound.las" )

    #####
    # find all the boundary points of the pointcloud

    loose_boundary = boundary_of_center_object(pointcloud, downsample)
    if loose_boundary is None:
        log(" - boundary estimation failed, using lowest 30 percent of points" )
        loose_boundary = boundary_of_lowest_points(pointcloud, height_fraction=0.3 )

    ####
    # match the pointcloud boundary with the footprint boundary

    log( "Aligning footprints" )
    rot_matrix, rot_center, scale, translation = align_footprints(
        loose_boundary, fixed_boundary,
        allow_scaling=allow_scaling,
        allow_rotation=True,
        allow_translation=True)

    ####
    # Apply to the main pointcloud

    pointcloud.rotate(rot_matrix, origin=rot_center)
    pointcloud.scale(scale, origin=rot_center)
    pointcloud.translate(translation)

    save( loose_boundary, "aligned_bound.las" )
    save( pointcloud, 'pre_icp.las' )


def _fine_registration_helper(pointcloud, basemap):
    '''
    Perform icp on pointcloud with basemap, and return convergence indicator.
    Reject large translatoins.

    Returns:
        transf : np.array([4,4])
            transform
        success : Boolean
            if icp was successful
        fitness : float
            sort of sum of square differences, ie. smaller is better 
            
    ''' 
    converged, transf, estimate, fitness = icp(pointcloud, drivemap)

    # dont accept large translations
    translation = transf[1:3,3]
    if np.dot( translation, translation ) > 10 ** 2:
        converged = False
        fitness = 1e30

    return transf, converged, fitness

def fine_registration(pointcloud, drivemap):
    '''
    Final registration step using ICP.

    Find the local optimal postion of the pointcloud on the drivemap; due to
    our coarse_registration algorithm, we have to try two orientations:
    original, and rotated by 180 degrees around the z-axis.

    Arguments:
        pointcloud: pcl.PointCloud
                    The high-res object to register.

        drivemap:   pcl.PointCloud
                    A small part of the low-res drivemap on which to register

    '''

    # for rotation around z-axis
    rot=np.array([[-1,0,0],[0,-1,0],[0,0,1]])

    log("Starting registration pipeline")

    ####
    # do a ICP step

    log("ICP 1" )
    transfA, convergedA, fitnessA = icp(pointcloud, drivemap)

    pointcloud.rotate( rot )

    log("ICP 2" )
    transfB, convergedB, fitnessB = icp(pointcloud, drivemap)

    # pick best
    if fitnessB < fitnessA and convergedB:
        pointcloud.transform( transfB )
        return 

    # undo rotation
    pointcloud.rotate( rot )

    if fitnessA < fitnessB and convergedA:
        pointcloud.transform( transfA )
        return estimateA

    # ICP failed:
    # return the pointcloud with just footprints aligned
    pointcloud.rotate( rot )

    return pointcloud


if __name__ == '__main__':

    ####
    # Parse comamnd line arguments

    args = docopt(__doc__)

    sourcefile = args['<source>']
    drivemapfile = args['<drivemap>']
    footprintcsv = args['<footprint>']
    foutLas = args['<output>']
    up_file = args['-u']
    cam_file = args['-c']
    trust_up = args['-U']
    downsample = float(args['-d'])

    assert os.path.exists(sourcefile),   sourcefile + ' does not exist'
    assert os.path.exists(drivemapfile), drivemapfile + ' does not exist'
    assert os.path.exists(footprintcsv), footprintcsv + ' does not exist'

    #####
    # Setup * the low-res drivemap
    #       * footprint
    #       * pointcloud
    #       * up-vector

    log("Reading drivemap", drivemapfile)
    drivemap = load(drivemapfile)
    force_srs(drivemap, srs="EPSG:32633")


    log("Reading footprint", footprintcsv)
    footprint = load(footprintcsv)
    force_srs( footprint, srs="EPSG:32633" )
    set_srs( footprint, same_as=drivemap )


    log("Reading object", sourcefile)
    pointcloud = load(sourcefile)


    log("Reading up_file", up_file)
    up = None
    try:
        with open(up_file) as f:
            dic = json.load(f)
        up = np.array(dic['estimatedUpDirection'])
        log( "Up vector is: %s" % up)
    except:
        log( "Cannot parse upfile, aborting" )

    initial_registration(pointcloud, up, drivemap, trust_up=trust_up)
    coarse_registration(pointcloud, drivemap, footprint, downsample)
    fine_registration(pointcloud, drivemap)

    save( pointcloud, foutLas )

