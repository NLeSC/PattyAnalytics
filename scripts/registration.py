#!/usr/bin/env python2.7
"""Registration script.

Usage:
  registration.py [-h] [-d <sample>] [-u <upfile>] [-c <camfile>] <source> <drivemap> <footprint> <output>

Positional arguments:
  source       Source LAS file
  drivemap     Target LAS file to map source to
  footprint    Footprint for the source LAS file
  output       file to write output LAS to

Options:
  -d <sample>  Downsample source pointcloud to a percentage of number of points
               [default: 1.0].
  -u <upfile>  Json file containing the up vector relative to the pointcloud.
  -c <camfile> CSV file containing all the camera postionions.

"""

from __future__ import print_function
from docopt import docopt

from pcl.registration import icp
import numpy as np
import os
import json
from patty.conversions import (load, save, log, BoundingBox)
from patty.srs import (set_srs, force_srs, same_srs)
from patty.registration import (boundary_of_drivemap,
                                boundary_of_center_object,
                                boundary_via_lowest_points,
                                register_from_footprint,
                                rotate_upwards)
from patty.registration.stickscale import get_stick_scale



def registration_pipeline(pointcloud, up, drivemap, footprint, downsample=None):
    """Full registration pipeline for the Via Appia pointclouds.

    Arguments:
        pointcloud: pcl.PointCloud
                    The high-res object to register.

        up: np.array([3])
                    Up direction for the pointcloud

        drivemap:   pcl.PointCloud
                    A small part of the low-res drivemap on which to register

        footprint:  pcl.PointCloud
                    Pointlcloud containing the objects footprint

        downsample: float, default=None, no resampling
                    Downsample the high-res pointcloud before ICP step

    Returns:
        pc : pcl.PointCloud
                    The input pointcloud, but now registered.

    NOTE: Modifies the input pointcloud in-place, and leaves
    it in a undefined state.
    """

    log("Starting registration pipeline")

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
        rotate_upwards(pointcloud, up)
    else:
        log( "No upvector, skipping" )

    ###
    # find redstick scale, and use it if possible
    log("Redstick scaling")

    scale, confidence = get_stick_scale(pointcloud)
    log("Red stick scale=%s confidence=%s" % (scale, confidence))

    allow_scaling = True
    if (confidence > 0.5):
        log("Applying red stick scale")
        pointcloud.scale(scale)  # dont care about origin of scaling
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
    log("Estimating footprint from drivemap")

    fixed_boundary = boundary_of_drivemap(drivemap, footprint)

    save( fixed_boundary, "fixed_bound.las" )

    #####
    # find all the boundary points of the pointcloud

    log("Estimating footprint from pointcloud")

    loose_boundary = boundary_of_center_object(pointcloud, downsample)

    if loose_boundary is None:
        log(" - boundary estimation failed, using lowest 30 percent of points" )
        loose_boundary = boundary_via_lowest_points(pointcloud, height_fraction=0.3 )

    log(" - Number of boundary points found: %d" % len(loose_boundary) )

    save( loose_boundary, "loose_bound.las" )


    ####
    # match the pointcloud boundary with the footprint boundary

    log( "Aligning footprints" )
    rot_matrix, rot_center, scale, translation = register_from_footprint(
        loose_boundary, fixed_boundary,
        allow_scaling=allow_scaling,
        allow_rotation=True,
        allow_translation=True)

    log("Applying initial alignment to pointcloud:")
    log("rotate_center:                   %s" % rot_center )
    log("rotate_matrix:\n%s" % rot_matrix )
    log("scale (around rotate_center):    %s" % scale )
    log("translate                        %s" % translation )
    pointcloud.rotate(rot_matrix, origin=rot_center)
    pointcloud.scale(scale, origin=rot_center)
    pointcloud.translate(translation)

    # FIXME: debug output
    save( pointcloud, 'pre_icp.las' )

    ####
    # do a ICP step

    log("Starting ICP")

    log("Trying original orientation")
    convergedA, transfA, estimateA, fitnessA = icp(pointcloud, drivemap)
    force_srs( estimateA, same_as=pointcloud )

    log("converged: %s" % convergedA)
    log("fitness: %s" % fitnessA)
    log("transf :\n%s" % transfA)

    is_good_transA = True

    # dont accept large translations
    transA = transfA[1:3,3]
    if np.dot( transA, transA ) > 10 ** 2:
        is_good_transA = False


    rot=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    log("Trying rotated around z-axes, rotation:\n%s" % rot )

    pointcloud.rotate( rot )
    convergedB, transfB, estimateB, fitnessB = icp(pointcloud, drivemap)
    force_srs( estimateB, same_as=pointcloud )

    log("converged: %s" % convergedB)
    log("fitness: %s" % fitnessB)
    log("transf :\n%s" % transfB)

    is_good_transB = True

    # dont accept large translations
    transB = transfB[1:3,3]
    if np.dot( transB, transB ) > 10 ** 2:
        is_good_transB = False

    if fitnessB < fitnessA and is_good_transB:
        return estimateB

    if fitnessA < fitnessB and is_good_transA:
        return estimateA

    # undo rotation, and return the pointcloud with just footprints aligned
    pointcloud.rotate( rot )
    return pointcloud

def short_srs(pc):
    """Pretty print the SRS as EPSG:?"""
    try:
        s = "%s" % pc.srs
        return "OK"
    except:
        return "None"


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


    pointcloud = registration_pipeline(pointcloud, up, drivemap, footprint, downsample)

    log( "Saving to", foutLas)
    save(pointcloud, foutLas)

    log("Finished")
