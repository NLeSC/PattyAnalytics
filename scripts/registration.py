#!/usr/bin/env python2.7
"""Registration script.

Usage:
  registration.py [-h] [-d <sample>] [-u <upfile>] <source> <drivemap> <footprint> <output>

Positional arguments:
  source       Source LAS file
  drivemap     Target LAS file to map source to
  footprint    Footprint for the source LAS file
  output       file to write output LAS to

Options:
  -d <sample>  Downsample source pointcloud to a percentage of number of points
               [default: -1].
  -u <upfile>  Json file containing the up vector relative to the pointcloud.
"""

from __future__ import print_function
from docopt import docopt

from pcl.registration import icp
import numpy as np
import os
import json
from patty.conversions import (load, save, clone,
                               set_srs, force_srs, same_srs,
                               extract_mask, BoundingBox,log)
from patty.registration import (register_from_footprint,
                                point_in_polygon2d, downsample_random)
from patty.segmentation.dbscan import get_largest_dbscan_clusters
from patty.registration.stickscale import get_stick_scale


def fix_upside_down(up_file, pc):
    '''Rotates a pointcloud such that the given vector is up, ie. along [0,0,1].
    The pointcloud is flipped in-place.

    Arguments:
        up_file filename of the json file containing the relative up vector
        pc : pcl.PointCloud

    Returns:
        pc : pcl.PointCloud the input pointcloud, for convenience.

    '''
    if up_file in (None, ''):
        log( "No upfile, aborting" )
        return pc

    newz = np.array([0,0,1])
    try:
        with open(up_file) as f:
            dic = json.load(f)
        newz = np.array(dic['estimatedUpDirection'])
        log( "Up vector is: %s" % newz )
    except:
        log( "Cannot parse upfile, aborting" )
        return pc

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

    pc.rotate( rotation, origin=pc.center() )
    log( "Rotating pointcloud around origin, using:\n%s" % rotation )

    return pc

def find_largest_cluster(pointcloud, sample):
    log("Finding largest cluster")
    if sample != -1 and len(pointcloud) > sample:
        fraction = float(sample) / len(pointcloud)
        log("downsampling from %d to %d points (%d%%) for registration" % (
            len(pointcloud), sample, int(fraction * 100)))

        pc = downsample_random(pointcloud, fraction, random_seed=0)
    else:
        pc = pointcloud
    return get_largest_dbscan_clusters(pc, 0.7, .15, 250)


def cutout_edge(pointcloud, polygon2d, edge_width):
    """Cut boundary of pointcloud. Edge is considered to be a band around
    the given polygon2d of a given edge width.

    Arguments:
        pointcloud: pcl.PointCloud
                        Source point cloud
        polygon2d:  pcl.PointCloud
                        Boundary of edge to cut out
        edge_width: float
                        Width of the edge to cut out, relative to the scale
                        of the point cloud
    """

    # FIXME: will give overflow in many cases;
    # caller should make sure pointcloud and polygon have the same registration
    pc_array = np.asarray(pointcloud) + pointcloud.offset

    center = polygon2d.center()
    slightly_large_polygon = clone(polygon2d).scale(1.05, origin=center)
    large_polygon = clone(polygon2d).scale(1.05 + edge_width, origin=center)

    in_polygon = point_in_polygon2d(pc_array, slightly_large_polygon)
    in_large_polygon = point_in_polygon2d(pc_array, large_polygon)

    # Cut band between 1.05 and 1.05 + edge_width around the polygon
    return extract_mask(pointcloud, in_large_polygon & np.invert(in_polygon))


def registration_pipeline(pointcloud, drivemap, footprint, sample=-1):
    """Full registration pipeline for the Via Appia pointclouds.
    Modifies the input pointcloud in-place, and leaves it in a undefined state.

    Arguments:
        pointcloud: pcl.PointCloud
                    The high-res object to register

        drivemap:   pcl.PointCloud
                    A small part of the low-res drivemap on which to register

        footprint:  pcl.PointCloud
                    Pointlcloud containing the objects footprint

        sample: int, default=-1, no resampling
                    Downsample the high-res pointcloud before ICP step (UNIMPLEMENTED)

    Returns:
        pc : pcl.PointCloud
                    The input pointcloud, but now registered.
    """

    log("Aligning footprints")
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

    #####
    # find all the points in the drivemap along the footprint
    # use bottom two meters of drivemap (not trees)

    drivemap_array = np.asarray(drivemap)
    bb = BoundingBox(points=drivemap_array)
    if bb.size[2] > bb.size[1] or bb.size[2] > bb.size[0]:
        drivemap = extract_mask(drivemap, drivemap_array[:, 2] < bb.min[2] + 2)

    footprint_boundary = cutout_edge(drivemap, footprint, 0.25)

    ###
    # find redstick scale, and use it if possible
    scale, confidence = get_stick_scale(pointcloud)
    log("Red stick scale=%s confidence=%s" % (scale, confidence))

    allow_scaling = True
    if (confidence > 0.5):
        log("Applying red stick scale")
        pointcloud.scale(scale)  # dont care about origin of scaling
        allow_scaling = False
    else:
        log("Not applying red stick scale, confidence too low")
        allow_scaling = True

    ####
    # match the pointcloud boundary with the footprint boundary

    # Use downsampled pointcloud to speed up computation.
    if sample>0:
        log("Downsampling %f" % (sample/100.0))
        pc_sampled = downsample_random(pointcloud, sample / 100.0)
        log('  Points: %d -> %d'%(len(pointcloud),len(pc_sampled)))
    else:
        log("No downsampling")
        pc_sampled = pointcloud

    rot_matrix, rot_center, scale, translation = register_from_footprint(
        pc_sampled, footprint_boundary,
        allow_scaling=allow_scaling,
        allow_rotation=True,
        allow_translation=True)

    log("Applying initial alignment to pointcloud:")
    log("rotate_center:                   %s" % rot_center )
    log("rotate_matrix:\n%s" % rot_matrix )
    log("scale (around rotate_center):    %s" % scale )
    log("scale:                           %s" % scale )
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
    log("converged: %s" % convergedA)
    log("fitness: %s" % fitnessA)
    log("transf :\n%s" % transfA)

    # dont accept large translations
    transA = transfA[1:3,3]
    if np.dot( transA, transA ) > 10 ** 2:
        is_good_transA = False
    else:
        is_good_transB = True


    rot=np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    log("Trying rotated around z-axes, rotation:\n%s" % rot )

    pointcloud.rotate( rot )
    convergedB, transfB, estimateB, fitnessB = icp(pointcloud, drivemap)
    log("converged: %s" % convergedB)
    log("fitness: %s" % fitnessB)
    log("transf :\n%s" % transfB)

    # dont accept large translations
    transB = transfB[1:3,3]
    if np.dot( transB, transB ) > 10 ** 2:
        is_good_transB = False
    else:
        is_good_transB = True


    if fitnessB < fitnessA and is_good_transB:
        return estimateB

    if fitnessA < fitnessB and is_good_transA:
        return estimateA

    # undo rotation, and return the pointcloud with just footprints aligned
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
    sample = int(args['-d'])

    assert os.path.exists(sourcefile),   sourcefile + ' does not exist'
    assert os.path.exists(drivemapfile), drivemapfile + ' does not exist'
    assert os.path.exists(footprintcsv), footprintcsv + ' does not exist'

    #####
    # Setup * the low-res drivemap
    #       * footprint
    #       * pointcloud

    log("Reading footprint", footprintcsv)
    footprint = load(footprintcsv, srs="EPSG:32633", offset=[0, 0, 0]) # FIXME: set to via appia projection

    log("Reading drivemap", drivemapfile)
    drivemap = load(drivemapfile)
    force_srs(drivemap, srs="EPSG:32633")
    set_srs( drivemap, same_as=footprint)

    log("Reading object", sourcefile)
    pointcloud = load(sourcefile)

    if up_file is not None:
        log("Orient object right side up using '%s'" % up_file )
        fix_upside_down( up_file, pointcloud )
    else:
        log("No up-file given.")

    pointcloud = registration_pipeline(pointcloud, drivemap, footprint, sample)

    log( "Saving to", foutLas)
    save(pointcloud, foutLas)

    log("Finished")
