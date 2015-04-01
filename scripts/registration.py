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
  -d <sample>  Downsample source pointcloud to a maximum of <sample> points
               [default: -1].
  -u <upfile>  Json file containing the up vector relative to the pointcloud.
"""

from __future__ import print_function
from docopt import docopt

from pcl.registration import icp
import numpy as np
import os
from patty.conversions import (load, save, clone,
                               set_srs, force_srs, same_srs,
                               extract_mask, BoundingBox,log)
from patty.registration import (register_from_footprint,
                                point_in_polygon2d, downsample_random)
from patty.segmentation.dbscan import get_largest_dbscan_clusters
from patty.registration.stickscale import get_stick_scale


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
    Modifies the pointcloud in-place.

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
        Nothing, pointcloud is modified in-place
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

    rot_matrix, rot_center, scale, translation = register_from_footprint(
        pointcloud, footprint_boundary,
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

    ####
    # do a ICP step

    log("Starting ICP")
    converged, transf, estimate, fitness = icp(pointcloud, drivemap)

    log("converged: %s" % converged)
    log("fitness: %s" % fitness)
    log("transf :\n%s" % transf)

    log("Applying ICP transform to pointcloud")
    pointcloud.transform( transf )


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

    log("reading footprint", footprintcsv)
    footprint = load(footprintcsv, srs="EPSG:32633", offset=[0, 0, 0]) # FIXME: set to via appia projection

    log("reading drivemap", drivemapfile)
    drivemap = load(drivemapfile, same_as=footprint)

    log("reading source", sourcefile)
    pointcloud = load(sourcefile )

    # TODO: use up_file to orient the pointcloud upwards

    registration_pipeline(pointcloud, drivemap, footprint, sample)

    log( "Saving to", foutLas)
    save(pointcloud, foutLas)

    log("Finished")
