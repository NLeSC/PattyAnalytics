#!/usr/bin/env python2.7
"""Registration script.

Usage: registration.py [-h] <SOURCE> <DRIVEMAP> <FOOTPRINT> <OUTPUT>

Options:
  SOURCE     Source LAS file
  DRIVEMAP   Target LAS file to map source to
  FOOTPRINT  Footprint for the source LAS file
  OUTPUT     File to write output LAS to
"""

from __future__ import print_function
from docopt import docopt

import numpy as np
import time
import os
import sys
from patty.conversions import (load, save, loadCsvPolygon,
                               copy_registration, extract_mask)
from patty.registration import (get_pointcloud_boundaries, find_rotation,
                                register_offset_scale_from_ref, scale_points,
                                point_in_polygon2d)
from patty.segmentation.dbscan import get_largest_dbscan_clusters
from patty.registration.stickScale import getPreferredScaleFactor
from patty.utils import BoundingBox


def log(*args, **kwargs):
    print(time.strftime("[%H:%M:%S]"), *args, **kwargs)


def process_args():
    args = docopt(__doc__)

    sourceFile = args['<SOURCE>']
    drivemapFile = args['<DRIVEMAP>']
    footprintCsv = args['<FOOTPRINT>']
    foutLas = args['<OUTPUT>']

    return sourceFile, drivemapFile, footprintCsv, foutLas


def registrationPipeline(sourceFile, drivemapFile, footprintCsv, f_out):
    """Single function wrapping whole script, so it can be unit tested"""
    assert os.path.exists(sourceFile), sourceFile + ' does not exist'
    assert os.path.exists(drivemapFile), drivemapFile + ' does not exist'
    assert os.path.exists(footprintCsv), footprintCsv + ' does not exist'

    log("reading source", sourceFile)
    pointcloud = load(sourceFile)
    log("reading drivemap ", drivemapFile)
    drivemap = load(drivemapFile)
    footprint = loadCsvPolygon(footprintCsv)

    # Footprint is off by some meters
    footprint[:, 0] += -1.579381346780
    footprint[:, 1] += 0.52519696509

    drivemap_array = np.asarray(drivemap) + drivemap.offset

    # Get the pointcloud of the drivemap within the footprint
    in_footprint = point_in_polygon2d(drivemap_array, footprint)

    # Get a boundary around the drivemap footprint
    large_footprint = scale_points(footprint, 2)
    in_large_footprint = point_in_polygon2d(drivemap_array, large_footprint)
    footprint_boundary = extract_mask(
        drivemap, in_large_footprint & np.invert(in_footprint))

    log("Finding largest cluster")
    cluster = get_largest_dbscan_clusters(pointcloud, 0.7, .15, 250)

    log(cluster.offset)
    boundary_bb = BoundingBox(points=cluster)
    log(boundary_bb)

    log("Detecting boundary")
    search_radius = boundary_bb.diagonal / 100.0
    boundary = get_pointcloud_boundaries(
        cluster, search_radius=search_radius,
        normal_search_radius=search_radius)

    if len(boundary) == len(cluster) or len(boundary) == 0:
        # DISCARD BOUNDARY INFORMATION
        log("Boundary information could not be retrieved")
        print('BoundaryLen:', len(boundary))
        print('ClusterLen:', len(cluster))
        sys.exit(1)
    else:
        log("Finding rotation:")
        transform = find_rotation(boundary, footprint_boundary)
        log(transform)

        log("Rotating pointcloud...")
        boundary.transform(transform)
        cluster.transform(transform)
        pointcloud.transform(transform)

        log("Calculating scale and shift from boundary to footprint")
        registered_offset, registered_scale = \
            register_offset_scale_from_ref(boundary, footprint)
        registered_scale = getPreferredScaleFactor(pointcloud,
                                                   registered_scale)

        log("Scaling pointcloud: %f" % registered_scale)
        pc_array = np.asarray(pointcloud)
        pc_array *= registered_scale
        cluster_array = np.asarray(cluster)
        cluster_array *= registered_scale

        log("Adding offset:")
        copy_registration(pointcloud, boundary)
        copy_registration(cluster, boundary)
        log(pointcloud.offset)

# TODO: set the right height
# footprint_drivemap_array = np.asarray(footprint_drivemap)[2]
# pc_array = np.asarray(cluster)[2]
# ref_boundary_height = ((footprint_drivemap_array.min()
#                         + footprint_drivemap_array.max()) / 2.0
#                        + footprint_drivemap.offset[2])
# register(pointcloud, offset=[pointcloud.offset[0], pointcloud.offset[1],
#          ref_boundary_height])

    log("Writing output")
    save(pointcloud, f_out)
    save(cluster, f_out + ".cluster.las")
    save(boundary, f_out + ".boundary.las")


if __name__ == '__main__':
    sourceFile, drivemapFile, footprintCsv, foutLas = process_args()
    registrationPipeline(sourceFile, drivemapFile, footprintCsv, foutLas)
