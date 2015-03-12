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
import argparse
import pcl.registration
import time
import os
import sys
from patty.conversions import loadLas, writeLas, loadCsvPolygon, copy_registration, extract_mask
from patty.registration import registration, principalComponents
from patty.segmentation import largest_dbscan_cluster
from patty.registration.stickScale import getPreferredScaleFactor
from patty.utils import BoundingBox

def log(*args, **kwargs):
    print(time.strftime("[%H:%M:%S]"), *args, **kwargs)

def process_args():
    """ Parse arguments from the command-line using argparse """
    args = docopt(__doc__)

    sourceLas    = args['<SOURCE>']
    drivemapLas  = args['<DRIVEMAP>']
    footprintCsv = args['<FOOTPRINT>']
    foutLas      = args['<OUTPUT>']

    return sourceLas, drivemapLas, footprintCsv, foutLas

def registrationPipeline(sourceLas, drivemapLas, footprintCsv, f_out):
    """Single function wrapping whole script, so it can be unit tested"""
    assert os.path.exists(sourceLas),sourceLas + ' does not exist'
    assert os.path.exists(drivemapLas),drivemapLas + ' does not exist'
    assert os.path.exists(footprintCsv),footprintCsv + ' does not exist'

    log("reading source", sourceLas)
    pointcloud = loadLas(sourceLas)
    log("reading drivemap ", drivemapLas)
    drivemap = loadLas(drivemapLas)
    footprint = loadCsvPolygon(footprintCsv)

    # Footprint is off by some meters
    footprint[:,0] += -1.579381346780
    footprint[:,1] += 0.52519696509

    drivemap_array = np.asarray(drivemap) + drivemap.offset

    # Get the pointcloud of the drivemap within the footprint
    in_footprint = registration.point_in_polygon2d(drivemap_array, footprint)
    footprint_drivemap = extract_mask(drivemap, in_footprint)

    # Get a boundary around the drivemap footprint
    large_footprint = registration.scale_points(footprint, 2)
    in_large_footprint = registration.point_in_polygon2d(drivemap_array, large_footprint)
    footprint_boundary = extract_mask(drivemap, in_large_footprint & np.invert(in_footprint))

    log("Finding largest cluster")
    cluster = largest_dbscan_cluster(pointcloud, .15, 250)

    log(cluster.offset)
    boundary_bb = BoundingBox(points=cluster)
    log(boundary_bb)

    log("Detecting boundary")
    search_radius = boundary_bb.diagonal / 100.0
    boundary = registration.get_pointcloud_boundaries(cluster, search_radius=search_radius, normal_search_radius=search_radius)
    print(len(boundary))

    if len(boundary) == len(cluster) or len(boundary) == 0:
        # DISCARD BOUNDARY INFORMATION
        log("Boundary information could not be retrieved")
        sys.exit(1)
    else:
        log("Finding rotation:")
        transform = registration.find_rotation(boundary, footprint_boundary)
        log(transform)

        log("Rotating pointcloud...")
        boundary.transform(transform)
        cluster.transform(transform)
        pointcloud.transform(transform)

        log("Calculating scale and shift from pointcloud boundary to footprint")
        registered_offset, registered_scale = registration.register_offset_scale_from_ref(boundary, footprint)
        registered_scale = getPreferredScaleFactor(pointcloud, registered_scale)

        log("Scaling pointcloud: %f" % registered_scale)
        pc_array = np.asarray(pointcloud)
        pc_array *= registered_scale
        cluster_array = np.asarray(cluster)
        cluster_array *= registered_scale

        log("Adding offset:")
        copy_registration(pointcloud, boundary)
        copy_registration(cluster, boundary)
        log(pointcloud.offset)

    # set the right height
    # footprint_drivemap_array = np.asarray(footprint_drivemap)[2]
    # pc_array = np.asarray(cluster)[2]
    # ref_boundary_height = (footprint_drivemap_array.min() + footprint_drivemap_array.max())/2.0 + footprint_drivemap.offset[2]
    # register(pointcloud, offset=[pointcloud.offset[0], pointcloud.offset[1], ref_boundary_height])

    log("Writing output")
    writeLas(f_out, pointcloud)
    writeLas(f_out + ".cluster.las", cluster)
    writeLas(f_out + ".boundary.las", boundary)

if __name__ == '__main__':
    sourceLas, drivemapLas, footprintCsv, foutLas = process_args()
    registrationPipeline(sourceLas, drivemapLas, footprintCsv, foutLas)
