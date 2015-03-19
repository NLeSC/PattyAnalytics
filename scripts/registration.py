#!/usr/bin/env python2.7
"""Registration script.

Usage:
  registration.py [-h] [-d <sample>] <source> <drivemap> <footprint> <output>

Positional arguments:
  source       Source LAS file
  drivemap     Target LAS file to map source to
  footprint    Footprint for the source LAS file
  output       file to write output LAS to

Options:
  -d <sample>  Downsample source pointcloud to a maximum of <sample> points
               [default: -1].
"""

from __future__ import print_function
from docopt import docopt

import numpy as np
import time
import os
import sys
from patty.conversions import (load, save, load_csv_polygon,
                               copy_registration, extract_mask, BoundingBox)
from patty.registration import (get_pointcloud_boundaries, find_rotation,
                                register_offset_scale_from_ref, scale_points,
                                point_in_polygon2d, downsample)
from patty.segmentation.dbscan import get_largest_dbscan_clusters
from patty.registration.stickscale import get_preferred_scale_factor


def log(*args, **kwargs):
    print(time.strftime("[%H:%M:%S]"), *args, **kwargs)


def find_largest_cluster(pointcloud, sample):
    if sample != -1 and len(pointcloud) > sample:
        fraction = float(sample) / len(pointcloud)
        log("downsampling from %d to %d points (%d%%) for registration" % (
            len(pointcloud), sample, int(fraction * 100)
        ))
        pc = downsample(pointcloud, fraction, random_seed=0)
    else:
        pc = pointcloud
    return get_largest_dbscan_clusters(pc, 0.7, .15, 250)


def detect_boundary(pointcloud):
    log("Detecting boundary")
    boundary_bb = BoundingBox(points=pointcloud)
    search_radius = boundary_bb.diagonal / 100.0
    return get_pointcloud_boundaries(
        pointcloud, search_radius=search_radius,
        normal_search_radius=search_radius)


def cutout_edge(pointcloud, polygon2d, polygon_width):
    pc_array = np.asarray(pointcloud) + pointcloud.offset

    slightly_large_polygon = scale_points(polygon2d, 1.05)
    in_polygon = point_in_polygon2d(pc_array, slightly_large_polygon)

    large_polygon = scale_points(polygon2d, polygon_width)
    in_large_polygon = point_in_polygon2d(pc_array, large_polygon)
    return extract_mask(pointcloud,
                        in_large_polygon & np.invert(in_polygon))


def registration_pipeline(sourcefile, drivemapfile, footprintCsv, f_out,
                          f_outdir, sample=-1):
    """Single function wrapping whole script, so it can be unit tested"""
    assert os.path.exists(sourcefile), sourcefile + ' does not exist'
    assert os.path.exists(drivemapfile), drivemapfile + ' does not exist'
    assert os.path.exists(footprintCsv), footprintCsv + ' does not exist'

    log("reading source", sourcefile)
    pointcloud = load(sourcefile)
    log("reading drivemap ", drivemapfile)
    drivemap = load(drivemapfile)

    drivemap_array = np.asarray(drivemap)
    bb = BoundingBox(points=drivemap_array)
    # use bottom two meters of drivemap (not trees)
    if bb.size[2] > bb.size[1] or bb.size[2] > bb.size[0]:
        drivemap = extract_mask(drivemap, drivemap[:,2] < bb.min[3] + 2)
    
    footprint = load_csv_polygon(footprintCsv)

    if f_outdir is None:
        f_outdir = os.path.dirname(f_out)

# Footprint is NO LONGER off by some meters
# footprint[:, 0] += -1.579381346780
# footprint[:, 1] += 0.52519696509
    footprint_boundary = cutout_edge(drivemap, footprint, 1.5)

    log("Finding largest cluster")
    cluster = find_largest_cluster(pointcloud, sample)

    log("Detecting boundary")
    boundary = detect_boundary(cluster)

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
        rotate180 = np.eye((4,4))
        rotate180[1,1] = rotate180[2,2] = -1

        # TODO: detect up/down 
        upIsDown = False
        if upIsDown:
            transform = np.dot(rotate180, transform)
        
        log("Rotating pointcloud...")
        boundary.transform(transform)
        cluster.transform(transform)
        pointcloud.transform(transform)

        with open(f_out + '.rotation.csv', 'w') as f:
            for row in transform:
                print(','.join(np.char.mod('%f', row)), file=f)
        with open(f_out + '.rotation_offset.csv', 'w') as f:
            print(','.join(np.char.mod('%f', pointcloud.offset)), file=f)

        log("Calculating scale and shift from boundary to footprint")
        registered_offset, registered_scale = \
            register_offset_scale_from_ref(boundary, footprint)
        with open(f_out + '.translation.csv', 'w') as f:
            str_arr = np.char.mod('%f', registered_offset - pointcloud.offset)
            print(','.join(str_arr), file=f)

        registered_scale = get_preferred_scale_factor(pointcloud,
                                                      registered_scale)

        with open(f_out + '.scaling_factor.csv', 'w') as f:
            print(registered_scale, file=f)

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
    save(footprint_boundary, f_out + ".footboundary.las")


if __name__ == '__main__':
    args = docopt(__doc__)

    sourcefile = args['<source>']
    drivemapfile = args['<drivemap>']
    footprintCsv = args['<footprint>']
    foutLas = args['<output>']
    foutDir = args['-o']
    sample = int(args['-d'])

    registration_pipeline(sourcefile, drivemapfile, footprintCsv, foutLas,
                          foutDir, sample)
