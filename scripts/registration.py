#!/usr/bin/env python2.7
"""Registration script.

Usage:
  registration.py [-h] [-o <dir>] [-d <sample>] <source> <drivemap> <footprint> <output>

Positional arguments:
  source       Source LAS file
  drivemap     Target LAS file to map source to
  footprint    Footprint for the source LAS file
  output       file to write output LAS to

Options:
  -o <dir>     Output directory for translation and transformation matrix.
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
                                point_in_polygon2d, downsample, is_upside_down)
from patty.segmentation.dbscan import get_largest_dbscan_clusters
from patty.registration.stickscale import get_preferred_scale_factor


def log(*args, **kwargs):
    print(time.strftime("[%H:%M:%S]"), *args, **kwargs)


def registration_pipeline(sourcefile, drivemapfile, footprintCsv, f_out,
                          f_outdir, upfile=None, sample=-1):
    """Single function wrapping whole script, so it can be unit tested"""
    assert os.path.exists(sourcefile), sourcefile + ' does not exist'
    assert os.path.exists(drivemapfile), drivemapfile + ' does not exist'
    assert os.path.exists(footprintCsv), footprintCsv + ' does not exist'

    log("reading source", sourcefile)
    pointcloud = load(sourcefile)
    log("reading drivemap ", drivemapfile)
    drivemap = load(drivemapfile)
    footprint = load_csv_polygon(footprintCsv)

    if f_outdir is None:
        f_outdir = os.path.dirname(f_out)

# Footprint is NO LONGER off by some meters
# footprint[:, 0] += -1.579381346780
# footprint[:, 1] += 0.52519696509

    drivemap_array = np.asarray(drivemap) + drivemap.offset

    # Get the pointcloud of the drivemap within the footprint
    in_footprint = point_in_polygon2d(drivemap_array, footprint)

    # Get a boundary around the drivemap footprint
    large_footprint = scale_points(footprint, 2)
    in_large_footprint = point_in_polygon2d(drivemap_array, large_footprint)
    footprint_boundary = extract_mask(
        drivemap, in_large_footprint & np.invert(in_footprint))

    log("Finding largest cluster")
    if sample != -1 and len(pointcloud) > sample:
        fraction = float(sample) / len(pointcloud)
        log("downsampling from %d to %d points (%d%%) for registration" % (
            len(pointcloud), sample, int(fraction * 100)
        ))
        pc = downsample(pointcloud, fraction, random_seed=0)
    else:
        pc = pointcloud
    cluster = get_largest_dbscan_clusters(pc, 0.7, .15, 250)

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

        print(is_upside_down(upfile, transform))

        with open(os.path.join(f_outdir, 'rotation.csv'), 'w') as f:
            for row in transform:
                print(','.join(np.char.mod('%f', row)), file=f)
        with open(os.path.join(f_outdir, 'rotation_offset.csv'), 'w') as f:
            print(','.join(np.char.mod('%f', pointcloud.offset)), file=f)

        log("Calculating scale and shift from boundary to footprint")
        registered_offset, registered_scale = \
            register_offset_scale_from_ref(boundary, footprint)
        with open(os.path.join(f_outdir, 'translation.csv'), 'w') as f:
            print(','.join(np.char.mod('%f',
                                       registered_offset - pointcloud.offset)), file=f)

        registered_scale = get_preferred_scale_factor(pointcloud,
                                                      registered_scale)

        with open(os.path.join(f_outdir, 'scaling_factor.csv'), 'w') as f:
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


if __name__ == '__main__':
    args = docopt(__doc__)

    sourcefile = args['<source>']
    drivemapfile = args['<drivemap>']
    footprintCsv = args['<footprint>']
    foutLas = args['<output>']
    foutDir = args['-o']
    up_file = args['-u']
    sample = int(args['-d'])

    registration_pipeline(sourcefile, drivemapfile, footprintCsv, foutLas,
                          foutDir, up_file, sample)
