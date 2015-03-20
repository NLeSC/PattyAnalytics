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


def find_largest_cluster(pointcloud, sample):
    log("Finding largest cluster")
    if sample != -1 and len(pointcloud) > sample:
        fraction = float(sample) / len(pointcloud)
        log("downsampling from %d to %d points (%d%%) for registration" % (
            len(pointcloud), sample, int(fraction * 100)))
        pc = downsample(pointcloud, fraction, random_seed=0)
    else:
        pc = pointcloud
    return get_largest_dbscan_clusters(pc, 0.7, .15, 250)


def cutout_edge(pointcloud, polygon2d, polygon_width):
    pc_array = np.asarray(pointcloud) + pointcloud.offset

    slightly_large_polygon = scale_points(polygon2d, 1.05)
    in_polygon = point_in_polygon2d(pc_array, slightly_large_polygon)

    large_polygon = scale_points(polygon2d, polygon_width)
    in_large_polygon = point_in_polygon2d(pc_array, large_polygon)
    return extract_mask(pointcloud,
                        in_large_polygon & np.invert(in_polygon))


def registration_pipeline(sourcefile, drivemapfile, footprintcsv, f_out,
                          f_outdir, upfile=None, sample=-1):
    """Single function wrapping whole script, so it can be unit tested"""
    assert os.path.exists(sourcefile), sourcefile + ' does not exist'
    assert os.path.exists(drivemapfile), drivemapfile + ' does not exist'
    assert os.path.exists(footprintcsv), footprintcsv + ' does not exist'

    #####
    # Setup * the low-res drivemap
    #       * footprint
    #       * pointcloud

    log("reading drivemap ", drivemapfile)
    drivemap = load(drivemapfile)

    log("reading footprint ", footprintcsv )
    footprint = load_csv_polygon(footprintcsv)

    log("reading source", sourcefile)
    pointcloud = load(sourcefile)


    #####
    # find all the points in the drivemap along the footprint
    # use bottom two meters of drivemap (not trees)

    drivemap_array = np.asarray(drivemap)
    bb = BoundingBox(points=drivemap_array)
    if bb.size[2] > bb.size[1] or bb.size[2] > bb.size[0]:
        drivemap = extract_mask(drivemap, drivemap_array[:, 2] < bb.min[2] + 2)

    footprint_boundary = cutout_edge(drivemap, footprint, 1.5)


    #####
    # find all the points in the pointcloud that would
    # correspond to the footprint_boundary

    cluster = find_largest_cluster(pointcloud, sample)
    boundary = get_pointcloud_boundaries( pointcloud )

    if len(boundary) == len(cluster) or len(boundary) == 0:
        # DISCARD BOUNDARY INFORMATION
        log("Boundary information could not be retrieved")
        print('BoundaryLen:', len(boundary))
        print('ClusterLen:', len(cluster))
        sys.exit(1)


    ####
    # match the pointcloud boundary with the footprint boundary

    log("Finding rotation:")
    transform = find_rotation(boundary, footprint_boundary)
    log(transform)

    if is_upside_down(upfile, transform[:3, :3]):
        transform = np.dot( np.eye(4)*[1,-1,-1,1], transform)

    ####
    # apply the rotation

    log("Rotating pointcloud...")
    boundary.transform(transform)
    cluster.transform(transform)
    pointcloud.transform(transform)




    # construct output file dir/basename
    if f_outdir is None:
        f_out = os.path.abspath( f_out )
    else:
        f_out = os.path.join( f_outdir, f_out )

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

    boundary_zmean = np.asarray(boundary)[:, 2].mean()
    boundary_zmean += boundary.offset[2]
    footprint_zmean = np.asarray(footprint_boundary)[:, 2].mean()
    footprint_zmean += footprint_boundary.offset[2]
    boundary.offset[2] += footprint_zmean - boundary_zmean

    ####
    # possibly update scalefactor when the redstick detection works
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
    footprintcsv = args['<footprint>']
    foutLas = args['<output>']
    up_file = args['-u']
    sample = int(args['-d'])

    registration_pipeline(sourcefile, drivemapfile, footprintcsv, foutLas,
                          None, up_file, sample)
