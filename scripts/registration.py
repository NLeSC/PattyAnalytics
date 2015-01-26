#!/usr/bin/env python2.7

from __future__ import print_function
import numpy as np
import argparse
import pcl.registration
import time
import os.path
from patty.conversions import loadLas, writeLas, loadCsvPolygon, copy_registration, extract_mask
from patty.registration import registration
from patty.segmentation import dbscan

def log(*args, **kwargs):
    print(time.strftime("[%H:%M:%S]"), *args, **kwargs)

def process_args():
    """ Parse arguments from the command-line using argparse """

    # Implemented registration functions
    funcs = {
        'icp': pcl.registration.icp,
        'gicp': pcl.registration.gicp,
        'icp_nl': pcl.registration.icp_nl,
        'ia_ransac': pcl.registration.ia_ransac
    }

    parser = argparse.ArgumentParser(description='Registration for a LAS point clouds to another')
    parser.add_argument('-f','--function', choices=funcs.keys(), help='Advanced registration algorithm to run. Choose between gicp, icp, icp_nl, and ia_ransac.')
    parser.add_argument('source', metavar="SOURCE", help="Source LAS file")
    parser.add_argument('drivemap', metavar="DRIVEMAP", help="Target LAS file to map source to")
    parser.add_argument('footprint', metavar="FOOTPRINT", help="Footprint for the source LAS file")
    parser.add_argument('output', metavar="OUTPUT", help="File to write output LAS to")
    
    args = parser.parse_args()

    assert os.path.exists(args.source)
    assert os.path.exists(args.drivemap)
    assert os.path.exists(args.footprint)

    log("reading source", args.source)
    pointcloud = loadLas(args.source)
    log("reading drivemap ", args.drivemap)
    drivemap = loadLas(args.drivemap)
    footprint = loadCsvPolygon(args.footprint)
    
    if args.function is None:
        algo = None
    else:
        algo = funcs[args.function]
    
    return args, pointcloud, drivemap, footprint, args.output, algo

def bounding_box(pointcloud):
    arr = np.asarray(pointcloud)
    log(arr.min(axis=0), "to", arr.max(axis=0))

if __name__ == '__main__':
    args, pointcloud, drivemap, footprint, f_out, algo = process_args()
    
    # Footprint is off by some meters
    footprint[:,0] += -1.579381346780
    footprint[:,1] += 0.52519696509
    
    drivemap_array = np.asarray(drivemap) + drivemap.offset

    # Get the pointcloud of the drivemap within the footprint
    in_footprint  = registration.point_in_polygon2d(drivemap_array, footprint)
    footprint_drivemap = extract_mask(drivemap, in_footprint)

    # Get a boundary around the drivemap footprint
    large_footprint = registration.scale_points(footprint, 2)
    in_large_footprint = registration.point_in_polygon2d(drivemap_array, large_footprint)
    footprint_boundary = extract_mask(drivemap, in_large_footprint & np.invert(in_footprint))
    
    log("Finding largest cluster")
    cluster = dbscan.largest_dbscan_cluster(pointcloud, .15, 250)
    log(cluster.offset)
    bounding_box(cluster)
    
    log("Detecting boundary")
    boundary = registration.get_pointcloud_boundaries(cluster)
    
    log("Finding rotation")
    pc_transform = registration.principal_axes_rotation(np.asarray(boundary))
    log(pc_transform)
    # Rotate over Z, seems to work in our case...
    pc_transform[2] *= -1.
    fp_transform = registration.principal_axes_rotation(footprint)
    log(fp_transform)
    transform = np.linalg.inv(fp_transform) * pc_transform
    boundary.transform(transform)

    bounding_box(boundary)
    
    # FIXME: Correlate red sticks bounding box and scale to footprint found here
    
    log("Registering pointcloud boundary to footprint")
    registered_offset, registered_scale = registration.register_offset_scale_from_ref(boundary, footprint)
    copy_registration(pointcloud, boundary)
    copy_registration(cluster, boundary)

    bounding_box(boundary)
    bounding_box(pointcloud)
    log(pointcloud.offset)
    
    # rotate and scale up
    transform[:3,:3] *= registered_scale
    pointcloud.transform(transform)
    cluster.transform(transform)

    bounding_box(pointcloud)
    bounding_box(cluster)

    # set the right height
    # footprint_drivemap_array = np.asarray(footprint_drivemap)[2]
    # pc_array = np.asarray(cluster)[2]
    # ref_boundary_height = (footprint_drivemap_array.min() + footprint_drivemap_array.max())/2.0 + footprint_drivemap.offset[2]
    # register(pointcloud, offset=[pointcloud.offset[0], pointcloud.offset[1], ref_boundary_height])
    
    log("Writing output")
    writeLas(f_out, pointcloud)
    writeLas(f_out + ".cluster.las", cluster)
    writeLas(f_out + ".boundary.las", boundary)
