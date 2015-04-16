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
               [default: 0.1].
  -v <voxel>   Downsample source pointcloud using voxel filter to speedup ICP
               [default: 0.05]
  -s <scale>   User override for initial scale factor
  -U           Dont trust the upvector completely and estimate it in this script, too
  -u <upfile>  Json file containing the up vector relative to the pointcloud.
  -c <camfile> CSV file containing all the camera postionions. [UNIMPLEMENTED]
"""

from __future__ import print_function
from docopt import docopt

import numpy as np
import os
import json
from patty.utils import (load, save, log)
from patty.srs import (set_srs, force_srs)

from patty.registration import (
    coarse_registration,
    fine_registration,
    initial_registration,
    )

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

    if args['-U']:
        trust_up = False
    else:
        trust_up = True

    try:
        downsample = float(args['-d'])
    except KeyError:
        downsample = 0.1

    try:
        voxel = float(args['-v'])
    except KeyError:
        voxel = 0.05

    try:
        initial_scale = float(args['-s'])
    except:
        initial_scale = None

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

    up = None
    try:
        with open(up_file) as f:
            dic = json.load(f)
        up = np.array(dic['estimatedUpDirection'])
        log("Reading up_file", up_file)
    except:
        log( "Cannot parse upfile, skipping" )

    initial_registration(pointcloud, up, drivemap, trust_up=trust_up, initial_scale=initial_scale)
    save( pointcloud, "initial.las" )
    center = coarse_registration(pointcloud, drivemap, footprint, downsample)
    save( pointcloud, "coarse.las" )
    fine_registration(pointcloud, drivemap, center, voxelsize=voxel)

    save( pointcloud, foutLas )

