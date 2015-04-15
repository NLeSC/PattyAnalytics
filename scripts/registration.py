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
               [default: 1.0].
  -U           Trust the upvector completely and dont estimate it in this script, too
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
    trust_up = args['-U']
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

    initial_registration(pointcloud, up, drivemap, trust_up=trust_up)
    fine_registration(pointcloud, drivemap)
    center = coarse_registration(pointcloud, drivemap, footprint, downsample)
    fine_registration(pointcloud, drivemap, center, voxelsize=voxel)

    save( pointcloud, foutLas )

