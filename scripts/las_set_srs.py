#!/usr/bin/env python
"""Set the spatial reference system (SRS/CRS) of a LAS file with the EPSG
number. This script completely reads a pointcloud, sets the SRS of the original
header and writes the entire pointcloud out to LAS.

Usage: las_set_srs.py  [-h] [--srs <srs>] <infile> <outfile>

Options:
  -s <srs>, --srs <srs>   EPSG number [default: 4326] (latlon).
"""

import liblas
import osgeo.osr as osr
from docopt import docopt

if __name__ == "__main__":
    args = docopt(__doc__)

    osrs = osr.SpatialReference()
    osrs.SetFromUserInput("EPSG:{}".format(args['--srs']))

    lsrs = liblas.srs.SRS()
    lsrs.set_wkt(osrs.ExportToWkt())

    f1 = liblas.file.File(args['<infile>'])
    header = f1.header
    header.set_srs(lsrs)

    print '%s' % lsrs

    f2 = liblas.file.File(args['<outfile>'], header=header, mode="w")
    for p in f1:
        f2.write(p)

    f1.close()
    f2.close()
