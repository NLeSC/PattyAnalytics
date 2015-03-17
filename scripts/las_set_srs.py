#!/usr/bin/env python
"""Set the spatial reference system (SRS/CRS) of a LAS file with the EPSG number.

This script completely reads a pointcloud, sets the SRS of the original header
and writes the entire pointcloud out.

Usage: las_set_srs.py  [-h] <INFILE> <SRS> <OUTFILE>

Options:
  INFILE     Source LAS file
  SRS        EPSG number
  OUTFILE    Target LAS file to write to
"""

import liblas
import osgeo.osr as osr
from docopt import docopt

if __name__ == "__main__":
    args = docopt(__doc__)

    osrs = osr.SpatialReference()
    osrs.SetFromUserInput( "EPSG:{}".format( args['<SRS>'] ) )

    lsrs = liblas.srs.SRS()
    lsrs.set_wkt( osrs.ExportToWkt() )

    f1 = liblas.file.File( args['<INFILE>'] )
    header = f1.header
    header.set_srs( lsrs )

    f2 = liblas.file.File( args['<OUTFILE>', header=header, mode="w" )
    for p in f1:
        f2.write(p)

    f1.close()
    f2.close()

