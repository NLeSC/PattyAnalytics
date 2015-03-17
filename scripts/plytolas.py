#!/usr/bin/env python
"""Convert a PLY file to a LAS file.

This procedure looses any information about the point normals, and the
geographic projection used. It keeps color information.

Usage: plytolas.py  [-h] <INFILE> <OUTFILE>

Options:
  INFILE     Source PLY file
  OUTFILE    Target LAS file to write to
"""

from patty.conversions import writeLas
import pcl
from docopt import docopt

if __name__=='__main__':
    args = docopt(__doc__)

    pc = pcl.load(args['<INFILE>'], loadRGB=True)
    writeLas(args['<OUTFILE>'], pc)
