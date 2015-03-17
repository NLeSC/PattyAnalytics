#!/usr/bin/env python
"""Convert a pointcloud file from one format to another.

This procedure looses any information about the point normals, and the
geographic projection used. It keeps color information. It recognises
PLY, PCD and LAS files.

Usage: convert.py  [-h] <INFILE> <OUTFILE>

Options:
  INFILE     Source pointcloud file
  OUTFILE    Target pointcloud file
"""

from patty.conversions import load, save
from docopt import docopt

if __name__=='__main__':
    args = docopt(__doc__)

    pc = load(args['<INFILE>'], loadRGB=True)
    save(args['<OUTFILE>'], pc)
