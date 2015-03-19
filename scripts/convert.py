#!/usr/bin/env python
"""Convert a pointcloud file from one format to another.

This procedure looses any information about the point normals, and the
geographic projection used. It keeps color information. It recognises
PLY, PCD and LAS files.

Usage:
  convert.py [-h] <infile> <outfile>
"""

from patty.conversions import load, save
from docopt import docopt

if __name__ == '__main__':
    args = docopt(__doc__)

    pc = load(args['<infile>'])
    save(pc, args['<outfile>'])
