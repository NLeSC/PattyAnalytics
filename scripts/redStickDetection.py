#!/usr/bin/env python
"""Segment points by colour from a pointcloud file and saves all reddish points
target pointcloud file. Autodectects ply, pcd and las.

Usage: las_set_srs.py  [-h] <INFILE> <SRS> <OUTFILE>

Options:
  INFILE     Source pointcloud file
  SRS        EPSG number
  OUTFILE    Target pointcloud file
"""

from docopt import docopt
from patty.segmentation.segRedStick import getRedMask
from patty.conversions import extract_mask, load, save

if __name__=='__main__':
    args = docopt(__doc__)

    pc = load(args['<INFILE>'], loadRGB=True)
    redPc = extract_mask(pc, getRedMask(pc))
    save(redPc, args['<OUTFILE>'])
