#!/usr/bin/env python
"""Segment points by colour from a pointcloud file and saves all reddish points
target pointcloud file. Autodectects ply, pcd and las files.

Usage: redStickDetection.py  [-h] <infile> <outfile>
"""

from docopt import docopt
from patty.segmentation.segRedStick import getRedMask
from patty.conversions import extract_mask, load, save

if __name__ == '__main__':
    args = docopt(__doc__)

    pc = load(args['<infile>'], loadRGB=True)
    redPc = extract_mask(pc, getRedMask(pc))
    save(redPc, args['<outfile>'])
