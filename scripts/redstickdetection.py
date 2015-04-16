#!/usr/bin/env python
"""Segment points by colour from a pointcloud file and saves all reddish points
target pointcloud file. Autodectects ply, pcd and las files.

Usage: redstickdetection.py  [-h] <infile> <outfile>
"""

from docopt import docopt
from patty.segmentation.segRedStick import get_red_mask
from patty.utils import extract_mask, load, save

if __name__ == '__main__':
    args = docopt(__doc__)

    pc = load(args['<infile>'])
    red_pc = extract_mask(pc, get_red_mask(pc))
    save(red_pc, args['<outfile>'])
