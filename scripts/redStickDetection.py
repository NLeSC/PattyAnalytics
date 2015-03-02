#!/usr/bin/env python

import pcl
import argparse
from patty.segmentation.segRedStick import getRedMask
from patty.conversions import extract_mask

if __name__=='__main__':
    """Segment points by colour from a pcd file and saves all reddish points into a pcd of ply file."""
    parser = argparse.ArgumentParser(description='Segment points by colour from a ply or pcd file and saves all reddish points into a pcd of ply file.')
    parser.add_argument('-i','--inFile', required=True, type=str, help='Input PLY/PCD file')
    parser.add_argument('-o','--outFile',required=True, type=str, help='Output PLY/PCD file')
    args = parser.parse_args()

    pc = pcl.load(sourcePath, loadRGB=True)
    redPc = extract_mask(pc, getRedMask(pc))
    pcl.save(redPc, args.outFile)
