#!/usr/bin/env python
import pcl
import argparse
from patty.registration.stickScale import getStickScale

# Takes a point cloud containing only the red segments of scale sticks and
# returns the scale estimation and a confidence indication.

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--inFile', required=True, type=str,
                   help='Input PCD/PLY file')
    p.add_argument('-e', '--eps', required=False, default=0.1, type=float,
                   help='Max. dist between samples for them to be in the'
                        ' same neighborhood.')
    p.add_argument('-s', '--minSamples', required=False, default=20, type=int,
                   help='Min. number of samples in neighborhood for a point to'
                        ' be a core point.')
    args = p.parse_args()

    pc = pcl.load(args.inFile)

    print(getStickScale(pc, args.eps, args.minSamples))
