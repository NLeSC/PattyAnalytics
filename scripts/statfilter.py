#!/usr/bin/env python

import pcl
import argparse

def statfilter( pc, k, s ):
    """Apply the PCL statistical outlier filter to a point cloud"""
    fil = pc.make_statistical_outlier_filter()
    fil.set_mean_k( k )
    fil.set_std_dev_mul_thresh( s )
    return fil.filter()

def main():
    parser = argparse.ArgumentParser(description="Apply a statistical outlier filter to a pointcloud")
    parser.add_argument("-o", "--outfile", type=str, help="The output namelist", required=True )
    parser.add_argument("-i", "--infile",  type=str, help="The input filename", required=True )
    parser.add_argument("-k", "--kmean",  type=int, help="k mean value (1000)", default=1000 )
    parser.add_argument("-s", "--stdev",  type=float, help="standard deviation cut-off (2.)", default=2.0 )
    args = parser.parse_args()

    pcl.save( statfilter( pcl.load(args.infile), args.kmean, args.stdev),  args.outfile )


if __name__ == "__main__":
    main()
