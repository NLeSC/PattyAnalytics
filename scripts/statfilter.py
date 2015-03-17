#!/usr/bin/env python
"""Apply a statistical outlier filter to a pointcloud.

Usage:
    statfilter.py [-k <kmeans>] [-s <stddev>] <infile> <outfile>

Description:
    -k <kmeans>, --kmeans <kmeans>      K means value [default: 1000].
    -s <stddev>, --stddev <stddev>      Standard deviation cut-off
                                        [default: 2.0].
"""

from docopt import docopt
from patty.conversions import load, save


def statfilter(pc, k, s):
    """Apply the PCL statistical outlier filter to a point cloud"""
    fil = pc.make_statistical_outlier_filter()
    fil.set_mean_k(k)
    fil.set_std_dev_mul_thresh(s)
    return fil.filter()


if __name__ == "__main__":
    args = docopt(__doc__)

    pc = load(args['<infile>'])
    filter = statfilter(pc, int(args['--kmeans']), float(args['--stddev']))
    save(filter, args['<outfile>'])
