#!/usr/bin/env python

"""Segmentation using DBSCAN.

Usage:
    dbscan [--rgb_weight=weight] <epsilon> <minpoints> <path>
"""

from docopt import docopt
import sys

from patty.segmentation import segment_dbscan
from patty.conversions import load, save

if __name__ == '__main__':
    args = docopt(__doc__, sys.argv[1:])

    rgb_weight = float(args['--rgb_weight'] or 0)
    eps = float(args['<epsilon>'])
    minpoints = int(args['<minpoints>'])

    # Kludge to get a proper exception for file not found
    # (PCL will report "problem parsing header!").
    with open(args['<path>']) as _:
        pc = load(args['<path>'], loadRGB=True)
    print("%d points" % len(pc))

    clusters = segment_dbscan(pc, epsilon=eps, minpoints=minpoints,
                              rgb_weight=rgb_weight)

    n_outliers = len(pc)
    for i, cluster in enumerate(clusters):
        print("%d points in cluster %d" % (len(cluster), i))
        save(cluster, 'cluster%d.ply' % i)
        n_outliers -= len(cluster)

    print("%d outliers" % n_outliers)
