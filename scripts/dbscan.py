#!/usr/bin/env python

"""Segmentation using DBSCAN.

Segments a pointcloud into clusters using a DBSCAN algorithm.

Usage:
    dbscan [-r <weight>] [-f <format>] [-o <dir>] <epsilon> <minpoints> <file>

Options:
    -r <weight>, --rgb_weight <weight>  weight assigned to color space
                                        [default 0.0].
    -f <format>, --format <format>      format of output files [default: las].
    -o <dir>, --output_dir <dir>        output directory for clusters
                                        [default: .].
"""

from docopt import docopt
import sys

from patty.segmentation import segment_dbscan
from patty.conversions import load, save

if __name__ == '__main__':
    args = docopt(__doc__, sys.argv[1:])

    rgb_weight = float(args['--rgb_weight'])
    eps = float(args['<epsilon>'])
    minpoints = int(args['<minpoints>'])

    # Kludge to get a proper exception for file not found
    # (PCL will report "problem parsing header!").
    with open(args['<file>']) as _:
        pc = load(args['<file>'], loadRGB=True)
    print("%d points" % len(pc))

    clusters = segment_dbscan(pc, epsilon=eps, minpoints=minpoints,
                              rgb_weight=rgb_weight)

    n_outliers = len(pc)
    for i, cluster in enumerate(clusters):
        print("%d points in cluster %d" % (len(cluster), i))
        filename = '%s/cluster%d.%s' % (
                args['--output_dir'], i, args['--format']
            )
        save(cluster, filename)
        n_outliers -= len(cluster)

    print("%d outliers" % n_outliers)
