"""Segmentation using DBSCAN.

Usage:
    dbscan <epsilon> <minpoints> <path>
"""

from docopt import docopt
import sys

import numpy as np
import pcl
from sklearn.cluster import dbscan

args = docopt(__doc__, sys.argv[1:])

eps = float(args['<epsilon>'])
minpoints = int(args['<minpoints>'])

# Kludge to get a proper exception for file not found
# (PCL will report "problem parsing header!").
with open(args['<path>']) as _:
    pc = pcl.load(args['<path>'])
X = np.asarray(pc)
print("%d points" % X.shape[0])

_, labels = dbscan(X, eps=eps, min_samples=minpoints, algorithm='kd_tree')

for label in np.unique(labels[labels != -1]):
    cluster = X[np.where(labels == label)]
    print("%d points in cluster %d" % (cluster.shape[0], label))
    out_pc = pcl.PointCloud(cluster)
    pcl.save(out_pc, 'cluster%d.ply' % label)

print("%d outliers" % np.sum(labels == -1))
out_pc = pcl.PointCloud(X[np.where(labels == -1)])
pcl.save(out_pc, 'outliers.ply')
