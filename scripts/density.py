# Calculate density of a point cloud, defined as num. of points per volume
# inside the convex hull.

from __future__ import division

import numpy as np
import pcl
from scipy.spatial import ConvexHull, Delaunay
import sys


p = pcl.load(sys.argv[1])
arr = p.to_array()

# http://stackoverflow.com/a/24734583/166749
def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6

hull = ConvexHull(arr)
simplices = np.column_stack((np.repeat(hull.vertices[0], hull.nsimplex),
                             hull.simplices))
tets = hull.points[simplices]
vol = np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                tets[:, 2], tets[:, 3]))

print("Points per unit volume: %.3g" % (len(arr) / vol))
