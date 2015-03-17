import colorsys
import numpy as np


def getRedMask(pointcloud):
    """Returns a mask for the red parts of a pointcloud"""
    redMask = np.empty(len(pointcloud), dtype=np.bool)
    for i in xrange(len(pointcloud)):
        R, G, B = pointcloud[i][3:6]
        H, S, _ = colorsys.rgb_to_hsv(
            np.float32(R), np.float32(G), np.float32(B))
        redMask[i] = H > 0.9 and S > 0.5

    return redMask
