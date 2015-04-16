import colorsys
import numpy as np


def get_red_mask(pointcloud):
    """Returns a mask for the red parts of a pointcloud.

    Red points are points that have hue larger than 0.9
    and saturation larger than 0.5 in HSV colorspace.
    """

    red_mask = np.empty(len(pointcloud), dtype=np.bool)
    for i in xrange(len(pointcloud)):
        red, grn, blu = pointcloud[i][3:6]
        hue, sat, _ = colorsys.rgb_to_hsv(
            np.float32(red), np.float32(grn), np.float32(blu))
        red_mask[i] = hue > 0.9 and sat > 0.5

    return red_mask
