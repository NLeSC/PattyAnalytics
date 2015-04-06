import os
from tempfile import NamedTemporaryFile

import pcl
import numpy as np
from patty import conversions

from helpers import make_tri_pyramid_with_base

from numpy.testing import assert_array_almost_equal

def _make_some_points():
    side = 10
    delta = 0.1
    offset = [0, 0, 0]

    points, _ = make_tri_pyramid_with_base(side, delta, offset)
    return pcl.PointCloudXYZRGB(points.astype(np.float32))


def test_read_write():
    ''' Test read and write functionality'''
    filename = './testIO.las'

    pc1 = _make_some_points()
    pc1 = pcl.PointCloudXYZRGB( [[0,0,0],[3,4,5]] )
    conversions.save(pc1, filename)

    pc2 = conversions.load(filename)

    # dont use set_srs function, they will be tested later
    pc1_arr = np.asarray(pc1)
    pc2_arr = np.asarray(pc2)

    if hasattr(pc1, 'offset' ):
        pc1_arr += pc1.offset
    if hasattr(pc2, 'offset' ):
        pc2_arr += pc2.offset

    assert_array_almost_equal(pc1_arr, pc2_arr, 2,
                              "Written/read point clouds are different!")
    os.remove(filename)


def test_auto_file_format():
    """Test saving and loading to a PLY file with a ".las" extension."""
    with NamedTemporaryFile(suffix='.las') as f:
        pc = _make_some_points()
        conversions.save(pc, f.name, format="PLY")

        # Both PCL's loader and ours should get this.
        pcl.load(f.name, format="ply")
        conversions.load(f.name, format="PLY")
