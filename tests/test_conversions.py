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

    pc = _make_some_points()
    pc = pcl.PointCloudXYZRGB( [[0,0,0],[3,4,5]] )
    conversions.force_srs(pc, srs="EPSG:28992", offset=[1,2,3])
    conversions.save(pc, filename)

    pc2 = conversions.load(filename, same_as=pc)

    pc_arr = pc.to_array()
    pc2_arr = pc2.to_array()
    # pc2_arr[:, 0] += pc2.offset[0]
    # pc2_arr[:, 1] += pc2.offset[1]
    # pc2_arr[:, 2] += pc2.offset[2]
    assert_array_almost_equal(pc_arr, pc2_arr, 2,
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
