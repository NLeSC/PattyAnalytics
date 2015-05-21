import os
from tempfile import NamedTemporaryFile

import pcl
import numpy as np
from patty import utils

from numpy.testing import assert_array_almost_equal
from nose.tools import assert_equal, assert_raises


def _compare( pcA, pcB ):
    ''' compare two pointclouds point-by-point'''

    pcA_arr = np.asarray(pcA)
    pcB_arr = np.asarray(pcB)

    # dont use set_srs function, they will be tested later
    if hasattr(pcA, 'offset' ):
        pcA_arr += pcA.offset
    if hasattr(pcB, 'offset' ):
        pcB_arr += pcB.offset

    assert_array_almost_equal(pcA_arr, pcB_arr, 2,
                              "Written/read point clouds are different!")


def test_read_write():
    ''' Test read and write LAS files functionality'''
    filename = './testIO.las'

    # make and save a pointcloud
    pc1 = pcl.PointCloud(10)
    pc1_arr = np.asarray(pc1)
    pc1_arr[:] = np.random.randn(*pc1_arr.shape)
    utils.save(pc1, filename)

    # reload it
    pc2 = utils.load(filename)

    _compare( pc1, pc2 )

    os.remove(filename)


def test_auto_file_format():
    """Test saving and loading pointclouds via the pcl loader"""

    # make and save a pointcloud
    pc = pcl.PointCloud(10)
    pc_arr = np.asarray(pc)
    pc_arr[:] = np.random.randn(*pc_arr.shape)

    with NamedTemporaryFile(suffix='.ply') as f:
        utils.save(pc, f.name)
        pc2 = utils.load(f.name)
        _compare( pc, pc2 )

    with NamedTemporaryFile(suffix='.pcd') as f:
        utils.save(pc, f.name)
        pc2 = utils.load(f.name)
        _compare( pc, pc2 )

    with NamedTemporaryFile(suffix='.las') as f:
        utils.save(pc, f.name, format="PLY")
        pc2 = utils.load(f.name, format="PLY")
        _compare( pc, pc2 )

    with NamedTemporaryFile(suffix='.las') as f:
        utils.save(pc, f.name, format="PCD")
        pc2 = utils.load(f.name, format="PCD")
        _compare( pc, pc2 )


def test_downsample_random():
    pc = pcl.PointCloud(10)
    a = np.asarray(pc)
    a[:] = np.random.randn(*a.shape)

    assert_raises(ValueError, utils.downsample_random, pc, 0)
    assert_raises(ValueError, utils.downsample_random, pc, 2)

    assert_equal(len(utils.downsample_random(pc, .39)), 4)
