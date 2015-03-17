import logging
from patty.conversions import load, save

from nose import SkipTest
from nose.tools import assert_greater

logging.basicConfig(level=logging.INFO)


def testFilter():
    '''
    Test Voxel Grid Filter functionality
    '''
    fileLas = 'data/footprints/162.las'
    fileLasOut = 'data/footprints/162_sparse.las'

    # Load point cloud
    pc = load(fileLas)
    # Build Voxel Grid Filter
    vgf = pc.make_voxel_grid_filter()

    # Filter with Voxel size 1
    vgf.set_leaf_size(1, 1, 1)
    pc2 = vgf.filter()

    # Filter with Voxel size 0.1
    vgf.set_leaf_size(0.1, 0.1, 0.1)
    pc3 = vgf.filter()

    # Filter with Voxel size 10
    vgf.set_leaf_size(10, 10, 10)
    pc4 = vgf.filter()

    assert_greater(len(pc), len(pc2))
    assert_greater(len(pc3), len(pc2))
    assert_greater(len(pc), len(pc4))

    pc2.offset = pc.offset
    save(pc2, fileLasOut)

    logging.info(
        'Voxel grid filter has been applied to point cloud (making it sparse).'
        ' You can view the results using CloudCompare.')
    logging.info('  Original point cloud: ' + fileLas)
    logging.info('  Sparse point cloud  : ' + fileLasOut)
