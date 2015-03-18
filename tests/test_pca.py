import logging
import numpy as np
from patty.conversions import load, save
from patty.registration import principal_axes_rotation

from nose import SkipTest

logging.basicConfig(level=logging.INFO)


def testPCARotation():
    raise SkipTest
    fileIn = 'data/footprints/162_inFootprint.las'
    fileOut = 'data/footprints/162_pca.las'
    pc = load(fileIn)
    transform = principal_axes_rotation(np.asarray(pc))
    pc.transform(transform)

    # TODO: How to test? assert what?

    save(pc, fileOut)
    logging.info(
        'Point cloud has been rotated to match PCA alignment.\n'
        'You can view the results using CloudCompare.')
    logging.info('  Original point cloud : ' + fileIn)
    logging.info('  Segmented point cloud: ' + fileOut)
