import unittest
import logging
import numpy as np
from patty.conversions import loadLas, writeLas
from patty.registration.principalComponents import principal_axes_rotation

logging.basicConfig(level=logging.INFO)


class TestPrincipalComponentRotation(unittest.TestCase):

    def testPCARotation(self):
        fileIn = 'data/footprints/162_inFootprint.las'
        fileOut = 'data/footprints/162_pca.las'
        pc = loadLas(fileIn)
        transform = principal_axes_rotation(np.asarray(pc))
        pc.transform(transform)

        # TODO: How to test? assert what?

        writeLas(fileOut, pc)
        logging.info(
            'Point cloud has been rotated to match PCA alignment. You can view the results using CloudCompare.')
        logging.info('  Original point cloud : ' + fileIn)
        logging.info('  Segmented point cloud: ' + fileOut)
