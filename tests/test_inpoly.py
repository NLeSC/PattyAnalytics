import unittest
import logging
from patty.conversions import loadLas, loadCsvPolygon, writeLas
from patty.registration import registration

logging.basicConfig(level=logging.INFO)

class TestInPoly(unittest.TestCase):
    def testInPoly(self):
        '''
        Test point cloud / footprint intersection functionality provided
        by patty.registration.registration.intersect_polgyon2d()
        '''
        fileLas = 'data/footprints/162.las'
        fileLasOut = 'data/footprints/162_inFootprint.las'
        filePoly = 'data/footprints/162.las_footprint.csv'
        pc = loadLas(fileLas)
        footprint = loadCsvPolygon(filePoly)
        pcIn = registration.intersect_polgyon2d(pc, footprint)
        assert len(pc)>=len(pcIn)
        assert len(pcIn)>0

        writeLas(fileLasOut, pcIn)
        logging.info('Point cloud has been segmented to match footprint. You can view the results using CloudCompare.')
        logging.info('  Original point cloud : ' + fileLas)
        logging.info('  Footprint used       : ' + filePoly)
        logging.info('  Segmented point cloud: ' + fileLasOut)
