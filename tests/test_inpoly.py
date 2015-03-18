import logging
from patty.conversions import load, save, load_csv_polygon
from patty.registration import intersect_polygon2d
from patty.registration import registration

from nose.tools import assert_true

logging.basicConfig(level=logging.INFO)


def test_in_poly():
    '''
    Test point cloud / footprint intersection functionality provided
    by patty.registration.intersect_polygon2d()
    '''
    fileLas = 'data/footprints/162.las'
    fileLasOut = 'data/footprints/162_inFootprint.las'
    filePoly = 'data/footprints/162.las_footprint.csv'
    pc = load(fileLas)
    footprint = load_csv_polygon(filePoly)
    pcIn = intersect_polygon2d(pc, footprint)
    assert_true(len(pc) >= len(pcIn))
    assert_true(len(pcIn) > 0)

    save(pcIn, fileLasOut)
    logging.info(
        'Point cloud has been segmented to match footprint.'
        ' You can view the results using CloudCompare.')
    logging.info('  Original point cloud : ' + fileLas)
    logging.info('  Footprint used       : ' + filePoly)
    logging.info('  Segmented point cloud: ' + fileLasOut)
