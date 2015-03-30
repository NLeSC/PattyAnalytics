import os
from tempfile import NamedTemporaryFile

import pcl
import numpy as np
from patty import conversions
from osgeo import osr

from helpers import make_tri_pyramid_with_base

from numpy.testing import assert_array_almost_equal
from nose.tools import assert_equal,assert_true,assert_false,assert_less

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


def test_force_srs():
    """Test the force_srs() function"""

    rdnew = osr.SpatialReference()
    rdnew.SetFromUserInput( "EPSG:28992" )

    latlon = osr.SpatialReference()
    latlon.SetFromUserInput( "EPSG:4326" )

    # using offset and srs (text)
    pcA = pcl.PointCloud( [[1,2,3]] )

    conversions.force_srs( pcA, offset=[0,0,0], srs="EPSG:28992" )
    assert_true( conversions.is_registered( pcA ) )

    assert_array_almost_equal( pcA.offset, np.zeros(3, dtype=np.float64), 15,
        "Offset not zero to 15 decimals" )

    assert_true( pcA.srs.IsSame( rdnew ) )

    # using same_as
    pcB = pcl.PointCloud( [[1,2,3]] )

    conversions.force_srs( pcB, same_as=pcA )
    assert_true( conversions.is_registered( pcB ) )

    assert_array_almost_equal( pcB.offset, np.zeros(3, dtype=np.float64), 15,
        "Offset not zero to 15 decimals" )
    assert_true( pcB.srs.IsSame( rdnew ) )

    # using no offset and osr.SpatialReference()
    pcC = pcl.PointCloud( [[1,2,3]] )

    conversions.force_srs( pcC, srs=rdnew )

    assert_array_almost_equal( pcC.offset, np.zeros(3, dtype=np.float64), 15,
        "Offset not zero to 15 decimals" )
    assert_true( pcC.srs.IsSame( rdnew ) )

    # testing if no actual transform occurs on the points
    conversions.force_srs(pcC, srs=latlon )
    assert_false( pcC.srs.IsSame( pcA.srs ) )

    assert_array_almost_equal( np.asarray(pcA), np.asarray(pcC), 8,
        "force_srs() should not alter points" )


def test_set_srs():
    """Test the set_srs() function"""

    rdnew = osr.SpatialReference()
    rdnew.SetFromUserInput( "EPSG:28992" )

    latlon = osr.SpatialReference()
    latlon.SetFromUserInput( "EPSG:4326" )

    # rdnew   122104             487272              0 
    # latlon: 4.904153991281891, 52.372337993959924, 42.97214563656598)

    pcA = pcl.PointCloud ( [[0.,0.,0.]] )
    conversions.force_srs( pcA, srs=latlon,
        offset=np.array([4.904153991281891, 52.372337993959924, 42.97214563656598], dtype=np.float64) )

    pcB = pcl.PointCloud ( [[0., 0., 0.]] )
    conversions.force_srs( pcB, srs=rdnew,
        offset=np.array([122104.0, 487272.0, 0.0], dtype=np.float64) )
    
    conversions.set_srs( pcA, same_as=pcB )

    assert_less( np.max( np.square( np.asarray( pcA ) ) ), 1e-3 ** 2, 
        "Coordinate transform not accurate to 1 mm" )
    
