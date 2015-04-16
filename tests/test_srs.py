import os

from patty.srs import set_srs, force_srs, is_registered
import pcl
import numpy as np
from osgeo import osr

from numpy.testing import assert_array_almost_equal
from nose.tools import assert_true,assert_false,assert_less

def test_force_srs():
    """Test the force_srs() function"""

    rdnew = osr.SpatialReference()
    rdnew.SetFromUserInput( "EPSG:28992" )

    latlon = osr.SpatialReference()
    latlon.SetFromUserInput( "EPSG:4326" )

    # using offset and srs (text)
    pcA = pcl.PointCloud( [[1,2,3]] )

    force_srs( pcA, offset=[0,0,0], srs="EPSG:28992" )
    assert_true( is_registered( pcA ) )

    assert_array_almost_equal( pcA.offset, np.zeros(3, dtype=np.float64), 15,
        "Offset not zero to 15 decimals" )

    assert_true( pcA.srs.IsSame( rdnew ) )

    # using same_as
    pcB = pcl.PointCloud( [[1,2,3]] )

    force_srs( pcB, same_as=pcA )
    assert_true( is_registered( pcB ) )

    assert_array_almost_equal( pcB.offset, np.zeros(3, dtype=np.float64), 15,
        "Offset not zero to 15 decimals" )
    assert_true( pcB.srs.IsSame( rdnew ) )

    # using no offset and osr.SpatialReference()
    pcC = pcl.PointCloud( [[1,2,3]] )

    force_srs( pcC, srs=rdnew )

    assert_true( pcC.srs.IsSame( rdnew ) )
    assert_false( hasattr( pcC, "offset" ) )

    # testing if no actual transform occurs on the points
    force_srs(pcC, srs=latlon )
    assert_false( pcC.srs.IsSame( pcA.srs ) )

    assert_array_almost_equal( np.asarray(pcA), np.asarray(pcC), 8,
        "force_srs() should not alter points" )


def test_set_srs():
    """Test the set_srs() function"""

    rdnew = osr.SpatialReference()
    rdnew.SetFromUserInput( "EPSG:28992" )

    latlon = osr.SpatialReference()
    latlon.SetFromUserInput( "EPSG:4326" )

    adam_latlon = np.array(
        [4.904153991281891, 52.372337993959924, 42.97214563656598],
        dtype=np.float64)

    adam_rdnew = np.array([122104.0, 487272.0, 0.0], dtype=np.float64)

    pcA = pcl.PointCloud ( [[0.,0.,0.]] )
    force_srs( pcA, srs=latlon, offset=adam_latlon )

    pcB = pcl.PointCloud ( [[0., 0., 0.]] )
    force_srs( pcB, srs=rdnew, offset=adam_rdnew )
    
    # latlon [degrees] -> rdnew [m]
    set_srs( pcA, same_as=pcB )

    assert_less( np.max( np.square( np.asarray( pcA ) ) ), 1e-3 ** 2, 
        "Coordinate transform not accurate to 1 mm %s" % np.asarray(pcA) )


    # rdnew [m] -> latlon [degrees]
    set_srs( pcB, srs=latlon, offset=[0,0,0] )

    # check horizontal error [degrees]
    error = np.asarray(pcB)[0] - adam_latlon

    assert_less( np.max( np.square(error[0:2]) ), (1e-6) ** 2, 
        "Coordinate transform rdnew->latlon not accurate to 1e-6 degree %s"
         % error[0:2] )

    # check vertical error [m]
    assert_less( abs(error[2]) , (1e-6) , 
        "Vertical Coordinate in of transform not accurate to 1e-6 meter %s"
         % abs(error[2]) )

