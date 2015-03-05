'''
Pointcloud functions for reading/writing LAS files, and functions for dealing with the spatial reference system.

Created on Oct 22, 2014

@author: carlosm, joris, jisk, lars, christiaan (NLeSC)
'''

# DONT: liblas is deprecated, use laspy instead!
#       laspy does not work nice with numpy, keep using liblas
# http://laspy.readthedocs.org/en/latest/
# https://github.com/grantbrown/laspy.git

import liblas
import pcl
import numpy as np
from patty.utils import BoundingBox

def loadLas(lasFile):
    """ Read a LAS file
    Returns:
        pointcloudxyzrgb, offset, scale)

    The pointcloud has color and XYZ coordinates
    The offset is the offset of the center point of the pointcloud
    The scale is the scale of the pointcloud."""

    try:
        print "--READING--", lasFile, "---------"
        las = liblas.file.File(lasFile)
        nPoints = las.header.get_count()
        data = np.zeros((nPoints, 6), dtype=np.float64)

        for i,point in enumerate(las):
            data[i] = (point.x,point.y,point.z,point.color.red/256,point.color.green/256,point.color.blue/256)

        bb = BoundingBox(points=data[:,0:3])
        # reduce the offset to decrease floating point errors
        data[:,0:3] -= bb.center
        
        pc = pcl.PointCloudXYZRGB(data.astype(np.float32))

        register(pc, offset=bb.center, precision=las.header.scale, crs_wkt=las.header.srs.get_wkt(), crs_proj4=las.header.srs.get_proj4())

        return pc
    finally:
        las.close()

def is_registered(pointcloud):
    """Returns True when a pointcleoud is registerd"""
    return hasattr(pointcloud, 'is_registered') and pointcloud.is_registered

def register(pointcloud, offset=None, precision=None, crs_wkt=None, crs_proj4=None,crs_verticalcs=None):
    """Register a pointcloud

    Arguments:
        offset=None
            Offset [dx, dy, dz] for the pointcloud.
            Pointclouds often use double precision coordinates, this is necessary for some spatial reference systems like standard lat/lon.
            Subtracting an offset, typically the center of the pointcloud, allows us to use floats without losing precission.
            If no offset is set, defaults to [0, 0, 0]

        precision=None
            Precision of the points, used to store into a LAS file. Update when scaling the pointcloud.
            If no precision is set, defaults to [0.01, 0.01, 0.01].

        crs_wkt=None
            Well Knonw Text form of the spatial reference system

        crs_proj4=None
            PROJ4 projection string for the spatial reference system

        crs_verticalcs=None
            Well Known Text form of the vertical coordinate system.
    """
    if not is_registered(pointcloud):
        pointcloud.is_registered = True
        pointcloud.offset = np.array([0., 0., 0.],dtype=np.float64)
        pointcloud.precision = np.array([0.01, 0.01, 0.01],dtype=np.float64)
        pointcloud.crs_wkt = ''
        pointcloud.crs_proj4 = ''
        pointcloud.crs_verticalcs = ''

    if offset is not None:
        pointcloud.offset = np.asarray(offset,dtype=np.float64)
    if precision is not None:
        pointcloud.precision = np.asarray(precision,dtype=np.float64)
    if crs_wkt is not None:
        pointcloud.crs_wkt = crs_wkt
    if crs_proj4 is not None:
        pointcloud.crs_proj4 = crs_proj4
    if crs_verticalcs is not None:
        pointcloud.crs_verticalcs = crs_verticalcs

def copy_registration(pointcloud_target, pointcloud_src):
    """Copy spatial reference system metadata from src to target.
    Arguments:
        pointcloud_target
        pointcloud_src
    """
    pointcloud_target.is_registered = True
    pointcloud_target.offset          = pointcloud_src.offset
    pointcloud_target.precision       = pointcloud_src.precision
    pointcloud_target.crs_wkt         = pointcloud_src.crs_wkt
    pointcloud_target.crs_proj4       = pointcloud_src.crs_proj4
    pointcloud_target.crs_verticalcs  = pointcloud_src.crs_verticalcs

def loadCsvPolygon(csvFile, delimiter=','):
    """Load a polygon from a simple CSV file
    Returns:
        numpy array containing the CSV file
    """
    return np.genfromtxt(csvFile, delimiter=delimiter)

def extract_mask(pointcloud, mask):
    pointcloud_new = pointcloud.extract(np.where(mask)[0])
    if(is_registered(pointcloud)):
        copy_registration(pointcloud_new, pointcloud)
    return pointcloud_new

def writeLas(lasFile, pc):
    """Write a pointcloud to a LAS file
    Arguments:
        lasFile  filename
        pc       Pointclout to write
    """

    try:
        print "--WRITING--", lasFile, "--------"
        f = liblas.schema.Schema()
        f.time = False
        f.color = True

        h = liblas.header.Header()
        h.schema = f
        h.dataformat_id = 3
        h.major_version = 1
        h.minor_version = 2

        register(pc)
        # FIXME: need extra precision to reduce floating point errors. We don't
        # know exactly why this works. It might reduce precision on the top of
        # the float, but reduces an error of one bit for the last digit.
        h.scale = np.asarray(pc.precision)*0.5
        h.offset = pc.offset

        if pc.crs_wkt != '':
            h.srs.set_wkt(pc.crs_wkt)
        if pc.crs_proj4 != '':
            h.srs.set_proj4(pc.crs_proj4)
        if pc.crs_verticalcs != '':
            h.srs.set_verticalcs(pc.crs_verticalcs)

        precise_points = np.array(pc, dtype=np.float64)
        precise_points /= h.scale
        h.min = precise_points.min(axis=0) + h.offset
        h.max = precise_points.max(axis=0) + h.offset
        las = liblas.file.File(lasFile, mode="w", header=h)

        for i in xrange(pc.size):
            pt = liblas.point.Point()
            pt.x,pt.y,pt.z = precise_points[i]
            r,g,b = pc[i][3:6]
            pt.color = liblas.color.Color( red = int(r) * 256, green = int(g) * 256, blue = int(b) * 256 )
            las.write(pt)
    finally:
        las.close()
