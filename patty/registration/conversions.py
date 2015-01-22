'''
Created on Oct 22, 2014

@author: carlosm
'''

# TODO: liblas is deprecated, use laspy instead!
# http://laspy.readthedocs.org/en/latest/
# https://github.com/grantbrown/laspy.git

import liblas
import pcl
import numpy as np



# def RegisteredPointCloud:
# holds a PointCloudXYZRBG
# holds a CRS (ie. latlon, or rome coordinates, or RD-NEW, etc. given as EPSG srid) int
# holds additional offset (to prevent underflow on cooridnates in the double->float conversion) double
# scaling probably not needed? double
 
# def RegisterPointCloud:
# input: intentin     a registered point cloud (drivemap)
#        intentin/out unregistered point cloud (object), PointXYZRGB
#        intentin     method?
# output: 
#                     the object translated/rotated to its new position
#                     fitness
#
# Approach: 1. estimate scale factor: drivemap part is just the ST_Buffer( , 5) on the object's footprint
#           2. scale
#           3. SAC-IA?
#           4. one / all of the ICP functions?
#           5. move object 
#           6. warp in a RegisterPointCloud class with correct metadata
#
#
# class RegisteredPointCloud(pcl.PointCloudXYZRGB):
#     """Point cloud with registration information.
#
#     Parameters
#     ----------
#     crs : int
#         Coordinate reference system. Will be passed to GDAL.
#         Laspy does not support reading/writing these in LAS files.
#     offset : 3-vector of float64
#         Offset from the origin; defaults to (0, 0, 0).
#     """
#     def __init__(self, data):
#         super(RegisteredPointCloud, self).__init__(data)
#
#         self.crs = 4326
#         self.offset = np.zeros(3)
#
#     def register(self, crs=None, offset=None):
#         if crs is not None:
#             self.crs = crs
#         if offset is not None:
#             self.offset = np.asarray(offset, dtype=np.float64)
#
#     def transform_crs(self, crs):
#         """Transform to other coordinate reference system."""
#         source = osr.SpatialReference()
#         source.ImportFromEPSG(self.crs)
#
#         target = osr.SpatialReference()
#         target.ImportFromEPSG(crs)
#
#         transf = osr.CoordinateTransformation(source, target)
#         point = ogr.Geometry(ogr.wkbPoint)
#         for i, (x, y, z) in enumerate(self):
#             point.AddPoint(x, y, z)
#             point.Transform(transf)
#             self[i] = point.GetPoint()
#
#         self.crs = crs



''' Read a las file
returns (pointcloudxyzrgb, offset, scale)
The pointcloud has color and XYZ coordinates
The offset is the offset of the center point of the pointcloud
The scale is the scale of the pointcloud.'''
def loadLas(lasFile):
    try:
        print "--READING--", lasFile, "---------"
        las = liblas.file.File(lasFile)
        nPoints = las.header.get_count()
        data = np.zeros((nPoints, 6), dtype=np.float64)
        min_point = np.array(las.header.get_min())
        max_point = np.array(las.header.get_max())
        offset = min_point + (max_point - min_point)/2
        scale = np.array(las.header.get_scale())
		
        # CRS = None # FIXME: keep track of CRS

        for i,point in enumerate(las):
            data[i] = (point.x,point.y,point.z,point.color.red/256,point.color.green/256,point.color.blue/256)

		# reduce the offset to decrease floating point errors
        data[:,0:3] -= offset
        
        pc = pcl.PointCloudXYZRGB(data.astype(np.float32))
        
        register(pc, offset, las.header.scale, las.header.srs.get_wkt(), las.header.srs.get_proj4())
        
        return pc
    finally:
        las.close()

def is_registered(pointcloud):
    return hasattr(pointcloud, 'is_registered') and pointcloud.is_registered

def register(pointcloud, offset=None, precision=None, crs_wkt=None, crs_proj4=None,crs_verticalcs=None):
    if not is_registered(pointcloud):
        pointcloud.is_registered = True
        pointcloud.offset = np.array([0., 0., 0.],dtype=np.float64)
        pointcloud.precision = np.array([0.01, 0.01, 0.01],dtype=np.float64)
        pointcloud.crs_wkt = ''
        pointcloud.crs_proj4 = ''
        pointcloud.crs_verticalcs = ''

    if offset is not None:
        pointcloud.offset = np.array(offset,dtype=np.float64)
    if precision is not None:
        pointcloud.precision = np.array(precision,dtype=np.float64)            
    if crs_wkt is not None:
        pointcloud.crs_wkt = crs_wkt
    if crs_proj4 is not None:
        pointcloud.crs_proj4 = crs_proj4
    if crs_verticalcs is not None:
        pointcloud.crs_verticalcs = crs_verticalcs

def copy_registration(pointcloud_target, pointcloud_src):
    pointcloud_target.is_registered = True
    pointcloud_target.offset          = pointcloud_src.offset
    pointcloud_target.precision       = pointcloud_src.precision
    pointcloud_target.crs_wkt         = pointcloud_src.crs_wkt
    pointcloud_target.crs_proj4       = pointcloud_src.crs_proj4
    pointcloud_target.crs_verticalcs  = pointcloud_src.crs_verticalcs

def loadCsvPolygon(csvFile, delimiter=','):
    return np.genfromtxt(csvFile, delimiter=delimiter)

def writeLas(lasFile, pc):
    # try:
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
        h.scale = np.array(pc.precision)*0.5 # FIXME: need extra precision to reduce floating point errors. We don't know exactly why this works. It might reduce precision on the top of the float, but reduces an error of one bit for the last digit.
            
        h.offset = pc.offset

        if pc.crs_wkt != '':
            h.srs.set_wkt(pc.crs_wkt)
        if pc.crs_proj4 != '':
            h.srs.set_proj4(pc.crs_proj4)
        if pc.crs_verticalcs != '':
            h.srs.set_verticalcs(pc.crs_verticalcs)        
        
        # # FIXME: set CRS
        
        a = np.asarray(pc)
        precise_points = np.array(a, dtype=np.float64)
        precise_points /= h.scale
        h.min = precise_points.min(axis=0) + h.offset
        h.max = precise_points.max(axis=0) + h.offset
        las = liblas.file.File(lasFile, mode="w", header=h)
        
        for i in xrange(pc.size):
            pt = liblas.point.Point()
            pt.x,pt.y,pt.z = precise_points[i]
            r,g,b = pc[i][3:6]
            pt.color = liblas.color.Color( red = int(round(r * 256.0)), green = int(round(g * 256.0)), blue = int(round(b * 256.0)) )
            las.write(pt)
    # finally:
        # las.close()

#
# def las2ply(lasFile, plyFile):
#     pc, scale = loadLas(lasFile)
#     pcl.save(pc, plyFile, format='PLY')
#
# def ply2las(plyFile, lasFile):
#     pc = pcl.load(plyFile, loadRGB=True)
#     writeLas( lasFile, pc )

if __name__ == '__main__':
    plyFile = 'tests/10.ply'
    lasFile = 'tests/10.las.out'

    print 'From ply to las...'
    pc = pcl.load(plyFile, format='PLY', loadRGB=True)

    # print 'From las to ply...'
    # las2ply(lasFile, plyFile)


# Test code for the projection CRS

# Rome / via appia is in EPSG:32633
# http://pcjericks.github.io/py-gdalogr-cookbook/projection.html
#
# from osgeo import gdal
#
# spatialRef = osr.SpatialReference()
# spatialRef.ImportFromEPSG(2927)         # from EPSG
#
# target = osr.SpatialReference()
# target.ImportFromEPSG(4326)
#
# transform = osr.CoordinateTransformation(source, target)
#
# point = ogr.CreateGeometryFromWkt("POINT (1120351.57 741921.42)")
# point.Transform(transform)
#
# print point.ExportToWkt()
#
#
# # http://pcjericks.github.io/py-gdalogr-cookbook/geometry.html#create-a-point
# from osgeo import ogr
# point = ogr.Geometry(ogr.wkbPoint)
# point.AddPoint(1198054.34, 648493.09)
# print point.ExportToWkt()
#
