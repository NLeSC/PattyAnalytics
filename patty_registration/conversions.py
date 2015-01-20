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

''' Read a las file
returns (pointcloudxyzrgb, offset, scale)
The pointcloud has color and XYZ coordinates
The offset is the offset of the center point of the pointcloud
The scale is the scale of the pointcloud.'''
def loadLas(lasFile):
    try:
        las = liblas.file.File(lasFile)
        nPoints = las.header.get_count()
        data_xyz = np.zeros((nPoints, 6), dtype=np.float64)
        min_point = np.array(las.header.get_min())
        max_point = np.array(las.header.get_max())
        offset = min_point + (max_point - min_point)/2
        scale = np.array(las.header.get_scale())
		
        # CRS = None # FIXME: keep track of CRS

        for i,point in enumerate(las):
            data_xyz[i,0:3] = point.x,point.y,point.z
            data_xyz[i,3:6] = point.color.red,point.color.green,point.color.blue
        
		# reduce the offset to decrease floating point errors
        data_xyz[:,0:3] -= offset
        # point cloud colors live in [0,1]^3 space, not in [0,255]^3
        data_xyz[:,3:6] /= 256.0

        pc = pcl.PointCloudXYZRGB(data_xyz.astype(np.float32))
        return pc, offset, scale
    finally:
        las.close()

def loadCsvPolygon(csvFile, delimiter=','):
    return np.genfromtxt(csvFile, delimiter=delimiter)

def writeLas(lasFile, pc, CRS = None):
    try:
        f = liblas.schema.Schema()
        f.time = False
        f.color = True

        h = liblas.header.Header()
        h.schema = f
        h.dataformat_id = 3
        h.minor_version = 2

        # FIXME: set CRS

        a = pc.to_array()
        h.min = a.min(axis=0)
        h.max = a.max(axis=0)

        h.scale = [1.0, 1.0, 1.0]
        h.offset = [0., 0., 0.]

        las = liblas.file.File(lasFile, mode="w", header=h)

        for i in range(pc.size):
            pt = liblas.point.Point()
            pt.x,pt.y,pt.z, r,g,b = pc[i]
            pt.color = liblas.color.Color( red = int(r * 256), green = int(g * 256), blue = int(b * 256) )
            las.write(pt)

    finally:
        las.close()


def las2ply(lasFile, plyFile):
    pc, offset = loadLas(lasFile)
    pcl.save(pc, plyFile, format='PLY')

def ply2las(plyFile, lasFile):
    pc = pcl.load(plyFile, loadRGB=True)
    writeLas( lasFile, pc )

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
