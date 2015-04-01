'''
Pointcloud functions for reading/writing LAS files, and functions for dealing
with the spatial reference system.
'''

from __future__ import print_function
import liblas
import pcl
import os
import numpy as np
import osgeo.osr as osr
import time

def _check_readable(filepath):
    """ Test whether filepath is readable, raises IOError otherwise """
    with open(filepath):
        pass


def _check_writable(filepath):
    """ Test whether filepath is writable, raises IOError otherwise """
    # either the path exists but is not writable, or the path does not exist
    # and the parent is not writable.
    filepath = os.path.abspath(filepath)
    if (os.path.exists(filepath) and (
            not os.path.isfile(filepath) or
            not os.access(filepath, os.W_OK)
            )) or not os.access(os.path.dirname(filepath), os.W_OK | os.X_OK):
        raise IOError("Cannot write to " + filepath)

def clone(pc):
    """Return a copy of a pointcloud, including registration metadata

    Arguments:
        pc: pcl.PointCloud()
    Returns:
        cp: pcl.PointCloud()
    """

    cp = pcl.PointCloud( np.asarray(pc) )
    if is_registered(pc):
        force_srs(cp, same_as=pc)

    return cp

def load(path, format=None, load_rgb=True, same_as=None,
         offset=np.array([0,0,0], dtype=np.float64 ), srs=None):
    """Read a pointcloud file.

    Supports LAS and CSV files, and lets PCD and PLY files be read by python-pcl.

    Arguments:
        path : string
            Filename.
        format : string, optional
            File format: "PLY", "PCD", "LAS", "CSV" or None to detect the format
            from the file extension.
        load_rgb : bool
            Whether RGB is loaded for PLY and PCD files. For LAS files, RGB is
            always read.

    Optional Arguments. These are passed to set_srs() if the pointcloud has a
    reference system (LAS), or to force_srs() if not.

        same_as : pcl.pointcloud
        offset : np.array([3], dtype=np.float64 )
        srs : object or osgeo.osr.SpatialReference

    Returns:
        pc : pcl.PointCloud
    """
    if format == 'las' or format is None and path.endswith('.las'):
        pc = _load_las(path)
    elif format == 'las' or format is None and path.endswith('.csv'):
        pc = _load_csv(path)
    else:
        _check_readable(path)
        pc = pcl.load(path, format=format, loadRGB=load_rgb)

    # Set SRS and offset
    if same_as or ((offset is not None) and (srs is not None)):
        if is_registered(pc):
            set_srs(pc, offset=offset, srs=srs, same_as=same_as )
        else:
            force_srs(pc, offset=offset, srs=srs, same_as=same_as )
    else:
        if not is_registered(pc):
            pc.offset = np.array( [0,0,0], dtype=np.float64 )

    return pc


def save(cloud, path, format=None, binary=False):
    """Save a pointcloud to file.

    Supports LAS and CSV files, and lets PCD and PLY files be saved by python-pcl.

    Arguments:
        cloud : pcl.PointCloud or pcl.PointCloudXYZRGB
            Pointcloud to save.
        path : string
            Filename.
        format : string
            File format: "PLY", "PCD", "LAS", "CSV" or None to detect the format
             from the file extension.
        binary : boolean
            Whether PLY and PCD files are saved in binary format.
    """
    if format == 'las' or format is None and path.endswith('.las'):
        _write_las(path, cloud)
    elif format == 'csv' or format is None and path.endswith('.csv'):
        _write_csv(path, cloud)
    else:
        _check_writable(path)
        if is_registered(cloud) and cloud.offset != np.zeros(3):
            cloud_array = np.asarray(cloud)
            cloud_array += cloud.offset
        pcl.save(cloud, path, format=format, binary=binary)


def _load_las(lasfile):
    """Read a LAS file

    Returns:
        registered pointcloudxyzrgb

    The pointcloud has color and XYZ coordinates, and the offset and precision
    set.
    """
    _check_readable(lasfile)

    las = None
    try:
        las = liblas.file.File(lasfile)
        lsrs = las.header.get_srs()

        n_points = las.header.get_count()
        data = np.zeros((n_points, 6), dtype=np.float64)

        for i, point in enumerate(las):
            data[i] = (point.x, point.y, point.z, point.color.red /
                       256, point.color.green / 256, point.color.blue / 256)

        # reduce the offset to decrease floating point errors
        bbox = BoundingBox(points=data[:, 0:3])
        center = bbox.center
        data[:, 0:3] -= center

        pointcloud = pcl.PointCloudXYZRGB(data.astype(np.float32))
        force_srs( pointcloud, srs=lsrs.get_wkt(), offset=center )

    finally:
        if las is not None:
            las.close()

    return pointcloud

def is_registered(pointcloud):
    """Returns True when a pointcloud is registered; ie coordinates are relative
       to a specific spatial reference system and offset."""
    return hasattr(pointcloud, 'is_registered') and pointcloud.is_registered

def same_srs(pcA, pcB):
    """True if the two pointclouds have the same coordinate system

    Arguments:
        pcA : pcl.PointCloud
        pcB : pc..PointCloud
    """

    if is_registered(pcA) and is_registered(pcB):
        if np.mean(pcA.offset - pcB.offset) < 1E-5:
            if pcA.srs.IsSame( pcB.srs ):
                return True
    return False

def set_srs(pc, srs=None, offset=np.array( [0,0,0], dtype=np.float64),
            same_as=None):
    """Set the spatial reference system (SRS) and offset for a pointcloud.
    This function transforms all the points to the new reference system, and
    updates the metadata accordingly.

    Either give a SRS and offset, or a reference pointcloud

    NOTE: Pointclouds in PCL do not have absolute coordinates, ie.
          latitude / longitude. This function sets metadata to the pointcloud
          describing an absolute frame of reference.
          It is left to the user to make sure pointclouds are in the same
          reference system, before passing them on to PCL functions. This
          can be checked with patty.conversions.same_srs().

    NOTE: To add a SRS to a point cloud, or to update incorrect metadata,
          use force_srs().

    Example:

        # set the SRS to lat/lon, don't use an offset, so it defaults to [0,0,0]
        set_srs( pc, srs="EPSG:4326" )

    Arguments:
        pc : pcl.Pointcloud, with pcl.is_registered() == True

        same_as : pcl.PointCloud

        offset : np.array([3], dtype=np.float64 )
            Must be added to the points to get absolute coordinates,
            neccesary to retain precision for LAS pointclouds.

        srs : object or osgeo.osr.SpatialReference
            If it is an SpatialReference, it will be used directly.
            Otherwise it is passed to osr.SpatialReference.SetFromUserInput()

    Returns:
        pc : pcl.PointCloud
            The input pointcloud.
    
    """
    if not is_registered(pc):
        raise TypeError( "Pointcloud is not registered" )

    if same_as:
        if is_registered(same_as):
            newsrs    = same_as.srs
            newoffset = same_as.offset
        else:
            raise TypeError("Reference pointcloud is not registered")

    else:
        if type(srs) == type(osr.SpatialReference()):
            newsrs = srs
        else:
            newsrs = osr.SpatialReference()
            newsrs.SetFromUserInput(srs)

        if offset is not None:
            newoffset = np.array( offset, dtype=np.float64 )
            if len(newoffset) != 3:
                raise TypeError("Offset should be an np.array([3])")
        else:
            newoffset = np.zeros([3], dtype=np.float64 )

    if not pc.srs.IsSame( newsrs ):
        T = osr.CoordinateTransformation( pc.srs, newsrs )

        data =  np.asarray( pc )

        # add old offset, do transformation, substract new offset
        precise_points = np.array(data, dtype=np.float64) + pc.offset
        precise_points = np.array( T.TransformPoints( precise_points ), dtype=np.float64 )
        precise_points -= newoffset

        # copy the float64 to pointcloud
        data[...] = np.asarray( precise_points, dtype=np.float32 )

        # fix metadata
        pc.srs = newsrs.Clone()
        pc.offset = np.array( newoffset, dtype=np.float64 )

    # FIXME do better comparison
    elif np.max(pc.offset - newoffset) > 1.e-5:
        pc.translate( pc.offset - newoffset )
        pc.offset = np.array( newoffset, dtype=np.float64)

    return pc

def force_srs(pc, srs=None, offset=np.array([0,0,0], dtype=np.float64),
              same_as=None):
    """Set a spatial reference system (SRS) and offset for a pointcloud.
    This function affects the metadata only, and sets pc.is_registered to True

    Either give a SRS and offset, or a reference pointcloud

    This is the recommended way to turn a python-pcl pointcloud to a 
    registerd pointcloud with absolute coordiantes.

    NOTE: To change the SRS for an already registered pointcloud, use set_srs()

    Example:

        # set the SRS to lat/lon, don't use offset
        force_srs( pc, srs="EPSG:4326", offset=[0,0,0] )

    Arguments:
        pc : pcl.Pointcloud

        same_as : pcl.PointCloud

        offset : np.array([3])
            Must be added to the points to get absolute coordinates,
            neccesary to retain precision for LAS pointclouds.

        srs : object or osgeo.osr.SpatialReference
            If it is an SpatialReference, it will be used directly.
            Otherwise it is passed to osr.SpatialReference.SetFromUserInput()

    Returns:
        pc : pcl.PointCloud
            The input pointcloud.
    
    """
    if same_as:
        if is_registered(same_as):
            pc.srs = same_as.srs.Clone()
            pc.offset = np.array( same_as.offset, dtype=np.float64 )
        else:
            raise TypeError("Reference pointcloud is not registered")
    else:
        if type(srs) == type(osr.SpatialReference()):
            pc.srs = srs.Clone()
        else:
            pc.srs = osr.SpatialReference()
            pc.srs.SetFromUserInput(srs)

        offset = np.asarray( offset, dtype=np.float64 )
        if len(offset) != 3:
            raise TypeError("Offset should be an np.array([3])")
        pc.offset = offset

    pc.is_registered = True

    return pc

def _load_csv(path, delimiter=','):
    """Load a set of points from a CSV file as 

    Returns:
        pc : pcl.PointCloud
    """
    return pcl.PointCloud( np.genfromtxt(path, delimiter=delimiter).astype( np.float32 ) )

def _write_csv(path, pc, delimiter=', '):
    """Write a pointcloud to a CSV file.

    Arguments:
        path: string
            Output filename
        pc: pcl.PointCloud
            Pointcloud to write
        delimiter: string
            Field delimiter to use, see np.savetxt documentation.

    """
    if not hasattr(pc, 'offset'):
        offset = np.zeros(3)
    else:
        offset = pc.offset

    np.savetxt(path, np.asarray(pc) + offset, delimiter=delimiter )


def extract_mask(pointcloud, mask):
    """Extract all points in a mask into a new pointcloud.

    Arguments:
        pointcloud : pcl.PointCloud
            Input pointcloud.
        mask : numpy.ndarray of bool
            mask for which points from the pointcloud to include.
    Returns:
        pointcloud with the same registration (if any) as the original one."""
    pointcloud_new = pointcloud.extract(np.where(mask)[0])
    if is_registered(pointcloud):
        force_srs(pointcloud_new, same_as=pointcloud)
    return pointcloud_new


def make_las_header(pointcloud):
    """Make a LAS header for given pointcloud.

    If the pointcloud is registered, this is taken into account for the
    header metadata.

    LAS rounds the coordinates on writing; this is controlled via the
    'precision' attribute of the input pointcloud. By default this is
    0.01 in units of the projection.

    Arguments:
        pointcloud : pcl.PointCloud
            Input pointcloud.
    Returns:
        header : liblas.header.Header
            Header for writing the pointcloud to a LAS file.
    """
    schema = liblas.schema.Schema()
    schema.time = False
    schema.color = True

    head = liblas.header.Header()
    head.schema = schema
    head.dataformat_id = 3
    head.major_version = 1
    head.minor_version = 2

    if is_registered(pointcloud):
        try:
            lsrs = liblas.srs.SRS()
            lsrs.set_wkt(pointcloud.srs.ExportToWkt())
            head.set_srs(lsrs)
        except liblas.core.LASException:
            pass

    if hasattr(pointcloud, 'offset'):
        head.offset = pointcloud.offset
    else:
        head.offset = np.zeros(3)

    # FIXME: need extra precision to reduce floating point errors. We don't
    # know exactly why this works. It might reduce precision on the top of
    # the float, but reduces an error of one bit for the last digit.
    if not hasattr(pointcloud, 'precision'):
        precision = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    else:
        precision = np.array( pointcloud.precision, dtype=np.float64)
    head.scale = precision * 0.5

    pc_array = np.asarray(pointcloud)
    head.min = pc_array.min(axis=0) + head.offset
    head.max = pc_array.max(axis=0) + head.offset
    return head


def _write_las(lasfile, pointcloud, header=None):
    """Write a pointcloud to a LAS file

    Arguments:
        lasfile : string
            Filename.

        pointcloud : pcl.PointCloud

        header : liblas.header.Header, optional
            See :func:`make_las_header`. If not given, makes a header using
            that function with default settings.
    """
    _check_writable(lasfile)

    if header is None:
        header = make_las_header(pointcloud)

    precise_points = np.array(pointcloud, dtype=np.float64)
    precise_points /= header.scale

    las = None
    try:
        las = liblas.file.File(lasfile, mode="w", header=header)

        for i in xrange(pointcloud.size):
            point = liblas.point.Point()
            point.x, point.y, point.z = precise_points[i]
            red, grn, blu = pointcloud[i][3:6]
            point.color = liblas.color.Color(
                red=int(red) * 256, green=int(grn) * 256, blue=int(blu) * 256)
            las.write(point)
    finally:
        if las is not None:
            las.close()


class BoundingBox(object):
    '''A bounding box for a sequence of points.

    Center, size and diagonal are updated when the minimum or maximum are
    updated.

    Constructor usage: either set points (any object that is converted to an
    NxD array by np.asarray, with D the number of dimensions) or a fixed min
    and max.
    '''

    def __init__(self, points=None, min=None, max=None):
        if min is not None and max is not None:
            self._min = np.asarray(min, dtype=np.float64)
            self._max = np.asarray(max, dtype=np.float64)
        elif points is not None:
            points_array = np.asarray(points)
            self._min = points_array.min(axis=0)
            self._max = points_array.max(axis=0)
        else:
            raise TypeError("Need to give min and max or matrix")

        self._reset()

    def __str__(self):
        return 'BoundingBox <%s - %s>' % (self.min, self.max)

    def _reset(self):
        self._center = None
        self._size = None

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, new_min):
        self._reset()
        self.min = new_min

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, new_max):
        self._reset()
        self.max = new_max

    @property
    def center(self):
        ''' Center point of the bounding box'''
        if self._center is None:
            self._center = (self.min + self.max) / 2.0
        return self._center

    @property
    def size(self):
        ''' N-dimensional size array '''
        if self._size is None:
            self._size = self.max - self.min
        return self._size

    @property
    def diagonal(self):
        ''' Length of the diagonal of the box. '''
        return np.linalg.norm(self.size)

    def contains(self, pos):
        ''' Whether the bounding box contains given position. '''
        return np.all((pos >= self.min) & (pos <= self.max))

def log(*args, **kwargs):
    """Simple logging function that prints to stdout"""
    print(time.strftime("[%F %H:%M:%S]", time.gmtime()), *args, **kwargs)

