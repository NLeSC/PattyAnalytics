'''
Pointcloud functions for reading/writing LAS files, and functions for dealing
with the spatial reference system.
'''

# DONT: liblas is deprecated, use laspy instead!
#       laspy does not work nice with numpy, keep using liblas
# http://laspy.readthedocs.org/en/latest/
# https://github.com/grantbrown/laspy.git

from __future__ import print_function
import liblas
import pcl
import os
import numpy as np


def _check_readable(filepath):
    """ Test whether filepath is readable, raises IOError otherwise """
    with open(filepath):
        pass


def _check_writable(filepath):
    """ Test whether filepath is writable, raises IOError otherwise """
    # either the path exists but is not writable, or the path does not exist
    # and the parent is not writable.
    if (os.path.exists(filepath) and (
            not os.path.isfile(filepath) or
            not os.access(filepath, os.W_OK)
            )) or not os.access(os.path.dirname(filepath), os.W_OK | os.X_OK):
        raise IOError("Cannot write to " + filepath)


def load(path, format=None, loadRGB=True):
    """ Read a pointcloud file.

    Supports LAS files, and lets PCD and PLY files be read by python-pcl.

    Arguments:
        path: file to load
        format: "PLY", "PCD", "LAS" or None. With none, it detects the filetype
             from the file extension.
        loadRGB: whether RGB is loaded for PLY and PCD files. For LAS files RGB
            is always read.
    Returns:
        registered pointcloud"""
    if format == 'las' or str(path).endswith('.las'):
        return load_las(path)
    else:
        _check_readable(path)
        pc = pcl.load(path, format=format, loadRGB=loadRGB)
        register(pc)
        return pc


def save(cloud, path, format=None, binary=False):
    """ Save a pointcloud to file.

    Supports LAS files, and lets PCD and PLY files be saved by python-pcl.

    Arguments:
        cloud: pcl.PointCloud/PointCloudXYZRGB
        file: file to save
        format: "PLY", "PCD", "LAS" or None. With none, it detects the filetype
             from the file extension.
        binary: whether PLY and PCD files are saved in binary format.
    """
    if format == 'las' or path.endswith('.las'):
        write_las(path, cloud)
    else:
        _check_writable(path)
        if is_registered(cloud) and cloud.offset != np.zeros(3):
            cloud_array = np.asarray(cloud)
            cloud_array += cloud.offset
        pcl.save(cloud, path, format=format, binary=binary)


def load_las(lasfile):
    """ Read a LAS file
    Returns:
        registered pointcloudxyzrgb

    The pointcloud has color and XYZ coordinates, and the offset and precision
    set."""
    _check_readable(lasfile)

    print("--READING--", lasfile, "---------")

    las = None
    try:
        las = liblas.file.File(lasfile)
        n_points = las.header.get_count()
        data = np.zeros((n_points, 6), dtype=np.float64)

        for i, point in enumerate(las):
            data[i] = (point.x, point.y, point.z, point.color.red /
                       256, point.color.green / 256, point.color.blue / 256)

        bb = BoundingBox(points=data[:, 0:3])
        # reduce the offset to decrease floating point errors
        data[:, 0:3] -= bb.center

        pc = pcl.PointCloudXYZRGB(data.astype(np.float32))

        register(pc, offset=bb.center, precision=las.header.scale,
                 crs_wkt=las.header.srs.get_wkt(),
                 crs_proj4=las.header.srs.get_proj4())

        return pc
    finally:
        if las is not None:
            las.close()


def is_registered(pointcloud):
    """Returns True when a pointcloud is registered."""
    return hasattr(pointcloud, 'is_registered') and pointcloud.is_registered


def register(pointcloud, offset=None, precision=None, crs_wkt=None,
             crs_proj4=None, crs_verticalcs=None):
    """Register a pointcloud

    Arguments:
        offset=None
            Offset [dx, dy, dz] for the pointcloud.
            Pointclouds often use double precision coordinates, this is
            necessary for some spatial reference systems like standard lat/lon.
            Subtracting an offset, typically the center of the pointcloud,
            allows us to use floats without losing precission.
            If no offset is set, defaults to [0, 0, 0]

        precision=None
            Precision of the points, used to store into a LAS file. Update
            when scaling the pointcloud.
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
        pointcloud.offset = np.array([0., 0., 0.], dtype=np.float64)
        pointcloud.precision = np.array([0.01, 0.01, 0.01], dtype=np.float64)
        pointcloud.crs_wkt = ''
        pointcloud.crs_proj4 = ''
        pointcloud.crs_verticalcs = ''

    if offset is not None:
        pointcloud.offset = np.asarray(offset, dtype=np.float64)
    if precision is not None:
        pointcloud.precision = np.asarray(precision, dtype=np.float64)
    if crs_wkt is not None:
        pointcloud.crs_wkt = crs_wkt
    if crs_proj4 is not None:
        pointcloud.crs_proj4 = crs_proj4
    if crs_verticalcs is not None:
        pointcloud.crs_verticalcs = crs_verticalcs


def copy_registration(target, src):
    """Copy spatial reference system metadata from src to target.

    Arguments:
        pointcloud_target: pcl.PointCloud
            pointcloud to copy registration to
        pointcloud_src: pcl.PointCloud
            registered pointcloud to copy registration from
    """
    target.is_registered = True
    target.offset = src.offset
    target.precision = src.precision
    target.crs_wkt = src.crs_wkt
    target.crs_proj4 = src.crs_proj4
    target.crs_verticalcs = src.crs_verticalcs


def load_csv_polygon(csvfile, delimiter=','):
    """Load a polygon from a simple CSV file

    Returns:
        numpy array containing the CSV file
    """
    return np.genfromtxt(csvfile, delimiter=delimiter)


def extract_mask(pointcloud, mask):
    """Extract all points in a mask into a new pointcloud.

    Arguments:
        pointcloud: pcl.PointCloud
            Input pointcloud
        mask: array of bool
            mask for which points from the pointcloud to include.
    Returns:
        pointcloud with the same registration (if any) as the original one."""
    pointcloud_new = pointcloud.extract(np.where(mask)[0])
    if is_registered(pointcloud):
        copy_registration(pointcloud_new, pointcloud)
    return pointcloud_new


def make_las_header(pc):
    """ Make a LAS header for given pointcloud.
    If the pointcloud is registered, this is taken into account for the
    header metadata. Has the side-effect of registering the given pointcloud.

    Arguments:
        pc: pcl.PointCloud
            Input pointcloud.
    Returns:
        liblas.header.Header for writing the pointcloud to LAS file with.
    """
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
    h.scale = np.asarray(pc.precision) * 0.5
    h.offset = pc.offset

    if pc.crs_wkt != '':
        h.srs.set_wkt(pc.crs_wkt)
    if pc.crs_proj4 != '':
        h.srs.set_proj4(pc.crs_proj4)
    if pc.crs_verticalcs != '':
        h.srs.set_verticalcs(pc.crs_verticalcs)

    pc_array = np.asarray(pc)
    h.min = pc_array.min(axis=0) + h.offset
    h.max = pc_array.max(axis=0) + h.offset
    return h


def write_las(lasfile, pc, header=None):
    """Write a pointcloud to a LAS file

    Arguments:
        lasfile : filename

        pc      : Pointclout to write
    """
    _check_writable(lasfile)

    print("--WRITING--", lasfile, "--------")
    if header is None:
        header = make_las_header(pc)

    precise_points = np.array(pc, dtype=np.float64)
    precise_points /= header.scale

    las = None
    try:
        las = liblas.file.File(lasfile, mode="w", header=header)

        for i in xrange(pc.size):
            pt = liblas.point.Point()
            pt.x, pt.y, pt.z = precise_points[i]
            r, g, b = pc[i][3:6]
            pt.color = liblas.color.Color(
                red=int(r) * 256, green=int(g) * 256, blue=int(b) * 256)
            las.write(pt)
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
            raise ValueError("Need to give min and max or matrix")

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


def center_boundingbox(pointcloud):
    """ Center the pointcloud on origin using the center of its bounding box.
    The offset compared to the original location is added to the
    pointcloud.offset. The pointcloud is registered after use.

    Arguments:
        pointcloud: pcl.PointCloud
            input pointcloud
    """
    register(pointcloud)
    pc_array = np.asarray(pointcloud)
    bb = BoundingBox(points=pc_array)
    pc_array -= bb.center
    pointcloud.offset += bb.center
