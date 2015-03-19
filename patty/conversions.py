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
    filepath = os.path.abspath(filepath)
    if (os.path.exists(filepath) and (
            not os.path.isfile(filepath) or
            not os.access(filepath, os.W_OK)
            )) or not os.access(os.path.dirname(filepath), os.W_OK | os.X_OK):
        raise IOError("Cannot write to " + filepath)


def load(path, format=None, load_rgb=True):
    """ Read a pointcloud file.

    Supports LAS files, and lets PCD and PLY files be read by python-pcl.

    Arguments:
        path: file to load
        format: "PLY", "PCD", "LAS" or None. With none, it detects the filetype
            from the file extension.
        load_rgb: whether RGB is loaded for PLY and PCD files. For LAS files
            RGB is always read.
    Returns:
        registered pointcloud"""
    if format == 'las' or str(path).endswith('.las'):
        return load_las(path)
    else:
        _check_readable(path)
        pointcloud = pcl.load(path, format=format, loadRGB=load_rgb)
        register(pointcloud)
        return pointcloud


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

        bbox = BoundingBox(points=data[:, 0:3])
        # reduce the offset to decrease floating point errors
        data[:, 0:3] -= bbox.center

        pointcloud = pcl.PointCloudXYZRGB(data.astype(np.float32))

        register(pointcloud, offset=bbox.center, precision=las.header.scale,
                 crs_wkt=las.header.srs.get_wkt(),
                 crs_proj4=las.header.srs.get_proj4())

        return pointcloud
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


def make_las_header(pointcloud):
    """ Make a LAS header for given pointcloud.
    If the pointcloud is registered, this is taken into account for the
    header metadata. Has the side-effect of registering the given pointcloud.

    Arguments:
        pointcloud: pcl.PointCloud
            Input pointcloud.
    Returns:
        liblas.header.Header for writing the pointcloud to LAS file with.
    """
    schema = liblas.schema.Schema()
    schema.time = False
    schema.color = True

    head = liblas.header.Header()
    head.schema = schema
    head.dataformat_id = 3
    head.major_version = 1
    head.minor_version = 2

    register(pointcloud)
    # FIXME: need extra precision to reduce floating point errors. We don't
    # know exactly why this works. It might reduce precision on the top of
    # the float, but reduces an error of one bit for the last digit.
    head.scale = np.asarray(pointcloud.precision) * 0.5
    head.offset = pointcloud.offset

    if pointcloud.crs_wkt != '':
        head.srs.set_wkt(pointcloud.crs_wkt)
    if pointcloud.crs_proj4 != '':
        head.srs.set_proj4(pointcloud.crs_proj4)
    if pointcloud.crs_verticalcs != '':
        head.srs.set_verticalcs(pointcloud.crs_verticalcs)

    pc_array = np.asarray(pointcloud)
    head.min = pc_array.min(axis=0) + head.offset
    head.max = pc_array.max(axis=0) + head.offset
    return head


def write_las(lasfile, pointcloud, header=None):
    """Write a pointcloud to a LAS file

    Arguments:
        lasfile : filename

        pointcloud      : Pointclout to write
    """
    _check_writable(lasfile)

    print("--WRITING--", lasfile, "--------")
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
    bbox = BoundingBox(points=pc_array)
    pc_array -= bbox.center
    pointcloud.offset += bbox.center
