'''
Pointcloud functions for reading/writing LAS files, and functions for dealing
with the spatial reference system.
'''

from __future__ import print_function
import liblas
import pcl
import os
import numpy as np
import time
from patty.srs import force_srs, is_registered

from sklearn.decomposition import PCA


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
        raise IOError("Cannot save to " + filepath)

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

def load(path, format=None, load_rgb=True ):
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

    return pc


def save(cloud, path, format=None, binary=False, las_header=None):
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
        las_header: liblas.header.Header
            LAS header to use. When none, a default header is created by
            make_las_header(). Default: None
    """
    if format == 'las' or format is None and path.endswith('.las'):
        _save_las(path, cloud, header=las_header)
    elif format == 'csv' or format is None and path.endswith('.csv'):
        _save_csv(path, cloud)
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
        lsrs = lsrs.get_wkt()

        n_points = las.header.get_count()
        precise_points = np.zeros((n_points, 6), dtype=np.float64)

        for i, point in enumerate(las):
            precise_points[i] = (point.x, point.y, point.z, point.color.red /
                       256, point.color.green / 256, point.color.blue / 256)

        # reduce the offset to decrease floating point errors
        bbox = BoundingBox(points=precise_points[:, 0:3])
        center = bbox.center
        precise_points[:, 0:3] -= center

        pointcloud = pcl.PointCloudXYZRGB(precise_points.astype(np.float32))
        force_srs( pointcloud, srs=lsrs, offset=center )

    finally:
        if las is not None:
            las.close()

    return pointcloud

def _load_csv(path, delimiter=','):
    """Load a set of points from a CSV file as 

    Returns:
        pc : pcl.PointCloud
    """
    precise_points = np.genfromtxt(path, delimiter=delimiter, dtype=np.float64 )
    offset = np.mean( precise_points, axis=0, dtype=np.float64 )
    pc = pcl.PointCloud(  np.array( precise_points - offset, dtype=np.float32 ) )

    force_srs(pc, offset=offset)
    return pc    

def _save_csv(path, pc, delimiter=', '):
    """Write a pointcloud to a CSV file.

    Arguments:
        path: string
            Output filename
        pc: pcl.PointCloud
            Pointcloud to save
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

    # FIXME: this format version assumes color is present
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


def _save_las(lasfile, pointcloud, header=None):
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

    # deal with color
    if len(pointcloud[0]) > 3:
        do_RGB = True
    else:
        do_RGB = False

    precise_points = np.array(pointcloud, dtype=np.float64)
    precise_points /= header.scale

    las = None
    try:
        las = liblas.file.File(lasfile, mode="w", header=header)

        for i in xrange(pointcloud.size):
            point = liblas.point.Point()
            point.x, point.y, point.z = precise_points[i]
            if do_RGB:
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


def measure_length(pointcloud):
    """Returns the length of a point cloud in its longest direction."""
    if len(pointcloud) < 2:
        return 0

    pca = PCA(n_components=1)
    pc_array = np.asarray(pointcloud)
    pca.fit(pc_array)
    primary_axis = np.dot(pc_array, np.transpose(pca.components_))[:, 0]
    return np.max(primary_axis) - np.min(primary_axis)


def downsample_voxel(pc, voxel_size=0.01):
    '''Downsample a pointcloud using a voxel grid filter.
    Resulting pointcloud has the same SRS and offset as the input.

    Arguments:
        pc         : pcl.PointCloud
                     Original pointcloud
        float      : voxel_size
                     Grid spacing for the voxel grid
    Returns:
        pc : pcl.PointCloud
             filtered pointcloud
    '''
    pc_filter = pc.make_voxel_grid_filter()
    pc_filter.set_leaf_size(voxel_size, voxel_size, voxel_size)
    newpc = pc_filter.filter()

    force_srs(newpc, same_as=pc)

    return newpc


def downsample_random(pc, fraction, random_seed=None):
    """Randomly downsample pointcloud to a fraction of its size.

    Returns a pointcloud of size fraction * len(pc), rounded to the nearest
    integer.  Resulting pointcloud has the same SRS and offset as the input.

    Use random_seed=k for some integer k to get reproducible results.
    Arguments:
        pc : pcl.PointCloud
            Input pointcloud.
        fraction : float
            Fraction of points to include.
        random_seed : int, optional
            Seed to use in random number generator.

    Returns:
        pcl.Pointcloud
    """
    if not 0 < fraction <= 1:
        raise ValueError("Expected fraction in (0,1], got %r" % fraction)

    rng = np.random.RandomState(random_seed)

    k = max(int(round(fraction * len(pc))), 1)
    sample = rng.choice(len(pc), k, replace=False)
    new_pc = pc.extract(sample)

    force_srs(new_pc, same_as=pc)

    return new_pc

