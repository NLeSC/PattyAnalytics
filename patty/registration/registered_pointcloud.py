import laspy.file
import laspy.header
import numpy as np
from pcl import PointCloudXYZRGB


class RegisteredPointCloud(PointCloudXYZRGB):
    """Point cloud with registration information.

    Parameters
    ----------
    crs : int
        Coordinate reference system. Will be passed to GDAL.
        Laspy does not support reading/writing these in LAS files.
    offset : 3-vector of float64
        Offset from the origin; defaults to (0, 0, 0).
    """
    def __init__(self, crs=None, offset=None):
        if crs is None:
            crs = 4326
        if offset is None:
            offset = np.zeros(3)
        else:
            offset = np.asarray(offset, dtype=np.float64)

        self.crs = crs
        self.offset = offset

    def transform_crs(self, crs):
        """Transform to other coordinate reference system."""
        source = osr.SpatialReference()
        source.ImportFromEPSG(self.crs)

        target = osr.SpatialReference()
        target.ImportFromEPSG(crs)

        transf = osr.CoordinateTransformation(source, target)
        point = ogr.Geometry(ogr.wkbPoint)
        for i, (x, y, z) in enumerate(self):
            point.AddPoint(x, y, z)
            point.Transform(transf)
            self[i] = point.GetPoint()

        self.crs = crs


def from_las(f, crs=None):
    """Read RegisteredPointCloud from LAS file f."""
    f = laspy.file.File(f)
    pc = RegisteredPointCloud(crs=crs, offset=f.header.offset)
    pc._las_header = f.header

    points = f.points['point']
    np_points = np.empty((len(points), 6), dtype=np.float32)
    np_points[:, 0] = points['X']
    np_points[:, 1] = points['Y']
    np_points[:, 2] = points['Z']
    np_points[:, 3] = points['red']
    np_points[:, 4] = points['green']
    np_points[:, 5] = points['blue']

    # Loss of precision of up to 1.4cm for Via Appia data.
    np_points[:, :3] *= f.header.scale
    pc.from_array(np_points)

    return pc


def to_las(pc, fname, scale=None, store_rgb=True):
    """Write pc to LAS file with name fname. Destroys pc.

    Reuses original scale if no scale is specified; can cause precision loss.
    """
    if hasattr(pc, "_las_header"):
        header = pc._las_header
    else:
        header = laspy.header.Header()

    format_id = 2 if store_rgb else 1       # see LAS standard

    f = laspy.file.File(fname, mode='w', header=header)
    f.header.set_dataformatid(format_id)
    f.header.offset = pc.offset.tolist()
