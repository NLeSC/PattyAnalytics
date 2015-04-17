from __future__ import print_function
import numpy as np
import osgeo.osr as osr


def is_registered(pointcloud):
    """
    Returns True when a pointcloud is registered; ie coordinates are relative
    to a specific spatial reference system or offset.

    In that case, first transform one pointcloud to the reference system
    of the other, before doing processing on the points:

       set_srs(pcA, same_as=pcB)
    """
    return hasattr(pointcloud, 'srs') or hasattr(pointcloud, 'offset')


def same_srs(pc_one, pc_two):
    """
    True if the two pointclouds have the same coordinate system

    Arguments:
        pc_one : pcl.PointCloud
        pc_two : pc..PointCloud
    """

    is_reg_one = is_registered(pc_one)
    is_reg_two = is_registered(pc_two)

    # both pointcloud are pure pcl: no offset nor SRS
    if not is_reg_one and not is_reg_two:
        return True

    # only one of the pointclouds is registered
    if ((not is_reg_one) and is_reg_two) or ((not is_reg_two) and is_reg_one):
        return False

    srs_one = None
    srs_two = None
    try:
        srs_one = pc_one.srs
        srs_two = pc_two.srs

        if not srs_one.IsSame(srs_two):
            # SRS present, but different
            return False
    except:
        # one of the pointclouds does not have a SRS
        return False

    off_one = None
    off_two = None
    try:
        off_one = pc_one.offset
        off_two = pc_two.offset
    except:
        # one of the pointclouds does not have an offset
        return False

    # absolute(off_one - off_two) <= (atol + rtol * absolute(off_two))
    if not np.allclose(off_one, off_two, rtol=1e-06, atol=1e-08):
        return False

    return True


def set_srs(pc, srs=None, offset=np.array([0, 0, 0], dtype=np.float64),
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
          can be checked with patty.utils.same_srs().

    NOTE: To add a SRS to a point cloud, or to update incorrect metadata,
          use force_srs().

    Example:

        # set the SRS to lat/lon,
        # don't use an offset, so it defaults to [0,0,0]
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
        raise TypeError("Pointcloud is not registered")
        return None

    update_offset = False
    update_srs = False

    if same_as:

        # take offset and SRS from reference pointcloud
        update_offset = True
        update_srs = True
        if is_registered(same_as):
            newsrs = same_as.srs
            newoffset = same_as.offset
        else:
            raise TypeError("Reference pointcloud is not registered")
    else:

        # take offset and SRS from arguments

        # sanitize offset
        if offset is not None:
            update_offset = True

            newoffset = np.array(offset, dtype=np.float64)
            if len(newoffset) != 3:
                raise TypeError("Offset should be an np.array([3])")

        # sanitize SRS
        if srs is not None:

            # argument is a SRS, use it
            if type(srs) == type(osr.SpatialReference()):
                update_srs = True
                newsrs = srs

            # argument is not an SRS, try to convert it to one
            elif isinstance(srs, osr.SpatialReference):
                newsrs = osr.SpatialReference()
                if newsrs.SetFromUserInput(srs) == 0:
                    update_srs = True

            # illegal input
            else:
                raise TypeError(
                    "SRS should be a string or a osr.SpatialReference")
    # Apply

    # add old offset
    data = np.asarray(pc)
    precise_points = np.array(data, dtype=np.float64) + pc.offset

    # do transformation, this resets the offset to 0
    if update_srs and not pc.srs.IsSame(newsrs):
        try:
            transform = osr.CoordinateTransformation(pc.srs, newsrs)
            precise_points = np.array(transform.TransformPoints(precise_points),
                                      dtype=np.float64)
            pc.srs = newsrs.Clone()
            pc.offset = np.array([0, 0, 0], dtype=np.float64)
        except:
            print("WARNING, CAN'T DO COORDINATE TRANSFORMATION")

    # substract new offset
    if update_offset:
        precise_points -= newoffset
        pc.offset = np.array(newoffset, dtype=np.float64)

    # copy the float64 to the pointcloud
    data[...] = np.asarray(precise_points, dtype=np.float32)

    return pc


def force_srs(pc, srs=None, offset=None, same_as=None):
    """
    Set a spatial reference system (SRS) and offset for a pointcloud.
    Either give a SRS and offset, or a reference pointcloud
    This function affects the metadata only.

    This is the recommended way to turn a python-pcl pointcloud to a
    registerd pointcloud with absolute coordiantes.

    NOTE: To change the SRS for an already registered pointcloud, use set_srs()

    Example:

        # set the SRS to lat/lon, leave offset unchanged
        force_srs( pc, srs="EPSG:4326" )

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
            pc.offset = np.array(same_as.offset, dtype=np.float64)
    else:
        if type(srs) == type(osr.SpatialReference()):
            pc.srs = srs.Clone()
        elif srs is not None:
            pc.srs = osr.SpatialReference()
            pc.srs.SetFromUserInput(srs)
        else:
            pc.srs = osr.SpatialReference()

        if offset is not None:
            offset = np.asarray(offset, dtype=np.float64)
            if len(offset) != 3:
                raise TypeError("Offset should be an np.array([3])")
            else:
                pc.offset = offset

    return pc
