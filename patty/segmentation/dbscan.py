"""
Point cloud segmentation using the DBSCAN clustering algorithm.

DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
Finds core samples of high density and expands clusters from them.
Good for data which contains clusters of similar density.

See the scikit-learn documentation for reference:
http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html.
"""
import numpy as np
from sklearn.cluster import dbscan
from patty.conversions import extract_mask


def dbscan_labels(pointcloud, epsilon, minpoints, rgb_weight=0,
                  algorithm='ball_tree'):
    '''
    Find an array of point-labels of clusters found by the DBSCAN algorithm.

    Parameters
    ----------
    pointcloud : pcl.PointCloud
        Input pointcloud.
    epsilon : float
        Neighborhood radius for DBSCAN.
    minpoints : integer
        Minimum neighborhood density for DBSCAN.
    rgb_weight : float, optional
        If non-zero, cluster on color information as well as location;
        specifies the relative weight of the RGB components to spatial
        coordinates in distance computations.
        (RGB values have wildly different scales than spatial coordinates.)

    Returns
    -------
    labels : Sequence
        A sequence of labels per point. Label -1 indicates a point does not
        belong to any cluster, other labels indicate the cluster number a
        point belongs to.
    '''

    if rgb_weight > 0:
        X = pointcloud.to_array()
        X[:, 3:] *= rgb_weight
    else:
        X = pointcloud

    _, labels = dbscan(X, eps=epsilon, min_samples=minpoints,
                       algorithm=algorithm)
    return np.asarray(labels)


def segment_dbscan(pointcloud, epsilon, minpoints, **kwargs):
    """Run the DBSCAN clustering+outlier detection algorithm on pointcloud.

    Parameters
    ----------
    pointcloud : pcl.PointCloud
        Input pointcloud.
    epsilon : float
        Neighborhood radius for DBSCAN.
    minpoints : integer
        Minimum neighborhood density for DBSCAN.
    **kwargs : keyword arguments, optional
        arguments passed to _dbscan_labels

    Returns
    -------
    clusters : iterable over registered PointCloud
    """
    labels = dbscan_labels(pointcloud, epsilon, minpoints, **kwargs)

    return (extract_mask(pointcloud, labels == label)
            for label in np.unique(labels[labels != -1]))


def largest_dbscan_cluster(pointcloud, epsilon=0.1, minpoints=250,
                           rgb_weight=0):
    '''
    Finds the largest cluster found in the pointcloud

    Parameters
    ----------
    pointcloud : pcl.PointCloud
        Input pointcloud.
    epsilon : float
        Neighborhood radius for DBSCAN.
    minpoints : integer
        Minimum neighborhood density for DBSCAN.
    rgb_weight : float, optional
        If non-zero, cluster on color information as well as location;
        specifies the relative weight of the RGB components to spatial
        coordinates in distance computations.
        (RGB values have wildly different scales than spatial coordinates.)

    Returns
    -------
    cluster : pcl.PointCloud
        Registered pointcloud of the largest cluster found by dbscan.
    '''
    labels = dbscan_labels(
        pointcloud, epsilon, minpoints, rgb_weight=rgb_weight)

    # Labels start at -1, so increase all by 1.
    bins = np.bincount(labels + 1)

    # Pointcloud is the only cluster
    if len(bins) < 2:
        return extract_mask(pointcloud, np.ones(len(pointcloud), dtype=bool))

    # Indexes are automatically moved one back by [1:]
    max_label = np.argmax(bins[1:])
    return extract_mask(pointcloud, labels == max_label)


def get_largest_dbscan_clusters(pointcloud, min_return_fragment=0.7,
                                epsilon=0.1, minpoints=250, rgb_weight=0):
    '''
    Finds the largest clusters containing together at least min_return_fragment
    of the complete point cloud. In case less points belong to clusters, all
    clustered points are returned.

    Parameters
    ----------
    pointcloud : pcl.PointCloud
        Input pointcloud.
    min_return_fragment : float
        Minimum desired fragment of pointcloud to be returned
    epsilon : float
        Neighborhood radius for DBSCAN.
    minpoints : integer
        Minimum neighborhood density for DBSCAN.
    rgb_weight : float, optional
        If non-zero, cluster on color information as well as location;
        specifies the relative weight of the RGB components to spatial
        coordinates in distance computations.
        (RGB values have wildly different scales than spatial coordinates.)

    Returns
    -------
    cluster : pcl.PointCloud
        Registered pointcloud of the largest cluster found by dbscan.
    '''
    labels = dbscan_labels(pointcloud, epsilon, minpoints,
                           rgb_weight=rgb_weight).astype(np.int64)
    selection, selected_count = _get_top_labels(labels, min_return_fragment)

    # No clusters were found
    if selected_count < min_return_fragment * len(labels):
        return extract_mask(pointcloud, np.ones(len(pointcloud), dtype=bool))
    else:
        mask = [label in selection for label in labels]
        return extract_mask(pointcloud, mask)


def _get_top_labels(labels, min_return_fragment):
    """Return labels of the smallest set of clusters that contain at least
    min_return_fragment of the points (or everything)."""

    # +1 to make bincount happy, [1:] to get rid of outliers.
    bins = np.bincount(labels + 1)[1:]
    labelbinpairs = sorted(enumerate(bins), key=lambda x: x[1])

    total = len(labels)
    minimum = min_return_fragment * total
    selected = []
    selected_count = 0
    while selected_count < minimum and len(labelbinpairs) > 0:
        label, count = labelbinpairs.pop()
        selected.append(label)
        selected_count += count
    return selected, selected_count
