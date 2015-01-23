import numpy as np
from sklearn.cluster import dbscan

def _dbscan_labels(pointcloud, epsilon, minpoints):
    ''' returns an array of point-labels of all dbscan clusters found '''
    _, labels = dbscan(pointcloud, eps=epsilon, min_samples=minpoints,
                       algorithm='ball_tree')
    return labels

def segment_dbscan(pointcloud, epsilon, minpoints, rgb_weight=0):
    """Run the DBSCAN clustering+outlier detection algorithm on pointcloud.

    Parameters
    ----------
    pointcloud : pcl.PointCloud
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
    clusters : iterable over PointCloud
    """
    if rgb_weight > 0:
        X = pointcloud.to_array()
        print(repr(X.mean(axis=0)))
        X[:, 3:] *= rgb_weight
        print(repr(X.mean(axis=0)))
    else:
        X = np.asarray(pointcloud)
    labels = _dbscan_labels(X, epsilon, minpoints)

    return (pointcloud.extract(np.where(labels == label)[0])
            for label in np.unique(labels[labels != -1]))


def largest_dbscan_cluster(pointcloud, epsilon=0.1, minpoints=250):
    ''' returns the largest cluster found in the pointcloud'''
    labels = _dbscan_labels(pointcloud, epsilon, minpoints)

    # Labels start at -1, so increase all by 1.
    bins = np.bincount(np.asarray(labels) + 1)
    
    # Pointcloud is the only cluster
    if len(bins) < 2:
        return pointcloud

    # Move it back
    max_label = np.argmax(bins) - 1
    return pointcloud.extract(np.where(labels == max_label)[0])
