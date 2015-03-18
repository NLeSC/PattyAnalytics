import numpy as np
from sklearn.decomposition import PCA


def measure_length(pointcloud):
    """Returns the length of a point cloud in its longest direction."""
    if len(pointcloud) < 2:
        return 0

    pca = PCA(n_components=1)
    pc_array = np.asarray(pointcloud)
    pca.fit(pc_array)
    primary_axis = np.dot(pc_array, np.transpose(pca.components_))[:, 0]
    return np.max(primary_axis) - np.min(primary_axis)
