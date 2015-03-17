import numpy as np
from sklearn.decomposition import PCA


def measureLength(pointCloud):
    """Returns the length of a point cloud in its longest direction."""
    if len(pointCloud) == 0:
        return 0

    pca = PCA(n_components=1)
    pc_array = np.asarray(pointCloud)
    pca.fit(pc_array)
    primary_axis = np.dot(pc_array, np.transpose(pca.components_))[:, 0]
    return np.max(primary_axis) - np.min(primary_axis)
