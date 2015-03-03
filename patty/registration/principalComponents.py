import numpy as np
from sklearn.decomposition import PCA

def principal_axes_rotation(data):
    '''Find the 3 princial axis of the XYZ array, and the rotation to align it to the x,y, and z axis.

    Arguments:
        data    pointcloud
    Returns:
        transformation matrix
    '''
    pca = PCA(n_components=3)
    pca.fit(np.asarray(data))
    transform = np.zeros((4,4))
    transform[:3,:3] = np.array(pca.components_)
    transform[3,3] = 1.0
    
    return np.matrix(transform)

