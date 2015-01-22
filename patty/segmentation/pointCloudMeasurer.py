import numpy as np
from sklearn.decomposition import PCA
    
def measureLength(pointCloud):
    """Returns the length of a point cloud in its longest direction."""           
    pca = PCA(n_components = 1)
    pca.fit(pointCloud)    
    transformed = np.dot(pointCloud, np.transpose(pca.components_))
    length = np.max(transformed, 0)[0] - np.min(transformed, 0)[0]
    return length
