import numpy as np
from sklearn.decomposition import PCA

def pcaRotate(pc):
    pca = PCA(n_components=3)
    pca.fit(pc.to_array()[:,:3])

    # Use 3 Principal components as rotation matrix.
    # Embed components on 4x4 matrix
    t = pca.components_
    t = np.concatenate((t,np.zeros((1,3))), axis=0)
    t = np.concatenate((t,np.zeros((4,1))), axis=1)
    t[3,3] = 1

    return pc.transform(t)
