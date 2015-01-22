import numpy as np
from sklearn.decomposition import PCA
class PointCloudMeasurer:
    
    
    def measureLength(self, pointCloud):
        np.mean(pointCloud)        
        pca = PCA(n_components = 1)
        pca.fit(pointCloud)
        
        t = pca.components_
        
        
        
        
        
        
        
        print pointCloud        
        transformed = np.dot(t, pointCloud)
        print t
        print transformed        
        return np.max(transformed, 0)[0] - np.min(transformed, 0)[0]
