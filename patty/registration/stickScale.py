import pcl
import numpy as np
import argparse
from sklearn.cluster import dbscan
from patty.segmentation.pointCloudMeasurer import measureLength

def getStickScale(array, eps = 0.1, minSamples = 20):
    """Takes a point cloud, as a numpy array, containing only the red segments 
    of scale sticks and returns the median, min and max of the scale estimation. 
    Nb. Outliers with respect to individual stick measurements have not been 
    removed so be warned when using the min and max values."""
    ar = array[:, [0,1,2]]
    clusters = getClusters(ar, eps, minSamples)
    segmentLengths = np.array(map(lambda cluster : measureLength(cluster), clusters))
    stickLengths = segmentLengths * 4.
    return np.median(stickLengths), np.min(stickLengths), np.max(stickLengths)

def getClusters(ar, eps, minSamples):
    """Returns a list of clustered point clouds."""
    _, labels = dbscan(ar, eps=eps, min_samples=minSamples, algorithm='kd_tree')
        
    uniqueLabels = np.unique(labels)
    for label in uniqueLabels:
        if label == -1:
            continue            
        mask = labels == label
        indices = np.where(mask)
        yield ar[indices]


        
    
    
