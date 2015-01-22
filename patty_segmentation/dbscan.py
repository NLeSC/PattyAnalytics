"""Segmentation using DBSCAN.

Usage:
    dbscan pointcloud <epsilon> <minpoints>
"""

from sklearn.cluster import dbscan
import numpy as np

def segment_dbscan(pointcloud, epsilon, minpoints):
    ''' returns a number of clusters'''
    _, labels = dbscan(np.asarray(pointcloud), eps=epsilon, min_samples=minpoints, algorithm='kd_tree')
    
    clusters = []
    
    for label in np.unique(labels[labels != -1]):
        clusters.append(pointcloud.extract(np.where(labels == label)[0]))
    
    return clusters