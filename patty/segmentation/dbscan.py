"""Segmentation using DBSCAN.

Usage:
    dbscan pointcloud <epsilon> <minpoints>
"""

from sklearn.cluster import dbscan
import numpy as np

def dbscan_labels(pointcloud, epsilon, minpoints):
    ''' returns a number of clusters'''
    _, labels = dbscan(np.asarray(pointcloud), eps=epsilon, min_samples=minpoints, algorithm='kd_tree')
    return labels

def segment_dbscan(pointcloud, epsilon, minpoints):
    ''' returns a number of clusters'''
    labels = dbscan_labels(pointcloud, epsilon, minpoints)
    
    clusters = []
    
    for label in np.unique(labels[labels != -1]):
        clusters.append(pointcloud.extract(np.where(labels == label)[0]))
    
    return clusters
    
def largest_dbscan_cluster(pointcloud, epsilon=0.1, minpoints=250):
    ''' returns a number of clusters'''
    labels = dbscan_labels(pointcloud, epsilon, minpoints)
    
    bins = np.bincount(labels)
    max_label = np.argmax(bins)
    
    return pointcloud.extract(np.where(labels == max_label)[0])
