"""Segmentation using DBSCAN.

Usage:
    dbscan pointcloud <epsilon> <minpoints>
"""

from sklearn.cluster import dbscan
import numpy as np

def dbscan_labels(pointcloud, epsilon, minpoints):
    ''' returns an array of point-labels of all dbscan clusters found '''
    _, labels = dbscan(np.asarray(pointcloud), eps=epsilon, min_samples=minpoints, algorithm='kd_tree')
    return labels

def segment_dbscan(pointcloud, epsilon, minpoints):
    ''' returns an array of pointclouds, each a cluster'''
    labels = dbscan_labels(pointcloud, epsilon, minpoints)
    
    clusters = []
    
    for label in np.unique(labels[labels != -1]):
        clusters.append(pointcloud.extract(np.where(labels == label)[0]))
    
    return clusters
    
def largest_dbscan_cluster(pointcloud, epsilon=0.1, minpoints=250):
    ''' returns the largest cluster found in the pointcloud'''
    labels = dbscan_labels(pointcloud, epsilon, minpoints)
    
    print np.unique(labels)

    # Labels start at -1, so increase all by 1.
    bins = np.bincount(np.array(labels) + 1)
    
    # Pointcloud is the only cluster
    if len(bins)<2:
        return pointcloud

    # Move it back
    max_label = np.argmax(bins[1:]) - 1
    
    return pointcloud.extract(np.where(labels == max_label)[0])
