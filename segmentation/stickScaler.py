import pcl
import numpy as np
import argparse
from sklearn.cluster import dbscan
from sklearn.decomposition import PCA
import clusterMeasurer

if __name__=='__main__':
    """"""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i','--inFile', required=True, type=str, help='Input PCD/PLY file')    
    parser.add_argument('-e','--eps',required=False, default=0.5, type=float, help='The maximum distance between two samples for them to be considered as in the same neighborhood.' )
    parser.add_argument('-s','--minSamples',required=False, default=5, type=int, help='The number of samples in a neighborhood for a point to be considered as a core point.' )
    args = parser.parse_args()
    
    pc = pcl.load(args.inFile)
    ar = np.asarray(pc)
    
    
    
    # Get clusters
    _, labels = dbscan(ar, eps=args.eps, min_samples=args.minSamples, algorithm='kd_tree')
    
    clusters = []
    uniqueLabels = np.unique(labels)
    for label in uniqueLabels:
        mask = labels == label
        print "Label " + `label` + " has " + `sum(mask)` + " points."
        
        
        indices = np.where(mask)
        clusters.append(ar[indices])
    # /Get clusters
    
    for cluster in clusters:
        clusterMeasurer
    
    
