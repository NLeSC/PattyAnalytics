import pcl
import numpy as np
import argparse
from sklearn.cluster import dbscan
from patty.segmentation.pointCloudMeasurer import measureLength
from patty.registration.stickScale import getStickScale

def getStickScale(ar, eps, minSamples):
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

if __name__=='__main__':
    """Takes a point cloud containing only the red segments of scale sticks and returns the median, min and max of the scale estimation."""
    parser = argparse.ArgumentParser(description='Takes a point cloud containing only the red segments of scale sticks and returns the median, min and max of the scale estimation.')
    parser.add_argument('-i','--inFile', required=True, type=str, help='Input PCD/PLY file')
    parser.add_argument('-e','--eps',required=False, default=0.1, type=float, help='The maximum distance between two samples for them to be considered as in the same neighborhood.' )
    parser.add_argument('-s','--minSamples',required=False, default=20, type=int, help='The number of samples in a neighborhood for a point to be considered as a core point.' )
    args = parser.parse_args()

    pc = pcl.load(args.inFile)
    ar = np.asarray(pc)

    print(getStickScale(ar, args.eps, args.minSamples))
