import numpy as np
from sklearn.cluster import dbscan
from patty.segmentation.pointCloudMeasurer import measureLength

# according to Rens, sticks are .8m and contain 4 segments:    
segmentsPerMeter = 5.0

def getStickScale(pc, eps = 0.1, minSamples = 20):
    """Takes a point cloud, as a numpy array, containing only the red segments 
    of scale sticks and returns the scale estimation with most support.
    Method: 
    pointcloud --dbscan--> clusters --lengthEstimation--> lengths --ransac--> best length
    
    Arguments:
        pc            Point cloud containing only measuring stick segments
                      (only the red, or only the white parts)
        eps           DBSCAN parameter: Maximum distance between two samples 
                      for them to be considered as in the same neighborhood.
        minSamples    DBSCAN parameter: The number of samples in a neighborhood 
                      for a point to be considered as a core point.
    Returns:
        scale         Estimate of the size of one actual meter in expressed
                      in units of the pointcloud's coordinates.
        confidence    A number expressing the reliability of the estimated
                      scale. Confidence is in [0, 1]. With a confidence greater
                      than .5, the estimate can be considered useable for 
                      further calculations.
    """
    ar = np.asarray(pc)[:, [0,1,2]]    
    clusters = getClusters(ar, eps, minSamples)
    meterClusters = [{'meter':measureLength(cluster) * segmentsPerMeter, 'cluster': cluster} for cluster in clusters]
    scale, votes, supportingClusterCount = ransac(meterClusters)
    confidence = getConfidenceLevel(votes, supportingClusterCount)
    return scale, confidence
    
def ransac(meterClusters, relativeInlierMargin = 0.05):
    """Very simple RANSAC implementation for finding the value with most
    support in a list of scale estimates. I.e. only one parameter is searched 
    for. The number of points in the cluster on which the scale estimate was 
    based is taken into account."""    
    biggestClusterSize = sorted(meterClusters, key= lambda meterCluster : len(meterCluster['cluster']))[-1]['meter']
    margin = relativeInlierMargin * biggestClusterSize
    #meterClusters = sorted(meterClusters, key= lambda meterCluster : meterCluster['meter']) # only for printing within loop, doesn't change outcome.
        
    best = None
    bestVoteCount = 0
    bestSupport = []    
    for meterCluster in meterClusters:
        meter = meterCluster['meter']
        support = [meterCluster for meterCluster in meterClusters if abs(meter - meterCluster['meter']) < margin]
        voteCount = sum([len(meterCluster['cluster']) for meterCluster in support])
        #print 'cluster with meter ' + `meter` + ' has ' + `len(meterCluster['cluster'])` + ' own votes and ' + `len(support)` + ' supporting clusters totalling ' + `voteCount` + ' votes.'
        
        if voteCount > bestVoteCount:
            best = meterCluster
            bestVoteCount = voteCount
            bestSupport = support

    estimate = np.mean([meterCluster['meter'] for meterCluster in bestSupport])    
    return estimate, bestVoteCount, len(bestSupport)
    

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

def getConfidenceLevel(votes, supportingClusterCount):
    """ Gives a confidence score in [0, 1]. This score should give the
    user some idea of the reliability of the estimate. Above .5 can be
    considered usable."""
    # Higher number of votes implies more detail which gives us more confidence (but 500 is enough)
    upperLimitVotes = 500.0
    lowerLimitVotes = 0.0
    voteBasedConfidence = getScoreInInterval(votes, lowerLimitVotes, upperLimitVotes)
    
    # Higher number of supporting clusters tells us multiple independent sources gave this estimate
    upperLimitClusters = 3.0
    lowerLimitClusters = 0.0
    clusterBasedConfidence = getScoreInInterval(supportingClusterCount, lowerLimitClusters, upperLimitClusters)
    
    return min(voteBasedConfidence, clusterBasedConfidence)
    
def getScoreInInterval(value, lowerLimit, upperLimit):
    return (min(value, upperLimit) -lowerLimit) / (upperLimit - lowerLimit)
