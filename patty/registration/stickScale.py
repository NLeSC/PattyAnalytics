import numpy as np
from patty.segmentation.pointCloudMeasurer import measureLength
from patty.segmentation import segment_dbscan
from patty.conversions import extract_mask
from patty.segmentation.segRedStick import getRedMask

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
    clusters = segment_dbscan(pc, eps, minSamples, algorithm='kd-tree')
    for cluster in clusters:
        cluster.meter = measureLength(cluster) * segmentsPerMeter
    scale, votes, supportingClusterCount = ransac(clusters)
    confidence = getConfidenceLevel(votes, supportingClusterCount)
    return scale, confidence

def getPreferredScaleFactor(pointcloud, origScaleFactor):
    # Get reg_scale_2 from red stick
    pcReds = extract_mask(pointcloud, getRedMask(pointcloud))
    redScale, redConf = getStickScale(pcReds) # eps and minSamples omitted -- default values

    # Choose best registered scale
    if redConf<0.5:
        return origScaleFactor
    else:
        return 1.0 / redScale

def ransac(meterClusters, relativeInlierMargin = 0.05):
    """Very simple RANSAC implementation for finding the value with most
    support in a list of scale estimates. I.e. only one parameter is searched 
    for. The number of points in the cluster on which the scale estimate was 
    based is taken into account."""    
    biggestClusterSize = max(meterClusters, key=len).meter
    margin = relativeInlierMargin * biggestClusterSize
    #meterClusters = sorted(meterClusters, key= lambda meterCluster : meterCluster['meter']) # only for printing within loop, doesn't change outcome.
    
    bestVoteCount = 0
    bestSupport = []
    for meterCluster in meterClusters:
        support = [supportCluster for supportCluster in meterClusters if abs(meterCluster.meter - supportCluster.meter) < margin]
        voteCount = sum([len(supportCluster) for supportCluster in support])
        #print 'cluster with meter ' + `meter` + ' has ' + `len(meterCluster['cluster'])` + ' own votes and ' + `len(support)` + ' supporting clusters totalling ' + `voteCount` + ' votes.'
        
        if voteCount > bestVoteCount:
            bestVoteCount = voteCount
            bestSupport = support

    estimate = np.mean([supportCluster.meter for supportCluster in bestSupport])
    return estimate, bestVoteCount, len(bestSupport)

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
