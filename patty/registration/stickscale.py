import numpy as np
from patty.segmentation.pointCloudMeasurer import measure_length
from patty.segmentation import segment_dbscan
from patty.conversions import extract_mask
from patty.segmentation.segRedStick import get_red_mask

# according to Rens, sticks are .8m and contain 4 segments:
segmentsPerMeter = 5.0


def get_stick_scale(pc, eps=0.1, min_samples=20):
    """Takes a point cloud, as a numpy array, containing only the red segments
    of scale sticks and returns the scale estimation with most support.
    Method:
    pointcloud --dbscan--> clusters --lengthEstimation-->
        lengths --ransac--> best length

    Arguments:
        pc            Point cloud containing only measuring stick segments
                      (only the red, or only the white parts)
        eps           DBSCAN parameter: Maximum distance between two samples
                      for them to be considered as in the same neighborhood.
        min_samples   DBSCAN parameter: The number of samples in a neighborhood
                      for a point to be considered as a core point.
    Returns:
        scale         Estimate of the size of one actual meter in expressed
                      in units of the pointcloud's coordinates.
        confidence    A number expressing the reliability of the estimated
                      scale. Confidence is in [0, 1]. With a confidence greater
                      than .5, the estimate can be considered useable for
                      further calculations.
    """
    cluster_generator = segment_dbscan(
        pc, eps, min_samples, algorithm='kd_tree')

    sizes = [{'len': len(cluster),
              'meter': measure_length(cluster) * segmentsPerMeter}
             for cluster in cluster_generator]
    scale, votes, supportingClusterCount = ransac(sizes)
    confidence = get_confidence_level(votes, supportingClusterCount)
    return scale, confidence


def get_preferred_scale_factor(pointcloud, origScaleFactor):
    # Get reg_scale_2 from red stick
    pc_reds = extract_mask(pointcloud, get_red_mask(pointcloud))
    if len(pc_reds) == 0:
        return origScaleFactor
    # eps and min_samples omitted -- default values
    redScale, redConf = get_stick_scale(pc_reds)

    # Choose best registered scale
    if redConf < 0.5:
        return origScaleFactor
    else:
        return 1.0 / redScale


def ransac(meterClusters, relativeInlierMargin=0.05):
    """Very simple RANSAC implementation for finding the value with most
    support in a list of scale estimates. I.e. only one parameter is searched
    for. The number of points in the cluster on which the scale estimate was
    based is taken into account."""
    biggestClusterSize = max(meterClusters, key=lambda x: x['len'])['meter']
    margin = relativeInlierMargin * biggestClusterSize
    # meterClusters = sorted(meterClusters, key= lambda meterCluster :
    # meterCluster['meter']) # only for printing within loop, doesn't change
    # outcome.

    bestVoteCount = 0
    bestSupport = []
    for clust in meterClusters:
        support = [supportCluster for supportCluster in meterClusters
                   if abs(clust['meter'] - supportCluster['meter']) < margin]
        voteCount = sum([supportCluster['len'] for supportCluster in support])
        # print 'cluster with meter ' + `meter` + ' has ' +
        # `len(meterCluster['cluster'])` + ' own votes and ' + `len(support)` +
        # ' supporting clusters totalling ' + `voteCount` + ' votes.'

        if voteCount > bestVoteCount:
            bestVoteCount = voteCount
            bestSupport = support

    estimate = np.mean([supportCluster['meter']
                       for supportCluster in bestSupport])
    return estimate, bestVoteCount, len(bestSupport)


def get_confidence_level(votes, supportingClusterCount):
    """ Gives a confidence score in [0, 1]. This score should give the
    user some idea of the reliability of the estimate. Above .5 can be
    considered usable.

    Arguments:
        votes: integer
            sum of number of points in inlying red clusters found
        supportingClusterCount: integer
            number of inlying red clusters found
    """
    # Higher number of votes implies more detail which gives us more
    # confidence (but 500 is enough)
    upperLimitVotes = 500.0
    lowerLimitVotes = 0.0
    voteBasedConfidence = get_score_in_interval(
        votes, lowerLimitVotes, upperLimitVotes)

    # Higher number of supporting clusters tells us multiple independent
    # sources gave this estimate
    upperLimitClusters = 3.0
    lowerLimitClusters = 0.0
    clusterBasedConfidence = get_score_in_interval(
        supportingClusterCount, lowerLimitClusters, upperLimitClusters)

    return min(voteBasedConfidence, clusterBasedConfidence)


def get_score_in_interval(value, lowerLimit, upperLimit):
    return (min(value, upperLimit) - lowerLimit) / (upperLimit - lowerLimit)
