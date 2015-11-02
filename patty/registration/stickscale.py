import numpy as np
from patty.segmentation import segment_dbscan
from patty.utils import extract_mask
from patty.utils import measure_length
from patty.segmentation.segRedStick import get_red_mask

# according to Rens, sticks are .8m and contain 4 segments:
SEGMENTS_PER_METER = 5.0


def get_stick_scale(pointcloud, eps=0.1, min_samples=20):
    """Takes a point cloud, as a numpy array, looks for red segments
    of scale sticks and returns the scale estimation with most support.
    Method:
    pointcloud --dbscan--> clusters --lengthEstimation-->
        lengths --ransac--> best length
    Arguments:
        pointcloud    Point cloud containing only measuring stick segments
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

    # quickly return for trivial case
    if pointcloud.size == 0:
        return 1, 0

    # find the red segments to measure
    pc_reds = extract_mask(pointcloud, get_red_mask(pointcloud))
    if len(pc_reds) == 0:
        # unit scale, zero confidence (ie. any other estimation is better)
        return 1.0, 0.0

    cluster_generator = segment_dbscan(
        pc_reds, eps, min_samples, algorithm='kd_tree')

    sizes = [{'len': len(cluster),
              'meter': measure_length(cluster) * SEGMENTS_PER_METER}
             for cluster in cluster_generator]

    if len(sizes) == 0:
        return 1.0, 0.0

    scale, votes, n_clusters = ransac(sizes)
    confidence = get_confidence_level(votes, n_clusters)
    return scale, confidence


def ransac(meter_clusters, rel_inlier_margin=0.05):
    """Very simple RANSAC implementation for finding the value with most
    support in a list of scale estimates. I.e. only one parameter is searched
    for. The number of points in the cluster on which the scale estimate was
    based is taken into account."""
    max_cluster_size = max(meter_clusters, key=lambda x: x['len'])['meter']
    margin = rel_inlier_margin * max_cluster_size
    # meter_clusters = sorted(meter_clusters, key= lambda meterCluster :
    # meterCluster['meter']) # only for printing within loop, doesn't change
    # outcome.

    best_vote_count = 0
    best_support = []
    for clust in meter_clusters:
        support = [supportCluster for supportCluster in meter_clusters
                   if abs(clust['meter'] - supportCluster['meter']) < margin]
        vote_count = sum([supportCluster['len'] for supportCluster in support])
        # print 'cluster with meter ' + `meter` + ' has ' +
        # `len(meterCluster['cluster'])` + ' own votes and ' + `len(support)` +
        # ' supporting clusters totalling ' + `vote_count` + ' votes.'

        if vote_count > best_vote_count:
            best_vote_count = vote_count
            best_support = support

    estimate = np.mean([supportCluster['meter']
                        for supportCluster in best_support])
    return estimate, best_vote_count, len(best_support)


def get_confidence_level(votes, n_clusters):
    """ Gives a confidence score in [0, 1]. This score should give the
    user some idea of the reliability of the estimate. Above .5 can be
    considered usable.

    Arguments:
        votes: integer
            sum of number of points in inlying red clusters found
        n_clusters: integer
            number of inlying red clusters found
    """
    # Higher number of votes implies more detail which gives us more
    # confidence (but 500 is enough)
    upper_lim_votes = 500.0
    lower_lim_votes = 0.0
    vote_based_conf = get_score_in_interval(
        votes, lower_lim_votes, upper_lim_votes)

    # Higher number of supporting clusters tells us multiple independent
    # sources gave this estimate
    upper_lim_clusters = 3.0
    lower_lim_clusters = 0.0
    cluster_based_confidence = get_score_in_interval(
        n_clusters, lower_lim_clusters, upper_lim_clusters)

    return min(vote_based_conf, cluster_based_confidence)


def get_score_in_interval(value, lower_lim, upper_lim):
    return (min(value, upper_lim) - lower_lim) / (upper_lim - lower_lim)
