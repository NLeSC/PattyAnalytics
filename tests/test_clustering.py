import pcl
from patty.segmentation.dbscan import (get_largest_dbscan_clusters,
                                       _get_top_labels)
import numpy as np
from nose.tools import assert_equal, assert_equals


def test_largest_dbscan_cluster():
    """largest_dbscan_clusters returns the correct number of points"""
    # Arrange
    ar = get_one_big_and_10_small_clusters()
    desired_fragment = 0.7
    expected = ar.shape[0] * desired_fragment
    pc = pcl.PointCloud(ar.astype(np.float32))

    # Act
    segmentedpc = get_largest_dbscan_clusters(
        pc, min_return_fragment=desired_fragment, epsilon=3., minpoints=5)
    segmented = segmentedpc.to_array()

    # Assert
    actual = segmented.shape[0]
    message = 'expected: %r, actual: %r' % (expected, actual)
    assert_equals(actual, expected, msg=message)


def get_one_big_and_10_small_clusters():
    ar = np.empty([0, 3], dtype=np.float32)
    rn = np.random.RandomState(1234)
    big = rn.randn(100, 3) + 10
    ar = np.vstack([ar, big])
    for i in range(0, 10):
        small = rn.rand(10, 3) - (10 * i)
        ar = np.vstack([ar, small])
    return ar


def test_get_top_labels():
    # With outliers
    labels = np.array([0, 0, 0, -1, -1, 0, 1, 1, 2, 3])
    assert_equal(_get_top_labels(labels, .6), ([0, 1], 6))

    # Without outliers
    labels = np.array([0, 1, 0, 0, 2, 2, 0, 0, 2, 2])
    assert_equal(_get_top_labels(labels, .6), ([0, 2], 9))
