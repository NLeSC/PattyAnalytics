import pcl
from patty.segmentation.dbscan import (get_largest_dbscan_clusters,
                                       _get_top_labels, dbscan_labels)
import numpy as np
import unittest
from nose.tools import assert_equal, assert_equals


def test_largest_dbscan_clusters():
    """get_largest_dbscan_clusters returns at least desired fragment of
    points"""
    # Arrange
    pc = get_one_big_and_10_small_clusters()
    desired_fragment = 0.7
    expected = pc.size * desired_fragment

    # Act
    segmentedpc = get_largest_dbscan_clusters(
        pc, min_return_fragment=desired_fragment, epsilon=3., minpoints=5)
    segmented = segmentedpc.to_array()

    # Assert
    actual = segmented.shape[0]
    message = 'expected: %r, actual: %r' % (expected, actual)
    assert_equals(actual, expected, msg=message)


def get_one_big_and_10_small_clusters():
    '''Create clusters as advertised'''
    ar = np.empty([0, 3], dtype=np.float32)
    rn = np.random.RandomState(1234)
    big = rn.randn(100, 3) + 10
    ar = np.vstack([ar, big])
    for i in range(0, 10):
        small = rn.rand(10, 3) - (10 * i)
        ar = np.vstack([ar, small])
    pc = pcl.PointCloud(ar.astype(np.float32))
    return pc


def test_get_top_labels():
    '''Test _get_top_labels function from patty.segmentation.dbscan'''
    # With outliers
    labels = np.array([0, 0, 0, -1, -1, 0, 1, 1, 2, 3])
    assert_equal(_get_top_labels(labels, .6), ([0, 1], 6))

    # Without outliers
    labels = np.array([0, 1, 0, 0, 2, 2, 0, 0, 2, 2])
    assert_equal(_get_top_labels(labels, .6), ([0, 2], 9))


class Dbscan_labels(unittest.TestCase):

    def setUp(self):
        self.pc = pcl.PointCloudXYZRGB(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    def test_dbscan_labels(self):
        '''Points with different color should be one cluster with
        rgb_weight = 0'''
        labels = dbscan_labels(self.pc, 0.1, 1, rgb_weight=0)
        labelcount = len(np.unique(labels))
        assert_equals(labelcount, 1)

    def test_dbscan_labels_colored(self):
        '''Points with different color should be two clusters with
        rgb_weight = 1'''
        labels = dbscan_labels(self.pc, 0.1, 1, rgb_weight=1)
        labelcount = len(np.unique(labels))
        assert_equals(labelcount, 2)
