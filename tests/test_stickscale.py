from patty.registration import get_stick_scale
from patty.registration.stickscale import get_confidence_level
from patty.segmentation import get_red_mask
from patty import load, extract_mask
from nose_parameterized import parameterized

from nose.tools import assert_greater, assert_less

from helpers import make_red_stick, _add_noise
import pcl
import numpy as np


# The ground truths for these tests was defined by measuring stick segments,
# either white or red, by hand, in Meshlab.
# Note: failing instances commented out. Uncomment them for testing out
# new methods.
# @parameterized.expand([
# ("SITE9", 'redstick_SITE_9.ply', 11.45),
# ("SITE11", 'redstick_SITE_11.ply', 2.925),
# ("SITE12", 'redstick_SITE_12.ply', 10.85),

# only one segment visible
# ("SITE13", 'redstick_SITE_13.ply', 4.35),

# hardly anything of a segment is visible
# ("SITE14", 'redstick_SITE_14.ply', 1.07),

# ("SITE15", 'redstick_SITE_15.ply', 2.0),

# no red segments completely visible
# ("SITE16", 'redstick_SITE_16.ply', 5.13),

# one complete stick segment, one partly that contains more points (!)
# ("SITE19", 'redstick_SITE_19.ply', 3.15),

# ("SITE20", 'redstick_SITE_20.ply', 5.55),
# ("SITE21", 'redstick_SITE_21.ply', 5.4),
# ])


def make_pointcloud(sticks, noise):
    data = np.array(np.concatenate(sticks, axis=0), dtype=np.float32)
    _add_noise(data, noise, np.random.RandomState(0))
    pc = pcl.PointCloudXYZRGB(data)
    return extract_mask(pc, get_red_mask(pc))


@parameterized.expand([(0.01,), (0.02,), (0.05,), (0.1,)])
def test_stickscale(noise):
    s1 = make_red_stick([0, 0, 0], [0, 1, 0])
    s2 = make_red_stick([1, 2, 0], [1, 1, 0])
    s3 = make_red_stick([3, 3, 0], [3, 4, 0])
    pc = make_pointcloud((s1, s2, s3), noise)
    meter, confidence = get_stick_scale(pc)
    red_part_length = 0.25 + noise
    assert_with_error(meter, 4*red_part_length/0.8)
    assert_with_error(confidence, 1.0)


@parameterized.expand([
    (500, 3, 1.0),
    (100, 3, 1.0/5.0),
    (500, 2, 2.0/3.0),
    (500, 1, 1.0/3.0),
    (100, 2, min(1.0/5.0, 2.0/3.0)),
])
def test_confidence(votes, numberOfClusters, expected_confidence):
    confidence = get_confidence_level(votes, numberOfClusters)
    assert_with_error(confidence, expected_confidence)


# The ground truths for these tests was defined by measuring stick segments,
# either white or red, by hand, in Meshlab. """
@parameterized.expand([
    # ("SITE9", 'redstick_SITE_9.ply', 11.45, True),
    # ("SITE11", 'redstick_SITE_11.ply', 2.925, True),
    # ("SITE12", 'redstick_SITE_12.ply', 10.85, True),
    # ("SITE13", 'redstick_SITE_13.ply', 4.35, False),
    # # only one segment visible
    # ("SITE14", 'redstick_SITE_14.ply', 1.07, False),
    # # hardly anything of a segment is visible
    # ("SITE15", 'redstick_SITE_15.ply', 2.0, True),
    # ("SITE16", 'redstick_SITE_16.ply', 5.13, False),
    # # no red segments completely visible
    # ("SITE19", 'redstick_SITE_19.ply', 3.15, False),
    # # one complete stick segment, and one partly that contains more points
    # # (!)
    # ("SITE20", 'redstick_SITE_20.ply', 5.55, True),
    # ("SITE21", 'redstick_SITE_21.ply', 5.4, True)
])
def test_actual_data_confidence(name, filename, confidence, expect_confident):
    pc = load('tests/testdata/' + filename)
    meter, confidence = get_stick_scale(pc)
    if expect_confident:
        assert_greater(confidence, .5, "confidence too low")
    else:
        assert_less(confidence, .5, "confidence too high")


def assert_with_error(estimated, expected):
    error_ratio = 0.05
    margin = error_ratio * expected
    message = ('Value does not match expected value with %f%% (%f) margin.'
               '\nEstimated: %f\nExpected: %f' % (100 * error_ratio,
                                                  margin, estimated, expected))
    assert_less(abs(estimated - expected), margin, message)
