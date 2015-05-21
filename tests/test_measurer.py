import numpy as np
from patty.utils import measure_length
from numpy.testing import assert_almost_equal


def test_centered_line_on_x_axis():
    '''Test measure_length function from patty.utils'''
    line = np.array([[-5., 0, 0], [0, 0, 0], [5., 0, 0]])
    expected = 10
    length = measure_length(line)
    assert_almost_equal(expected, length)


def test_offcentered_line_on_y_axis():
    '''Test measure_length function from patty.utils'''
    line = np.array([[0, -1., 0], [0, 0, 0], [0, 9., 0]])
    expected = 10
    length = measure_length(line)
    assert_almost_equal(expected, length)


def test_rectangle_in_y_z_plane():
    '''Test measure_length function from patty.utils'''
    line = np.array([[0, -1., 0], [0, 9., 0], [0, -1., 4], [0, 9., 4]])
    expected = 10
    length = measure_length(line)
    assert_almost_equal(expected, length)


def test_single_point():
    '''Test measure_length function from patty.utils'''
    with np.errstate(all='raise'):
        line = np.array([[0, -1., 0]])
        expected = 0
        length = measure_length(line)
        assert_almost_equal(expected, length)


def test_no_points():
    '''Test measure_length function from patty.utils'''
    line = np.array([])
    expected = 0
    length = measure_length(line)
    assert_almost_equal(expected, length)
