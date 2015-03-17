import numpy as np
from patty.segmentation.pointCloudMeasurer import measureLength
from numpy.testing import assert_almost_equal


def test_centeredLineOnXaxis_correctResult():
    # Arrange
    line = np.array([[-5., 0, 0], [0, 0, 0], [5., 0, 0]])
    expected = 10

    # Act
    length = measureLength(line)

    # Assert
    assert_almost_equal(expected, length)


def test_offcenteredLineOnYaxis_correctResult():
    # Arrange
    line = np.array([[0, -1., 0], [0, 0, 0], [0, 9., 0]])
    expected = 10

    # Act
    length = measureLength(line)

    # Assert
    assert_almost_equal(expected, length)


def test_rectangleInYZplane_correctLength():
    # Arrange
    line = np.array([[0, -1., 0], [0, 9., 0], [0, -1., 4], [0, 9., 4]])
    expected = 10

    # Act
    length = measureLength(line)

    # Assert
    assert_almost_equal(expected, length)


def test_singlePoint_return0():
    # Arrange
    line = np.array([[0, -1., 0]])
    expected = 0

    # Act
    length = measureLength(line)

    # Assert
    assert_almost_equal(expected, length)


def test_noPoints_return0():
    # Arrange
    line = np.array([])
    expected = 0

    # Act
    length = measureLength(line)

    # Assert
    assert_almost_equal(expected, length)
