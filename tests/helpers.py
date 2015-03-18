import numpy as np


def __makeTriangle__(sX, sY, dX, dY, delta):
    '''Create a right rectangle, alinged with X and Y axes, with x-side size sX
    and y-side size sY. Triangle is offset by dX and dY.'''
    x1 = np.arange(0, sX, delta)
    y1 = np.zeros(x1.shape)

    y2 = np.arange(0, sY, delta)
    x2 = np.zeros(y2.shape)

    x3 = np.arange(0, sX, delta)
    y3 = sY - x3 * sY / sX

    xs = np.hstack([x1, x2, x3]) - dX
    ys = np.hstack([y1, y2, y3]) - dY

    return xs, ys


def __makeTriPyramid__(sX, sY, sZ, dX, dY, dZ, delta):
    '''Create a right rectangle triangular pyramid, alinged with X and Y axes,
    with x-side at the base size of sX and y-side size at the base of sY.
    Pyramid has high sZ. It is offset by dX, dY and dZ.'''
    points = []
    for z in np.arange(0, sZ, delta):
        ai = sX - z * sX / sZ
        bi = sY - z * sY / sZ
        xs, ys = __makeTriangle__(ai, bi, dX, dY, delta)
        points.append((xs, ys, z * np.ones(xs.shape)))
    xs = np.hstack([x for x, y, z in points])
    ys = np.hstack([y for x, y, z in points])
    zs = np.hstack([z for x, y, z in points]) - dZ
    points = np.vstack([xs, ys, zs]).T
    return points


def __makeTriPyramidFootprint__(sX, sY, sZ, dX, dY, dZ):
    '''Create the footprint of a pyramid created by __makeTriPyramid__'''
    footprint = np.array([
        [0, 0, 0],
        [0, sY, 0],
        [sX, 0, 0],
        [0, 0, 0],
    ])
    footprint[:, 0] -= dX
    footprint[:, 1] -= dY
    footprint[:, 2] -= dZ
    return footprint


def __makeTriPyramidWithBase__(side, delta, offset):
    sX = side / 2
    sY = side
    sZ = side / 4

    dX = offset[0] + side / 2
    dY = offset[1] + side / 2
    dZ = offset[2]

    points = __makeTriPyramid__(sX, sY, sZ, dX, dY, dZ, delta)
    points += np.random.rand(points.shape[0], points.shape[1]) * 0.1

    dS = np.arange(0, side * 0.05, delta)
    for s in dS:
        xs, ys = __makeTriangle__(
            sX * (1 + s), sY * (1 + s), dX + s, dY + s, delta)
        zs = np.zeros(xs.shape) - dZ
        tmp = np.vstack([xs, ys, zs]).T
        points = np.vstack([points, tmp])

    footprint = __makeTriPyramidFootprint__(sX, sY, sZ, dX, dY, dZ)
    return points, footprint
