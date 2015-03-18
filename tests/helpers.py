import numpy as np


def make_triangle(sx, sy, dx, dy, delta):
    x1 = np.arange(0, sx, delta)
    y1 = np.zeros(x1.shape)

    y2 = np.arange(0, sy, delta)
    x2 = np.zeros(y2.shape)

    x3 = np.arange(0, sx, delta)
    y3 = sy - x3 * sy / sx

    xs = np.hstack([x1, x2, x3]) - dx
    ys = np.hstack([y1, y2, y3]) - dy

    return xs, ys


def make_tri_pyramid(sx, sy, sz, dx, dy, dz, delta):
    points = []
    for z in np.arange(0, sz, delta):
        ai = sx - z * sx / sz
        bi = sy - z * sy / sz
        xs, ys = make_triangle(ai, bi, dx, dy, delta)
        points.append((xs, ys, z * np.ones(xs.shape)))
    xs = np.hstack([x for x, y, z in points])
    ys = np.hstack([y for x, y, z in points])
    zs = np.hstack([z for x, y, z in points]) - dz
    points = np.vstack([xs, ys, zs]).T
    return points


def make_tri_pyramid_footprint(sx, sy, sz, dx, dy, dz):
    footprint = np.array([
        [0, 0, 0],
        [0, sy, 0],
        [sx, 0, 0],
        [0, 0, 0],
    ])
    footprint[:, 0] -= dx
    footprint[:, 1] -= dy
    footprint[:, 2] -= dz
    return footprint


def make_tri_pyramid_with_base(side, delta, offset):
    rng = np.random.RandomState(0)
    sx = side / 2
    sy = side
    sz = side / 4

    dx = offset[0] + side / 2
    dy = offset[1] + side / 2
    dz = offset[2]

    points = make_tri_pyramid(sx, sy, sz, dx, dy, dz, delta)
    points += (rng.rand(points.shape) - 0.5) * 0.1

    for s in np.arange(0, side * 0.05, delta):
        xs, ys = make_triangle(sx * (1 + s), sy * (1 + s), dx + s, dz + s,
                               delta)
        zs = np.zeros(xs.shape) - dz
        tmp = np.vstack([xs, ys, zs]).T
        points = np.vstack([points, tmp])

    footprint = make_tri_pyramid_footprint(sx, sy, sz, dx, dy, dz)
    return points, footprint
