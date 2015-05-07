import numpy as np
import math


def make_triangle(sx, sy, dx, dy, delta):
    '''Create a right rectangle, alinged with X and Y axes, with x-side size sX
    and y-side size sY. Triangle is offset by dX and dY.'''
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
    '''Create a right rectangle triangular pyramid, alinged with X and Y axes,
    with x-side at the base size of sX and y-side size at the base of sY.
    Pyramid has high sZ. It is offset by dX, dY and dZ.'''
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
    '''Create the footprint of a pyramid created by make_tri_pyramid'''
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
    '''Create a pyramid as per make_tri_pyramid, suroundeded by a triangular
    flat base.'''
    rng = np.random.RandomState(0)
    sx = side / 2
    sy = side
    sz = side / 4

    dx = offset[0] + side / 2
    dy = offset[1] + side / 2
    dz = offset[2]

    points = make_tri_pyramid(sx, sy, sz, dx, dy, dz, delta)
    _add_noise(points, 0.1, rng)

    for s in np.arange(0, side * 0.05, delta):
        xs, ys = make_triangle(sx * (1 + s), sy * (1 + s), dx + s, dy + s,
                               delta)
        zs = np.zeros(xs.shape) - dz
        tmp = np.vstack([xs, ys, zs]).T
        points = np.vstack([points, tmp])

    footprint = make_tri_pyramid_footprint(sx, sy, sz, dx, dy, dz)
    return points, footprint


def _add_noise(points, size, rng):
    '''Add noise to an array of 2D points'''
    points += (rng.rand(points.shape[0], points.shape[1]) - 0.5) * size


def perpendicular_2d(a):
    '''Create a vector perpendicular to the original'''
    b = np.zeros(a.shape)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def rotation_around_axis(axis, theta):
    '''
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    '''
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2)
    b, c, d = -axis * math.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa+cc-bb-dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def make_half_red_stick(point_from, point_to, width=0.04, num_pts_per_line=50,
                        num_lines_per_stick=25):
    ''' Make a hollow red-white-red-white stick '''
    point_from = np.asarray(point_from, dtype=float)[:3]
    point_to = np.asarray(point_to, dtype=float)[:3]
    length = np.linalg.norm(point_to - point_from)
    width = length * 0.04
    axis = (point_to - point_from) * 1.0 / length
    origin = perpendicular_2d(axis) * width

    points = np.zeros((num_pts_per_line * num_lines_per_stick, 6))
    points[:, 3:6] = 255

    idx = 0
    unitline = np.linspace(0, 1, num_pts_per_line)
    for theta in np.linspace(0, math.pi, num_lines_per_stick):
        src = np.dot(rotation_around_axis(axis, theta), origin)

        # straight slope [0, 1]
        line = np.array((unitline, unitline, unitline)).T
        # linear function with slope
        line = point_from + src + (point_to - point_from) * line
        points[idx:idx + num_pts_per_line, :3] = line

        points[idx:idx + num_pts_per_line/2, 3:6] = [210, 25, 30]
        idx += num_pts_per_line

    return points


def make_red_stick(point_from, point_to, **kwargs):
    ''' Make a hollow red-white-red-white stick '''
    point_from = np.asarray(point_from, dtype=float)[:3]
    point_to = np.asarray(point_to, dtype=float)[:3]
    halfway = (point_to + point_from)/2
    half1 = make_half_red_stick(point_from, halfway, **kwargs)
    half2 = make_half_red_stick(halfway, point_to, **kwargs)
    return np.concatenate((half1, half2), axis=0)
