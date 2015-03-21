#!/usr/bin/env python2.7
"""Apply transformation to pointcloud.

The transformations (if given) are applied in the following order:
1. rotation
2. scaling
3. offset

Usage:
  transform.py [-r <rot>][-o <rot_origin>] [-t <translate>] [-s <scaling>] <source> <target>

Positional arguments:
  source            Source pointcloud file
  target            Target pointcloud file

Options:
  -r <rot>          CSV file of a 4x4 transformation matrix, assumed to be
                    normalized.
  -o <rot_origin>   CSV file with a 3d point of the origin for the rotation.
  -s <scaling>      CSV file with a single scaling multiplication factor.
  -t <translate>    CSV file with a 3d vector to translate the pointcloud with.
"""

from __future__ import print_function
from docopt import docopt

import numpy as np
import time
from patty import load, save, set_registeration


def log(*args, **kwargs):
    print(time.strftime("[%H:%M:%S]"), *args, **kwargs)


def csv_read(path):
    return np.genfromtxt(path, dtype=float, delimiter=',')


def rotate(pointcloud, matrix, offset=None):
    if offset is not None and np.any(offset != pointcloud.offset):
        add_offset = offset - pointcloud.offset
        pc_array = np.asarray(pointcloud)
        pc_array += add_offset
        set_registeration(pointcloud, offset=offset)

    pointcloud.transform(matrix)


def scale(pointcloud, factor):
    pc_array = np.asarray(pointcloud)
    pc_array *= factor
    set_registeration(pointcloud, precision=pointcloud.precision * factor)


def translate(pointcloud, offset):
    set_registeration(pointcloud, offset=pointcloud.offset + offset)

if __name__ == '__main__':
    args = docopt(__doc__)

    pc = load(args['<source>'])
    try:
        matrix = csv_read(args['-r'])
        try:
            offset = csv_read(args['-o'])
            rotate(pc, matrix, offset)
            log("rotated pointcloud with offset")
        except Exception as ex:
            rotate(pc, matrix)
            log("rotated pointcloud without offset", ex)
    except:
        log("not rotating pointcloud")

    try:
        factor = csv_read(args['-s'])
        scale(pc, factor)
        scale(pc, factor)
        log("scaled pointcloud by ", factor)
    except Exception as ex:
        log("not scaling pointcloud", ex)

    try:
        translate(pc, csv_read(args['-t']))
        log("translated pointcloud")
    except:
        log("not translating pointcloud")

    save(pc, args['<target>'])
