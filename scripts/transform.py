#!/usr/bin/env python2.7
"""Apply transformation to pointcloud.

The transformations (if given) are applied in the following order:
1. rotation
2. scaling
3. offset

Usage:
  transform.py [-o <origin>] [-r <rot>] [-t <translate>] [-s <scaling>] <source> <target>

Positional arguments:
  source            Source pointcloud file
  target            Target pointcloud file

Options:
  -r <rot>          CSV file of a 4x4 transformation matrix, assumed to be
                    normalized.
  -o <rot_origin>   CSV file with a 3d point of the origin for the rotation and scaling
  -s <scaling>      CSV file with a single scaling multiplication factor.
  -t <translate>    CSV file with a 3d vector to translate the pointcloud with.
"""

from __future__ import print_function
from docopt import docopt

import numpy as np
from patty.utils import load, save


def csv_read(path):
    return np.genfromtxt(path, dtype=float, delimiter=',')


if __name__ == '__main__':
    args = docopt(__doc__)

    pc = load(args['<source>'])

    try:
        offset = csv_read(args['-o'])
    except:
        offset = None

    try:
        matrix = csv_read(args['-r'])
        pc.rotate(matrix, origin=offset)
    except Exception as e:
        print('Problem with rotate: ', e)

    try:
        factor = csv_read(args['-s'])
        pc.scale(factor, origin=offset)
    except Exception as e:
        print('Problem with scale: ', e)

    try:
        vector = csv_read(args['-t'])
        pc.translate(vector)
    except Exception as e:
        print('Problem with translate: ', e)

    save(pc, args['<target>'])
