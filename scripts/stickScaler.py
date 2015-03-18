#!/usr/bin/env python
"""Takes a point cloud containing only the red segments of scale sticks and
returns the scale estimation and a confidence indication.

Usage: stickScaler.py [-e <eps>] [-s <minsamples>] <infile>

Options:
    -e <eps>, --eps <eps>   The maximum distance between two samples for them
                        to be considered as in the same neighborhood
                        [default: 0.1].
    -s <minsamples>, --minSamples <minsamples>
                        The number of samples in a neighborhood for a point to
                        be considered a core point [default: 20].
"""

from __future__ import print_function
from docopt import docopt
from patty.conversions import load
from patty.registration.stickScale import get_stick_scale

# Takes a point cloud containing only the red segments of scale sticks and
# returns the scale estimation and a confidence indication.

if __name__ == '__main__':
    args = docopt(__doc__)
    pc = load(args['<infile>'])

    print(get_stick_scale(pc, float(args['--eps']), int(args['--minSamples'])))
