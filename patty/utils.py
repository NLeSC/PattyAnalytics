import numpy as np


def downsample(pc, fraction, random_seed=None):
    """Randomly downsample pointcloud to a fraction of its size.

    Returns a pointcloud of size fraction * len(pc), rounded to the nearest
    integer.

    Use random_seed=k for some integer k to get reproducible results.
    """

    rng = np.random.RandomState(random_seed)

    if not 0 < fraction <= 1:
        raise ValueError("Expected fraction in (0,1], got %r" % fraction)

    k = max(int(round(fraction * len(pc))), 1)
    sample = rng.choice(len(pc), k, replace=False)
    return pc.extract(sample)
