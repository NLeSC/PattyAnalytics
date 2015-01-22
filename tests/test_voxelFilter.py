import unittest
from patty.conversions.conversions import loadLas, writeLas

class TestVoxelFilter(unittest.TestCase):
    def testFilter162(self):
        pc = loadLas('data/footprints/162.las')
        vgf = pc.make_voxel_grid_filter()
        vgf.set_leaf_size(1, 1, 1)

        pc2 = vgf.filter()
        pc2.offset = pc.offset
        writeLas('data/footprints/162_sparse.las', pc2)

