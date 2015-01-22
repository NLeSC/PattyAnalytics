import unittest
from patty.conversions.conversions import loadLas, writeLas
from patty.registration.principalComponents import pcaRotate

class TestPrincipalComponentRotation(unittest.TestCase):
    def testPCARotation(self):
        fileIn = 'data/footprints/162.las'
        fileOut = 'data/footprints/162_pca.las'
        pc = loadLas(fileIn)
        pc2 = pcaRotate(pc)
        writeLas(fileOut, pc2)

