import pcl
import unittest
import numpy as np
from patty.registration.stickScale import getStickScale
from numpy.testing import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import nottest
from nose_parameterized import parameterized
import math

class TestStickScale2(unittest.TestCase):
    """ The ground truths for these tests was defined by measuring stick
    segments, either white or red, by hand, in Meshlab. 
    Nb. Failing instances commented out. Uncomment them for testing out 
    new methods."""
    @parameterized.expand([        
        ("SITE9", 'redstick_SITE_9.ply', 11.45),
        ("SITE11", 'redstick_SITE_11.ply', 2.925),
        ("SITE12", 'redstick_SITE_12.ply', 10.85),
        #("SITE13", 'redstick_SITE_13.ply', 4.35),  # only one segment visible
        #("SITE14", 'redstick_SITE_14.ply', 1.07),  # hardly anything of a segment is visible
        ("SITE15", 'redstick_SITE_15.ply', 2.0),
        #("SITE16", 'redstick_SITE_16.ply', 5.13),  # no red segments completely visible
        #("SITE19", 'redstick_SITE_19.ply', 3.15),  # one complete stick segment, and one partly that contains more points (!)
        ("SITE20", 'redstick_SITE_20.ply', 5.55),
        ("SITE21", 'redstick_SITE_21.ply', 5.4)
    ])    
    def test_actualData_correctLength(self, name, fileName, expectedMeter):
        #Arrange
        pc = pcl.load('tests/testdata/' + fileName)
        
        #Act
        meter, confidence = getStickScale(pc)
                
        #Assert        
        assertWithError(meter, expectedMeter)
    
    
    """ The ground truths for these tests was defined by measuring stick
    segments, either white or red, by hand, in Meshlab. """
    @parameterized.expand([        
        ("SITE9", 'redstick_SITE_9.ply', 11.45, True),
        ("SITE11", 'redstick_SITE_11.ply', 2.925, True),
        ("SITE12", 'redstick_SITE_12.ply', 10.85, True),
        ("SITE13", 'redstick_SITE_13.ply', 4.35, False),  # only one segment visible
        ("SITE14", 'redstick_SITE_14.ply', 1.07, False),  # hardly anything of a segment is visible
        ("SITE15", 'redstick_SITE_15.ply', 2.0, True),
        ("SITE16", 'redstick_SITE_16.ply', 5.13, False),  # no red segments completely visible
        ("SITE19", 'redstick_SITE_19.ply', 3.15, False),  # one complete stick segment, and one partly that contains more points (!)
        ("SITE20", 'redstick_SITE_20.ply', 5.55, True),
        ("SITE21", 'redstick_SITE_21.ply', 5.4, True)
    ])    
    def test_actualData_correctConfidence(self, name, fileName, expectedMeter, expectConfident):
        #Arrange
        pc = pcl.load('tests/testdata/' + fileName)
        
        #Act
        meter, confidence = getStickScale(pc)
                
        #Assert
        if expectConfident:
            assertConfidenceAboveThreshold(confidence)            
        else:
            assertConfidenceBelowThreshold(confidence)
    
def assertConfidenceAboveThreshold(confidence):
    message = 'expected high confidence but confidence was low: ' + `confidence`
    assert confidence > 0.5, message

def assertConfidenceBelowThreshold(confidence):
    message = 'expected low confidence but confidence was high: ' + `confidence`
    assert confidence < 0.5, message    
        
def assertWithError(estimated, expected):
    errorRatio = 0.05
    errorMargin = errorRatio * expected
    message = 'Value does not match expected value with ' + `100 * errorRatio` + '% (' + `errorMargin` + ') margin.\nEstimated: ' + `estimated` + '\nExpected:  ' + `expected`
    assert abs(estimated - expected) < errorMargin, message
        
