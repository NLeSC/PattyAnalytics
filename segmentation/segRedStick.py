import tempfile
import pcl
import numpy as np
import argparse

def getInputPoinCloudAsArray(sourcePath):
    pc = pcl.PointCloudXYZRGB()
    pc.from_file(sourcePath)
    return pc.to_array()
    
def getReds(inArray):
    """Returns new array with only red parts of the input array"""
    redIndices = []
    for i in range(len(inArray)):
        x,y,z,R,G,B = ar[i]
        intensity = max(R + G + B, 1)
        r = R / intensity
        if r > args.minr:
            redIndices.append(i)
    return inArray[redIndices]
    
def saveArrayAsPointCloud(array):
    pc = pcl.PointCloudXYZRGB()
    pc.from_array(array)
    pcl.save(pc, args.outFile)    

if __name__=='__main__':
    """Segment points by colour from a pcd file and saves all reddish points into a pcd of ply file."""
    parser = argparse.ArgumentParser(description='Segment points by colour from a pcd file and saves all reddish points into a pcd of ply file.')
    parser.add_argument('-i','--inFile', required=True, type=str, help='Input PCD file')
    parser.add_argument('-o','--outFile',required=True, type=str, help='Output PLY/PCD file')
    parser.add_argument('-r','--minr',required=False, default=0.6, type=float, help='Minimal r (normalized RGB) value. range [0, 1]' )
    args = parser.parse_args()

    ar = getInputPoinCloudAsArray(args.inFile)    
    redsAr = getReds(ar)    
    saveArrayAsPointCloud(redsAr)

    
