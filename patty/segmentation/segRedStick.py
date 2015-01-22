import pcl
import argparse
    
def getReds(inArray, minr = 0.5):
    """Returns new array with only red parts of the input array"""
    redIndices = []
    for i in range(len(inArray)):
        x,y,z,R,G,B = inArray[i]
        intensity = float(max(R + G + B, 1))
        r = R / intensity
        if r > minr:
            redIndices.append(i)
    return inArray[redIndices]
    


    
