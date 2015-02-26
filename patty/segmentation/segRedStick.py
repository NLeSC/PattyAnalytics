import colorsys
import numpy as np

def getReds(inArray):
    """Returns new array with only red parts of the input array"""
    redIndices = []
    for i in range(len(inArray)):
        x,y,z,R,G,B = inArray[i]
        H,S,V = colorsys.rgb_to_hsv(np.float32(R),np.float32(G),np.float32(B))

        if H>0.9 and S>0.5:
            redIndices.append(i)
    return inArray[redIndices]
