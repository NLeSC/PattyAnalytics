import colorsys

def getRedsRGB(inArray, minr = 0.5):
    """Returns new array with only red parts of the input array"""
    redIndices = []
    for i in range(len(inArray)):
        x,y,z,R,G,B = inArray[i]
        intensity = float(max(R + G + B, 1))
        r = R / intensity
        if r > minr:
            redIndices.append(i)
    return inArray[redIndices]


def getRedsHSV(inArray):
    """Returns new array with only red parts of the input array"""
    redIndices = []
    for i in range(len(inArray)):
        x,y,z,R,G,B = inArray[i]
        H,S,V = colorsys.rgb_to_hsv(R,G,B)

        if S>0.5 and H>0.9 and V>180:
            redIndices.append(i)
    return inArray[redIndices]


def getReds(inArray, minr = 0.5):
    return getRedsHSV(inArray)
