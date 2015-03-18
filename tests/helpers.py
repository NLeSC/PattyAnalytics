import numpy as np

def makeTriangle(sX,sY,dX,dY,delta):
    x1 = np.arange(0,sX,delta)
    y1 = np.zeros(x1.shape)

    y2 = np.arange(0,sY,delta)
    x2 = np.zeros(y2.shape)

    x3 = np.arange(0,sX,delta)
    y3 = sY - x3 * sY/sX

    xs = np.hstack([x1,x2,x3]) - dX
    ys = np.hstack([y1,y2,y3]) - dY

    return xs,ys

def makeTriPyramid(sX,sY,sZ,dX,dY,dZ,delta):
    points = []
    for z in np.arange(0,sZ,delta):
        ai = sX - z * sX/sZ
        bi = sY - z * sY/sZ
        xs,ys = makeTriangle(ai,bi,dX,dY,delta)
        points.append((xs,ys,z * np.ones(xs.shape)))
    xs = np.hstack([x for x,y,z in points])
    ys = np.hstack([y for x,y,z in points])
    zs = np.hstack([z for x,y,z in points]) - dZ
    points = np.vstack([xs,ys,zs]).T
    return points

def makeTriPyramidFootprint(sX,sY,sZ,dX,dY,dZ):
    footprint = np.array([
        [0, 0, 0],
        [0, sY, 0],
        [sX, 0, 0],
        [0, 0, 0],
    ])
    footprint[:,0] -= dX
    footprint[:,1] -= dY
    footprint[:,2] -= dZ
    return footprint
