import numpy as np
import sys
import os

inputpath = sys.argv[1]

rows = 200
cols = 200

def generateH():
    h = np.random.normal(0.5, 0.1,rows*cols).reshape((rows,cols))
    return h

def generateM():
    M = np.ones((rows,cols))
    return M


def generateG():
    g = 0*np.ones((rows,cols,1))
    g[int(rows/2),:,0] = 1
    g[:,int(cols/2),0] = 1
    return g

def generateUV():
    u = np.zeros((rows,cols+1))
    v = np.zeros((rows+1,cols))

    sinAngles = np.zeros((rows,cols))
    cosAngles = np.zeros((rows,cols))
    radiuses = np.zeros((rows,cols))
    rh = int(rows/2)
    ch = int(cols/2)
    for r in range(rows):
        for c in range(cols):
            radiuses[r,c] = max(1,np.sqrt((r-rh)*(r-rh) + (c-ch)*(c-ch)))
            # print("r: {}, rh: {}, c: {}, ch: {}, arcsin({}), arccos({}), radius: {}".format(r, rh, c, ch, (r-rh)/radiuses[r,c],(c-ch)/radiuses[r,c],radiuses[r,c]))
            sinAngles[r,c] = np.arcsin((r-rh)/radiuses[r,c])
            cosAngles[r,c] = np.arccos((c-ch)/radiuses[r,c])
    u = np.ones((rows,cols+1))
    v = np.ones((rows+1,cols))

    u[:,:-1] = np.sin(sinAngles)*radiuses/ch
    v[:-1,:] = -np.cos(cosAngles)*radiuses/rh
    u[np.isnan(u)] = 0
    v[np.isnan(v)] = 0

    return (u,v)

def generatePigmentParameters():
    pigmentParameters = np.array([[0.22, 1.47, 0.57, 0.050, 0.003, 0.030, 0.02, 5.5, 0.81],
    [0.46, 1.07, 1.50, 1.280, 0.380, 0.210, 0.05, 7.0, 0.40],
    [0.10, 0.36, 3.45, 0.970, 0.650, 0.007, 0.05, 3.4, 0.81],
    [1.62, 0.61, 1.64, 0.010, 0.012, 0.003, 0.09, 1.0, 0.41],
    [1.52, 0.32, 0.25, 0.060, 0.260, 0.400, 0.01, 1.0, 0.31],
    [0.74, 1.54, 2.10, 0.090, 0.090, 0.004, 0.09, 9.3, 0.90],
    [0.14, 1.08, 1.68, 0.770, 0.015, 0.018, 0.02, 1.0, 0.63],
    [0.13, 0.81, 3.45, 0.005, 0.009, 0.007, 0.01, 1.0, 0.14],
    [0.06, 0.21, 1.78, 0.500, 0.880, 0.009, 0.06, 1.0, 0.08],
    [1.55, 0.47, 0.63, 0.010, 0.050, 0.035, 0.02, 1.0, 0.12],
    [0.86, 0.86, 0.06, 0.005, 0.005, 0.090, 0.01, 3.1, 0.91],
    [0.08, 0.11, 0.07, 1.250, 0.420, 1.430, 0.06, 1.0, 0.08]])

    return pigmentParameters
h = generateH()
np.save(os.path.join(inputpath,'heightfield.npy'),h)

M = generateM()
np.save(os.path.join(inputpath,'wetareamask.npy'),M)

(u,v) = generateUV()
np.save(os.path.join(inputpath,'xvelocity.npy'),u)
np.save(os.path.join(inputpath,'yvelocity.npy'),v)

g = generateG()
np.save(os.path.join(inputpath,'pigmentconcentration.npy'),g)

pigmentParameters = generatePigmentParameters()
np.save(os.path.join(inputpath,'pigmentparameters.npy'),pigmentParameters)
