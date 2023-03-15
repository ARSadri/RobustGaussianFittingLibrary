import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import RobustGaussianFittingLibrary as rgflib

LWidth = 3
font = {'weight' : 'bold',
        'size'   : 14}
params = {'legend.fontsize': 'x-large',
          'axes.labelsize' : 'x-large',
          'axes.titlesize' : 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}

def gkern(kernlen):
    lim = kernlen//2 + (kernlen % 2)/2
    x = np.linspace(-lim, lim, kernlen+1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/(kern2d.flatten().max())

def diffractionPatternMaker(XSZ, YSZ, WINSIZE, inputPeaksNumber, numOutliers):    
    inData = np.zeros((XSZ, YSZ), dtype='float32')
    peakMap = np.zeros((XSZ, YSZ), dtype='float32')
    
    inMask = np.ones(inData.shape, dtype = 'int8')
    # inMask[::64, ::64] = 0
    
    for ccnt in range(inData.shape[1]):
        for rcnt in range(inData.shape[0]):
            inData[rcnt, ccnt] += 100 + np.fabs(400*np.exp(-(((rcnt-512)**2+(ccnt-512)**2)**0.5 - 250)**2/(2*75**2)))
            inData[rcnt, ccnt] += 6*np.sqrt(inData[rcnt, ccnt])*np.random.randn(1)    
    
    randomLocations = np.random.rand(2,inputPeaksNumber)
    randomLocations[0,:] = XSZ/2 + np.floor(XSZ*0.8*(randomLocations[0,:] - 0.5))
    randomLocations[1,:] = YSZ/2 + np.floor(YSZ*0.8*(randomLocations[1,:] - 0.5))
    
    for cnt in np.arange(inputPeaksNumber):    
        bellShapedCurve = 600*gkern(WINSIZE)
        winXStart = (randomLocations[0, cnt] - (WINSIZE-1)/2).astype(np.int32)
        winXEnd = (randomLocations[0, cnt] + (WINSIZE+1)/2).astype(np.int32)
        winYStart = (randomLocations[1, cnt] - (WINSIZE-1)/2).astype(np.int32)
        winYEnd = (randomLocations[1, cnt] + (WINSIZE+1)/2).astype(np.int32)
        inData[ winXStart : winXEnd, winYStart : winYEnd ] += bellShapedCurve;
        peakMap[ winXStart : winXEnd, winYStart : winYEnd ] += bellShapedCurve;
        if (cnt >= inputPeaksNumber - numOutliers):
            inMask[ winXStart : winXEnd, winYStart : winYEnd ] = 0;    
    
    inData /= 100
    peakMap /= 100
    
    return(inData, inMask, randomLocations, peakMap)

def test_RobustAlgebraicPlaneFittingPy():
    print('test_RobustAlgebraicPlaneFittingPy')
    
    XSZ = 32
    YSZ = 32
    WINSIZE = 7
    inputPeaksNumber = 1
    numOutliers = 0
    print("Generating a pattern with " + str(inputPeaksNumber) + " peaks...")
    inImage, _, _, peakMap = diffractionPatternMaker(\
        XSZ, YSZ, WINSIZE, inputPeaksNumber, numOutliers)    
    
    fig = plt.figure(figsize=(8,8))
    plt.rc('font', **font)
    plt.rcParams.update(params)
    ax = fig.add_subplot(111)
    ax.imshow(inImage)
    
    fig = plt.figure(figsize=(8,8))
    plt.rc('font', **font)
    plt.rcParams.update(params)
    ax = fig.add_subplot(111)
    ax.imshow(peakMap)

    inX, inY = np.where(inImage>-1)
    inZ = inImage[inX, inY]
    
    inX_inl, inY_inl = np.where(peakMap == 0)
    inZ_inl = inImage[inX_inl, inY_inl]
    inX_outl, inY_outl = np.where(peakMap > 0)
    inZ_outl = inImage[inX_outl, inY_outl]
    
    plt.rc('font', **font)
    plt.rcParams.update(params)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(inX, inY, inZ, color = 'black', label='data')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('values')
    fig.legend()
    
    mP = rgflib.fitPlane(inX, inY, inZ)
    print(mP)
    Xax = np.arange(inX.min(), inX.max())
    Yax = np.arange(inY.min(), inY.max())
    X, Y = np.meshgrid(Xax, Yax)
    Zax_H = mP[0]*X + mP[1]*Y + mP[2] + 3*mP[3]
    Zax_U = mP[0]*X + mP[1]*Y + mP[2]
    Zax_L = mP[0]*X + mP[1]*Y + mP[2] - 3*mP[3]

    inds = np.where(inZ_outl > mP[0]*inX_outl + mP[1]*inY_outl + mP[2] + 3*mP[3])

    plt.rc('font', **font)
    plt.rcParams.update(params)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(inX_inl[::4], inY_inl[::4], inZ_inl[::4], 
               color = 'blue', label='inliers')
    ax.scatter(inX_outl[inds], inY_outl[inds], inZ_outl[inds], 
               color = 'red', label='outliers')
    c1 = ax.plot_surface(X, Y, Zax_H, color = 'm', label='upper threshold', alpha = 0.5)
    c1._facecolors2d = c1._facecolor3d
    c1._edgecolors2d = c1._edgecolor3d    
    
    c2 = ax.plot_surface(X, Y, Zax_U, color = 'red', label='model plane', alpha = 0.5)
    c2._facecolors2d = c2._facecolor3d
    c2._edgecolors2d = c2._edgecolor3d
    ax.set_xticks(np.linspace(X.min(), X.max(), 3).astype('int'))
    ax.set_yticks(np.linspace(Y.min(), Y.max(), 3).astype('int'))
    ax.set_zticks(np.linspace(inZ.min(), inZ.max(), 4).astype('int'))

    # c3 = ax.plot_surface(X, Y, Zax_L, color = 'm', label='lower threshold')
    # c3._facecolors2d = c3._facecolor3d
    # c3._edgecolors2d = c3._edgecolor3d

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('values')
    fig.legend()
    plt.show()
    
if __name__ == '__main__':
    test_RobustAlgebraicPlaneFittingPy()