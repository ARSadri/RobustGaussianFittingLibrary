import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import RobustGaussianFittingLibrary as rgflib

def gkern(kernlen):
    lim = kernlen//2 + (kernlen % 2)/2
    x = np.linspace(-lim, lim, kernlen+1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/(kern2d.flatten().max())

def diffractionPatternMaker(XSZ, YSZ, WINSIZE, inputPeaksNumber, numOutliers):    
    inData = np.zeros((XSZ, YSZ), dtype='float32')
    
    inMask = np.ones(inData.shape, dtype = 'int8')
    inMask[::64, ::64] = 0
    
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
        if (cnt >= inputPeaksNumber - numOutliers):
            inMask[ winXStart : winXEnd, winYStart : winYEnd ] = 0;    
    
    return(inData, inMask, randomLocations)

def test_fitBackground():
    print('test_fitBackground')
    XSZ = 512
    YSZ = 512
    WINSIZE = 7
    inputPeaksNumber = 50
    numOutliers = 0
    print("Generating a pattern with " + str(inputPeaksNumber) + " peaks...")
    inImage, inMask, randomLocations = diffractionPatternMaker(\
        XSZ, YSZ, WINSIZE, inputPeaksNumber, numOutliers)
    fig, axes = plt.subplots(1, 3)
    winXL = 200
    winXU = 300
    winYL = 200
    winYU = 300
    im0 = axes[0].imshow(inImage*inMask, vmin=0, vmax=1000)
    axes[0].set_xlim([winXL, winXU])
    axes[0].set_ylim([winYL, winYU])
    fig.colorbar(im0, ax=axes[0], shrink =0.5)

    mP = rgflib.fitBackground(inImage, inMask, 
                                                    winX = 64, 
                                                    winY = 64, 
                                                    numStrides=2) \
        + rgflib.fitBackground(inImage, inMask, 
                                                    winX = 32, 
                                                    winY = 32, 
                                                    numStrides=2) \
        + rgflib.fitBackground(inImage, inMask, 
                                                    winX = 16, 
                                                    winY = 16, 
                                                    numStrides=2)
    mP = mP/3
    
    im1 = axes[1].imshow(inMask*mP[0], vmin=0, vmax=1000)
    axes[1].set_xlim([winXL, winXU])
    axes[1].set_ylim([winYL, winYU])
    fig.colorbar(im1, ax=axes[1], shrink = 0.5)
    im2 = axes[2].imshow(inMask*(inImage - mP[0])/mP[1])
    axes[2].set_xlim([winXL, winXU])
    axes[2].set_ylim([winYL, winYU])
    fig.colorbar(im2, ax=axes[2], shrink = 0.5)
    plt.show()
    

if __name__ == '__main__':
    test_fitBackground()