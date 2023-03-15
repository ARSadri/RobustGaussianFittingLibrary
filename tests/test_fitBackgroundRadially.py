import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import RobustGaussianFittingLibrary as RGF

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
        winXStart = (randomLocations[0, cnt] - (WINSIZE-1)/2).astype('int64')
        winXEnd = (randomLocations[0, cnt] + (WINSIZE+1)/2).astype('int64')
        winYStart = (randomLocations[1, cnt] - (WINSIZE-1)/2).astype('int64')
        winYEnd = (randomLocations[1, cnt] + (WINSIZE+1)/2).astype('int64')
        inData[ winXStart : winXEnd, winYStart : winYEnd ] += bellShapedCurve;
        if (cnt >= inputPeaksNumber - numOutliers):
            inMask[ winXStart : winXEnd, winYStart : winYEnd ] = 0;    
    
    return(inData, inMask, randomLocations)

def test_fitBackgroundRadially():
    print('test_fitBackgroundRadially')
    XSZ = 1024
    YSZ = 1024
    WINSIZE = 7
    inputPeaksNumber = 5
    numOutliers = 0
    print("Generating a pattern with " + str(inputPeaksNumber) + " peaks...")
    inImage, inMask, randomLocations = diffractionPatternMaker(
        XSZ, YSZ, WINSIZE, 
        inputPeaksNumber, numOutliers)
    
    im0_img = inImage*inMask
    
    centX = im0_img.shape[1]/2
    centY = im0_img.shape[0]/2
    
    xx, yy = np.meshgrid(np.arange(im0_img.shape[1]), np.arange(im0_img.shape[0]))
    
    indsi, indsj = np.where( (290**2 <= (xx - centX)**2 + (yy - centY)**2) \
                           & ((xx - centX)**2 + (yy - centY)**2 <= 310**2) )
    
    vec = im0_img[indsi, indsj]
    
    print(vec.shape)
    
    plt.hist(vec), plt.show()
    
    time_time = time.time()
    print('Calculating mp', flush = True)
    mP, vecMP = RGF.fitBackgroundRadially(
        inImage, 
        inMask = inMask,
        shellWidth = 5,
        stride = 1,
        includeCenter=1,
        return_vecMP = True)
    print('time: ' + str(time.time() -time_time) + ' s', flush = True)
    plt.plot(vecMP[0] + vecMP[1] , label='avg + std')
    plt.plot(vecMP[0]            , label='avg')
    plt.plot(vecMP[0] - vecMP[1] , label='avg - std')
    plt.legend()
    plt.show()
    
    plt.imshow(mP[0]), plt.show()
    
    im1_img = inMask*mP[0]
    im2_img = inMask*(inImage - mP[0])/(mP[1]+0.001)
    im2_img[(np.fabs(mP[1])<1)]=0
    
    winXL = 0
    winXU = 1024
    winYL = 0
    winYU = 1024
    
    fig, axes = plt.subplots(1, 3)
    im0 = axes[0].imshow(im0_img, vmin=0, vmax=1000)
    axes[0].set_xlim([winXL, winXU])
    axes[0].set_ylim([winYL, winYU])
    fig.colorbar(im0, ax=axes[0], shrink = 0.5)
    im1 = axes[1].imshow(im1_img, vmin=0, vmax=1000)
    axes[1].set_xlim([winXL, winXU])
    axes[1].set_ylim([winYL, winYU])
    fig.colorbar(im1, ax=axes[1], shrink = 0.5)
    im2 = axes[2].imshow(im2_img, vmin = -6, vmax = 6)
    axes[2].set_xlim([winXL, winXU])
    axes[2].set_ylim([winYL, winYU])
    fig.colorbar(im2, ax=axes[2], shrink = 0.5)
    plt.show()

if __name__ == '__main__':
    test_fitBackgroundRadially()