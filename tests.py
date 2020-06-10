import RGFLib
import RGFLib.useMultiproc
import RGFLib.misc

import numpy as np
import time
import matplotlib.pyplot as plt
import os
import scipy.stats
np.set_printoptions(suppress = True)
np.set_printoptions(precision = 2)
 
def visOrderStat():
    # std of a few closests samplse of a gaussian to its average
    # is less than the actual std:
    allN = list([6000])
    intervals = np.arange(0.01,1.01,0.01)
    for N in allN:
        Data = np.random.randn(N)
        res = np.fabs(Data - Data.mean())
        inds = np.argsort(res)
        result = np.zeros(intervals.shape[0])
        for idx, k in enumerate(intervals):
            result[idx] = Data[inds[:int(k*N)]].std()
        plt.plot(intervals, result)
    plt.plot(intervals, np.power(intervals, 0.7)*np.exp(intervals)/2.78)
    #x = np.erfinv(m)/(2*sqrt(2))
    plt.plot(intervals, intervals)
    plt.legend(allN)
    plt.show()

def gkern(kernlen):
    lim = kernlen//2 + (kernlen % 2)/2
    x = np.linspace(-lim, lim, kernlen+1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/(kern2d.flatten().max())

def diffractionPatternMaker(XSZ, YSZ, WINSIZE, inputPeaksNumber, numOutliers):    
    inData = np.zeros((XSZ, YSZ), dtype='float32')
    
    inMask = np.ones(inData.shape, dtype = 'uint8')
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
        winXStart = (randomLocations[0, cnt] - (WINSIZE-1)/2).astype(np.int)
        winXEnd = (randomLocations[0, cnt] + (WINSIZE+1)/2).astype(np.int)
        winYStart = (randomLocations[1, cnt] - (WINSIZE-1)/2).astype(np.int)
        winYEnd = (randomLocations[1, cnt] + (WINSIZE+1)/2).astype(np.int)
        inData[ winXStart : winXEnd, winYStart : winYEnd ] += bellShapedCurve;
        if (cnt >= inputPeaksNumber - numOutliers):
            inMask[ winXStart : winXEnd, winYStart : winYEnd ] = 0;    
    
    return(inData, inMask, randomLocations)

def test_islandRemovalPy():
    inMask = np.ones((20,20), dtype='uint8')

    inMask[0,1] = 0
    inMask[1,1] = 0
    inMask[1,0] = 0

    inMask[3,3] = 0
    inMask[4,2] = 0
    inMask[4,4] = 0
    inMask[5,3] = 0

    inMask[0,4] = 0
    inMask[1,4] = 0
    inMask[1,5] = 0
    inMask[1,6] = 0
    inMask[0,6] = 0

    inMask[14,0] =0
    inMask[14,1] =0
    inMask[15,1] =0
    inMask[16,1] =0
    inMask[16,0] =0

    inMask[6,6] = 0
    inMask[6,7] = 0
    inMask[6,8] = 0
    inMask[6,9] = 0
    inMask[7,5] = 0
    inMask[7,8] = 0
    inMask[8,6] = 0
    inMask[8,7] = 0
    inMask[8,8] = 0
    inMask[8,9] = 0

    inMask[16,16] = 0
    inMask[16,17] = 0
    inMask[16,18] = 0
    inMask[16,19] = 0
    inMask[17,15] = 0
    inMask[17,18] = 0
    inMask[18,16] = 0
    inMask[18,17] = 0
    inMask[18,18] = 0
    inMask[18,19] = 0
    
    plt.imshow(inMask), plt.show()
    outMask = RGFLib.misc.islandRemovalPy(1 - inMask, minSize=2)
    plt.imshow(outMask), plt.show()
    
def test_bigTensor2SmallsInds():
    a = (100*np.random.randn(20,16,11)).astype('int')
    rowClmInds, segInds = RGFLib.useMultiproc.bigTensor2SmallsInds(a.shape, 2,3)
    print(rowClmInds)

def test_RobustAlgebraicPlaneFittingPy():
    N = 500
    numOut = 20
    inX = 100*np.random.rand(N)-50
    inY = 100*np.random.rand(N)-50
    inZ = 1*inX - 2 * inY + 50*np.random.randn(N) + 50
    inZ[((N-1)*np.random.rand(numOut)).astype('int')] = 100*np.random.rand(numOut) +500
    mP = RGFLib.fitPlane(inX, inY, inZ, 0.8, 0.5)
    print(mP)
    Xax = np.arange(inX.min(), inX.max())
    Yax = np.arange(inY.min(), inY.max())
    X, Y = np.meshgrid(Xax, Yax)
    Zax_H = mP[0]*X + mP[1]*Y + mP[2] + 3*mP[3]
    Zax_U = mP[0]*X + mP[1]*Y + mP[2]
    Zax_L = mP[0]*X + mP[1]*Y + mP[2] - 3*mP[3]

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(inX, inY, inZ)
    ax.plot_surface(X, Y, Zax_H)
    ax.plot_surface(X, Y, Zax_U)
    ax.plot_surface(X, Y, Zax_L)
    plt.show()

def test_RobustAlgebraicLineFittingPy():
    N = 50
    inX = 100*np.random.rand(N)-50
    inY = 2*inX + 10 + 10*np.random.randn(N)
    inY[int(N/2)] = 100
    mP = RGFLib.fitLine(inX, inY, 0.5, 0.3)
    Xax = np.arange(inX.min(), inX.max())
    Yax_U = mP[0]*Xax + mP[1] + 3*mP[2]
    Yax_M = mP[0]*Xax + mP[1]
    Yax_L = mP[0]*Xax + mP[1] - 3*mP[2]
    plt.scatter(inX, inY)
    plt.plot(Xax, Yax_M)
    plt.plot(Xax, Yax_U)
    plt.plot(Xax, Yax_L)
    plt.show()
    print(mP)
    
def test_fitBackground():
    XSZ = 512
    YSZ = 512
    WINSIZE = 7
    inputPeaksNumber = 25
    numOutliers = 5
    print("Generating a pattern with " + str(inputPeaksNumber) + " peaks...")
    inImage, inMask, randomLocations = diffractionPatternMaker(XSZ, YSZ, WINSIZE, inputPeaksNumber, numOutliers)
    
    plt.imshow(inImage*inMask, vmin=0, vmax=1000)
    plt.show()

    mP = RGFLib.fitBackground(inImage, inMask, winX = 64, winY = 64, stretch2CornersOpt=4, numModelParams = 4) \
        + RGFLib.fitBackground(inImage, inMask, winX = 32, winY = 32, stretch2CornersOpt=2, numModelParams = 4) \
        + RGFLib.fitBackground(inImage, inMask, winX = 16, winY = 16, stretch2CornersOpt=1, numModelParams = 4)
    mP = mP/3
    
    plt.imshow(mP[0], vmin=0, vmax=1000)
    plt.show()
    plt.imshow(inMask*(inImage - mP[0])/mP[1])
    plt.show()

def test_fitBackgroundTensor():
    imgDimX = 100
    imgDimY = 100
    Xax = np.arange(imgDimX)
    Yax = np.arange(imgDimY)
    inX, inY = np.meshgrid(Xax, Yax)
    img1 = np.random.randn(1, imgDimX,imgDimY)
    mP = RGFLib.fitPlane(inX = inX.flatten(), 
                                       inY = inY.flatten(),
                                       inZ = img1.flatten())
    print(mP)
    img2 = 10+np.random.randn(1, imgDimX,imgDimY)
    mP = RGFLib.fitPlane(inX = inX.flatten(), 
                                       inY = inY.flatten(),
                                       inZ = img2.flatten())
    print(mP)
    img3 = 100+10*np.random.randn(1, imgDimX,imgDimY)
    mP = RGFLib.fitPlane(inX = inX.flatten(), 
                                       inY = inY.flatten(),
                                       inZ = img3.flatten())
    print(mP)
    inTensor = np.concatenate((img1, img2, img3))
    print('input Tensor shape is: ', str(inTensor.shape))
    modelParamsMap = RGFLib.fitBackgroundTensor(inTensor)
    print(modelParamsMap)

def test_fitBackgroundTensor_multiproc():
    f_N, r_N, c_N = (100, 128, 512)
    inTensor = np.zeros((f_N, r_N, c_N), dtype='float32')
    for frmCnt in range(f_N):
        inTensor[frmCnt] = frmCnt+frmCnt**0.5*np.random.randn(r_N,c_N)

    print('input Tensor shape is: ', str(inTensor.shape))
    modelParamsMap = RGFLib.useMultiproc.fitBackgroundTensor_multiproc(inTensor,
                                                              winX = 64,
                                                              winY = 64)
    for frmCnt in list([f_N-1]):
        fig, axes = plt.subplots(2, 1)
        axes[0].imshow(modelParamsMap[0,frmCnt])
        axes[1].imshow(modelParamsMap[1,frmCnt])
        plt.show()

def test_SginleGaussianVec():    
    RNN0 = 50 + 5*np.random.randn(12)
    RNN1 = 200*(np.random.rand(24)-0.5)
    testData = np.concatenate((RNN0, RNN1)).flatten()
    np.random.shuffle(testData)
    print('testing RobustSingleGaussianVecPy')
    mP = RGFLib.fitValue(testData, topKthPerc = 0.43, bottomKthPerc=0.37, MSSE_LAMBDA=1.0)
    RGFLib.misc.naiveHist(testData, mP)
    plt.plot(testData,'.'), plt.show()
    plt.plot(testData,'.'), 
    plt.plot(np.array([0, testData.shape[0]]), np.array([mP[0]-3*mP[1], mP[0]-3*mP[1]]))
    plt.plot(np.array([0, testData.shape[0]]), np.array([mP[0], mP[0]]))
    plt.plot(np.array([0, testData.shape[0]]), np.array([mP[0]+3*mP[1], mP[0]+3*mP[1]]))
    plt.show()
    RGFLib.misc.sGHist(testData, mP)
    
def test_flatField():    
    RNN0 =  0 + 1*np.random.randn(2048)
    RNN1 =  6 + 6**0.5*np.random.randn(1024)
    RNN2 =  12 + 12**0.5*np.random.randn(512)
    RNN3 =  18 + 18**0.5*np.random.randn(256)
    data = np.concatenate((RNN0, RNN1, RNN2, RNN3)).flatten()
    np.random.shuffle(data)
    
    mP_All = np.zeros((2, 4))
    testData = data.copy()

    modelCnt = 0
    mP = RGFLib.fitValue(testData, 
                            topKthPerc = 0.49, bottomKthPerc=0.45, MSSE_LAMBDA=2.0)
    RGFLib.misc.naiveHist(data, mP)


    for modelCnt in range(4):
        mP = RGFLib.fitValue(testData, topKthPerc = 0.49, bottomKthPerc=0.45, MSSE_LAMBDA=1.0)
        probs = np.random.rand(testData.shape[0]) - np.exp(-(testData - mP[0])**2/(2*mP[1]**2))
        probs[testData<mP[0]] = 0
        probs[probs>mP[0]+3.0*mP[1]] = 1
        testData = testData[probs>0]
        mP_All[:, modelCnt] = mP
        
    RGFLib.misc.naiveHist_multi_mP(data, mP_All)
    RGFLib.misc.sGHist_multi_mP(data, mP_All, SNR=2.5)
    
def test_fitValueTensor_MultiProc():    
    SIGMA = 10
    RNN1 = SIGMA*np.random.randn(500-50-3, 18, 38)
    RNN2 = 5*SIGMA + 5*SIGMA*np.random.randn(50, 18, 38)
    RNU = 30*SIGMA+SIGMA*np.random.randn(3, 18, 38)

    testData = np.concatenate((RNN1, RNN2))
    testData = np.concatenate((testData, RNU))
    
    print('testing RobustSingleGaussianTensorPy')
    nowtime = time.time()
    modelParamsMap = RGFLib.fitValueTensor(testData)
    print(time.time() - nowtime)
    print(modelParamsMap)
    
    print('testing fitValueTensor_MultiProc')
    nowtime = time.time()
    modelParamsMap = RGFLib.useMultiproc.fitValueTensor_MultiProc(testData,
                                                           numRowSegs = 6,
                                                           numClmSegs = 12)
    print(time.time() - nowtime)
    print(modelParamsMap)
    
if __name__ == '__main__':
    print('PID ->' + str(os.getpid()))
    test_islandRemovalPy()
    test_bigTensor2SmallsInds()
    test_RobustAlgebraicPlaneFittingPy()
    test_RobustAlgebraicLineFittingPy()
    test_fitBackground()
    test_fitBackgroundTensor()
    test_fitBackgroundTensor_multiproc()
    test_SginleGaussianVec()
    test_flatField()
    test_fitValueTensor_MultiProc()