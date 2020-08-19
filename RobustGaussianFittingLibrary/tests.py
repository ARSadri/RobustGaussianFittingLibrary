#################################################################################################
# This file is part of RobustGaussianFittingLibrary, a free library WITHOUT ANY WARRANTY        # 
# Copyright: 2019-2020 Deutsches Elektronen-Synchrotron                                         # 
#################################################################################################

import RobustGaussianFittingLibrary 
import RobustGaussianFittingLibrary.useMultiproc
import RobustGaussianFittingLibrary.misc

import numpy as np
import time
import matplotlib.pyplot as plt
import os
import scipy.stats
from cProfile import label
np.set_printoptions(suppress = True)
np.set_printoptions(precision = 2)
 
LWidth = 3
font = {
        'weight' : 'bold',
        'size'   : 8}
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
 
 
def test_for_textProgBar():
    print('test_for_textProgBar')
    pBar = RobustGaussianFittingLibrary.misc.textProgBar(180)
    for _ in range(60):
        for _ in range(10000000):
            pass
        pBar.go(3)
    del pBar 

def visOrderStat():
    print('visOrderStat')
    # std of a few closests samplse of a gaussian to its average
    # is less than the actual std:
    allN = list([2000])
    intervals = np.arange(0.01,1.01,0.01)
    for N in allN:
        Data = np.random.randn(N)
        res = np.fabs(Data - Data.mean())
        inds = np.argsort(res)
        result_STD = np.zeros(intervals.shape[0])
        result_MSSE = np.zeros(intervals.shape[0])
        pBar = RobustGaussianFittingLibrary.misc.textProgBar(intervals.shape[0])
        for idx, k in enumerate(intervals):
            result_STD[idx] = Data[inds[:int(k*N)]].std()
            result_MSSE[idx] = RobustGaussianFittingLibrary.MSSE(Data[inds[:int(k*N)]], k=2)
            pBar.go()
        del pBar
        plt.plot(intervals, result_STD)
        plt.plot(intervals, result_MSSE)
    plt.plot(intervals, np.power(intervals, 1.4))
    plt.plot(intervals, intervals)
    plt.legend(allN)
    plt.title('The estimated STD by the portion of \ninliers of a Gaussian structure')
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

def test_removeIslands():
    print('test_removeIslands')
    #an island cannot be bigger than the stack size of your OS
    inMask = np.ones((20, 21), dtype='uint8')
    
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
    outMask = 1 - RobustGaussianFittingLibrary.misc.removeIslands(1 - inMask, minSize=2)
    plt.imshow(outMask), plt.show()
    
def test_bigTensor2SmallsInds():
    print('test_bigTensor2SmallsInds')
    a = (100*np.random.randn(20,16,11)).astype('int')
    rowClmInds, segInds = RobustGaussianFittingLibrary.useMultiproc.bigTensor2SmallsInds(a.shape, 2,3)
    print(rowClmInds)

def test_RobustAlgebraicPlaneFittingPy():
    print('test_RobustAlgebraicPlaneFittingPy')
    N = 500
    numOut = 20
    inX = 100*np.random.rand(N)-50
    inY = 100*np.random.rand(N)-50
    inZ = 1*inX - 2 * inY + 50*np.random.randn(N) + 50
    inZ[((N-1)*np.random.rand(numOut)).astype('int')] = 100*np.random.rand(numOut) +500
    mP = RobustGaussianFittingLibrary.fitPlane(inX, inY, inZ, 0.8, 0.5)
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
    print('test_RobustAlgebraicLineFittingPy')
    n_in = 100
    inSigma = 3
    inX = 200*(np.random.rand(n_in)-0.5)
    inY = 0.5*inX + 10 + inSigma*np.random.randn(n_in)
    n_out = 80
    outX = 200*(np.random.rand(n_out)-0.5)
    outY = 200*(np.random.rand(n_out)-0.25)
    X = np.concatenate((inX, outX))
    Y = np.concatenate((inY, outY))
    
    label = np.ones(X.shape[0], dtype='uint8')
    _errors = Y - (0.5*X + 10)
    label[np.fabs(_errors) >= 3*inSigma] = 0
    
    print(X.shape)
    mP = RobustGaussianFittingLibrary.fitLine(X, Y, 0.5, 0.3)
    Xax = np.arange(X.min(), X.max())
    Yax_U = mP[0]*Xax + mP[1] + 3*mP[2]
    Yax_M = mP[0]*Xax + mP[1]
    Yax_L = mP[0]*Xax + mP[1] - 3*mP[2]
    
    plt.rc('font', **font)
    plt.rcParams.update(params)
    plt.scatter(X[label==0], Y[label==0], color='royalblue', label='outliers', marker='o')
    plt.scatter(X[label==1], Y[label==1], color='mediumblue', label='outliers', marker='o')
    plt.plot(Xax, Yax_U, linewidth = 3, color = 'purple')
    plt.plot(Xax, Yax_M, linewidth = 3, color = 'green')
    plt.plot(Xax, Yax_L, linewidth = 3, color = 'red')
    plt.xlabel('X')
    plt.ylabel('Y').set_rotation(0)
    plt.ylabel('Y')
    plt.show()
    print(mP)
    RobustGaussianFittingLibrary.misc.naiveHistTwoColors(_errors, np.array([0, mP[2]]))
    
def test_fitBackground():
    print('test_fitBackground')
    XSZ = 512
    YSZ = 512
    WINSIZE = 7
    inputPeaksNumber = 50
    numOutliers = 0
    print("Generating a pattern with " + str(inputPeaksNumber) + " peaks...")
    inImage, inMask, randomLocations = diffractionPatternMaker(XSZ, YSZ, WINSIZE, inputPeaksNumber, numOutliers)
    fig, axes = plt.subplots(1, 3)
    winXL = 200
    winXU = 300
    winYL = 200
    winYU = 300
    im0 = axes[0].imshow(inImage*inMask, vmin=0, vmax=1000)
    axes[0].set_xlim([winXL, winXU])
    axes[0].set_ylim([winYL, winYU])
    fig.colorbar(im0, ax=axes[0], shrink =0.5)

    mP = RobustGaussianFittingLibrary.fitBackground(inImage, inMask, 
                                                    winX = 64, 
                                                    winY = 64, 
                                                    numStrides=2) \
        + RobustGaussianFittingLibrary.fitBackground(inImage, inMask, 
                                                    winX = 32, 
                                                    winY = 32, 
                                                    numStrides=2) \
        + RobustGaussianFittingLibrary.fitBackground(inImage, inMask, 
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


def test_fitBackgroundTensor():
    print('test_fitBackgroundTensor')
    imgDimX = 100
    imgDimY = 100
    Xax = np.arange(imgDimX)
    Yax = np.arange(imgDimY)
    inX, inY = np.meshgrid(Xax, Yax)
    img1 = 0+1*np.random.randn(1, imgDimX,imgDimY)
    mP = RobustGaussianFittingLibrary.fitPlane(inX = inX.flatten(), 
                                               inY = inY.flatten(),
                                               inZ = img1.flatten())
    print(mP)

    mP = RobustGaussianFittingLibrary.fitBackground(np.squeeze(img1))
    print(mP)
    
    img2 = 3+1*np.random.randn(1, imgDimX,imgDimY)
    mP = RobustGaussianFittingLibrary.fitPlane(inX = inX.flatten(), 
                                               inY = inY.flatten(),
                                               inZ = img2.flatten())
    print(mP)
    mP = RobustGaussianFittingLibrary.fitBackground(np.squeeze(img2))
    print(mP)

    img3 = 100+10*np.random.randn(1, imgDimX,imgDimY)
    mP = RobustGaussianFittingLibrary.fitPlane(inX = inX.flatten(), 
                                               inY = inY.flatten(),
                                               inZ = img3.flatten())
    print(mP)
    mP = RobustGaussianFittingLibrary.fitBackground(np.squeeze(img3))
    print(mP)
    
    
    inTensor = np.concatenate((img1, img2, img3))
    print('input Tensor shape is: ', str(inTensor.shape))
    modelParamsMap = RobustGaussianFittingLibrary.fitBackgroundTensor(inTensor, numStrides=5)
    print(modelParamsMap)

def test_fitBackgroundTensor_multiproc():
    print('test_fitBackgroundTensor_multiproc')
    f_N, r_N, c_N = (100, 128, 512)
    inTensor = np.zeros((f_N, r_N, c_N), dtype='float32')
    for frmCnt in range(f_N):
        inTensor[frmCnt] = frmCnt+frmCnt**0.5*np.random.randn(r_N,c_N)

    print('input Tensor shape is: ', str(inTensor.shape))
    modelParamsMap = RobustGaussianFittingLibrary.useMultiproc.fitBackgroundTensor_multiproc(inTensor,
                                                              winX = 64,
                                                              winY = 64)
    for frmCnt in list([f_N-1]):
        fig, axes = plt.subplots(2, 1)
        axes[0].imshow(modelParamsMap[0,frmCnt])
        axes[1].imshow(modelParamsMap[1,frmCnt])
        plt.show()

def test_SginleGaussianVec():
    print('test_SginleGaussianVec')
    RNN0 = 50 + 5*np.random.randn(12)
    RNN1 = 200*(np.random.rand(24)-0.5)
    testData = np.concatenate((RNN0, RNN1)).flatten()
    np.random.shuffle(testData)
    print('testing RobustSingleGaussianVecPy')
    mP = RobustGaussianFittingLibrary.fitValue(testData, topKthPerc = 0.43, bottomKthPerc=0.37, MSSE_LAMBDA=1.0)
    RobustGaussianFittingLibrary.misc.naiveHist(testData, mP)
    plt.plot(testData,'.'), plt.show()
    plt.plot(testData,'.'), 
    plt.plot(np.array([0, testData.shape[0]]), np.array([mP[0]-3*mP[1], mP[0]-3*mP[1]]))
    plt.plot(np.array([0, testData.shape[0]]), np.array([mP[0], mP[0]]))
    plt.plot(np.array([0, testData.shape[0]]), np.array([mP[0]+3*mP[1], mP[0]+3*mP[1]]))
    plt.show()
    RobustGaussianFittingLibrary.misc.sGHist(testData, mP)
    
def test_fitValue2Skewed():
    print('test_fitValue2Skewed')
    RNN0 = 50 + 5*np.random.randn(50)
    RNN1 = 200*(np.random.rand(50)-0.5)
    testData = np.concatenate((RNN0, RNN1)).flatten()
    np.random.shuffle(testData)
    print('testing fitValue2Skewed')
    mP = RobustGaussianFittingLibrary.fitValue2Skewed(testData, topKthPerc = 0.43, bottomKthPerc=0.37, MSSE_LAMBDA=3.0)
    RobustGaussianFittingLibrary.misc.naiveHist(testData, mP)
    plt.plot(testData,'.'), plt.show()
    plt.plot(testData,'.'), 
    plt.plot(np.array([0, testData.shape[0]]), np.array([mP[0]-3*mP[1], mP[0]-3*mP[1]]))
    plt.plot(np.array([0, testData.shape[0]]), np.array([mP[0], mP[0]]))
    plt.plot(np.array([0, testData.shape[0]]), np.array([mP[0]+3*mP[1], mP[0]+3*mP[1]]))
    plt.show()
    RobustGaussianFittingLibrary.misc.sGHist(testData, mP)    
    
def test_fitValue2Skewed_sweep_over_N():
    print('test_fitValue2Skewed_sweep_over_N')
    numIter = 100
    maxN = 500
    minN = 2
    mode_all = np.zeros((maxN-minN, numIter))
    std_all = np.zeros((maxN-minN, numIter))
    std_base = np.zeros((maxN-minN, numIter))
    rmod_all = np.zeros((maxN-minN, numIter))
    rstd_all = np.zeros((maxN-minN, numIter))
    pBar = RobustGaussianFittingLibrary.misc.textProgBar(maxN-minN)
    x = np.zeros(maxN-minN)
    for N in range(minN,maxN):
        for iter in range(numIter):
            RNN0 = np.random.randn(N)
            RNN1 = 7+5*(np.random.rand(int(N*0.5))-0.5)
            testData = np.concatenate((RNN0, RNN1)).flatten()
            #testData = RNN0.flatten()
            np.random.shuffle(testData)
            rmode, rstd = RobustGaussianFittingLibrary.fitValue2Skewed(testData, 
                                                                topKthPerc=0.5, 
                                                                bottomKthPerc=0.25,
                                                                MSSE_LAMBDA=3.0)
            mode_all[N-minN, iter] = RNN0.mean()
            std_all[N-minN, iter] = RobustGaussianFittingLibrary.MSSE(np.squeeze(RNN0), MSSE_LAMBDA = 3.0, k=int(0.5*N))
            std_base[N-minN, iter] = RNN0.std()
            rmod_all[N-minN, iter] = rmode
            rstd_all[N-minN, iter] = rstd
        x[N-minN] = testData.shape[0]
        pBar.go()
    del pBar
    
    mode_all = mode_all.mean(1)
    std_all = std_all.mean(1)
    std_base = std_base.mean(1)
    rmod_all = rmod_all.mean(1)
    rstd_all = rstd_all.mean(1)
    
    plt.plot(x, mode_all, '.')
    plt.plot(x, rmod_all, '.')
    plt.show()
    
    plt.plot(x, std_base, '.', label='std_base')
    plt.plot(x, std_all, '.', label='std_all')
    plt.plot(x, rstd_all, '.', label='rstd_all')
    plt.grid()
    plt.legend()
    plt.show()
    
def test_flatField():    
    print('test_flatField')
    RNN0 =  0 + 1*np.random.randn(2048)
    RNN1 =  6 + 6**0.5*np.random.randn(1024)
    RNN2 =  12 + 12**0.5*np.random.randn(512)
    RNN3 =  18 + 18**0.5*np.random.randn(256)
    data = np.concatenate((RNN0, RNN1, RNN2, RNN3)).flatten()
    np.random.shuffle(data)
    
    mP_All = np.zeros((2, 4))
    testData = data.copy()

    modelCnt = 0
    mP = RobustGaussianFittingLibrary.fitValue(testData, 
                            topKthPerc = 0.49, bottomKthPerc=0.45, MSSE_LAMBDA=2.0)
    RobustGaussianFittingLibrary.misc.naiveHist(data, mP)


    for modelCnt in range(4):
        mP = RobustGaussianFittingLibrary.fitValue(testData, topKthPerc = 0.49, bottomKthPerc=0.45, MSSE_LAMBDA=1.0)
        probs = np.random.rand(testData.shape[0]) - np.exp(-(testData - mP[0])**2/(2*mP[1]**2))
        probs[testData<mP[0]] = 0
        probs[probs>mP[0]+3.0*mP[1]] = 1
        testData = testData[probs>0]
        mP_All[:, modelCnt] = mP
        
    RobustGaussianFittingLibrary.misc.naiveHist_multi_mP(data, mP_All)
    RobustGaussianFittingLibrary.misc.sGHist_multi_mP(data, mP_All, SNR=2.5)
    
def test_fitValueTensor_MultiProc():
    print('test_fitValueTensor_MultiProc')
    SIGMA = 10
    RNN1 = SIGMA*np.random.randn(500-50-3, 18, 38)
    RNN2 = 5*SIGMA + 5*SIGMA*np.random.randn(50, 18, 38)
    RNU = 30*SIGMA+SIGMA*np.random.randn(3, 18, 38)

    testData = np.concatenate((RNN1, RNN2))
    testData = np.concatenate((testData, RNU))
    
    print('testing RobustSingleGaussianTensorPy')
    nowtime = time.time()
    modelParamsMap = RobustGaussianFittingLibrary.fitValueTensor(testData)
    print(time.time() - nowtime)
    print(modelParamsMap)
    
    print('testing fitValueTensor_MultiProc')
    nowtime = time.time()
    modelParamsMap = RobustGaussianFittingLibrary.useMultiproc.fitValueTensor_MultiProc(testData,
                                                           numRowSegs = 6,
                                                           numClmSegs = 12)
    print(time.time() - nowtime)
    print(modelParamsMap)

def test_fitValueSmallSample(): 
    print('test_fitValueSmallSample')
    inliers = np.random.randn(3)
    outliers = np.array([10])
    testData = np.hstack((inliers, outliers))
    np.random.shuffle(testData)
    print('testing RobustSingleGaussianVecPy')
    mP = RobustGaussianFittingLibrary.fitValue(testData, topKthPerc = 0.5, bottomKthPerc=0.4, MSSE_LAMBDA=3.0)
    print(mP)
    #inds = np.where(np.fabs(testData)<50)
    #print('(' + str(testData[inds].mean())+ ', ' + str(testData[inds].std()) + ')')
    print('inliers mean ' + str(inliers.mean()) + ' inliers std ' + str(inliers.std()))

def test_fitLineTensor_MultiProc():
    print('test_fitLineTensor_MultiProc')
    n_F, n_R, n_C = (500, 32, 32)
    dataX = np.zeros((n_F, n_R, n_C), dtype='float32')
    dataY = np.zeros((n_F, n_R, n_C), dtype='float32')
    for imgCnt in range(n_F):
        dataX[imgCnt] = imgCnt
        dataY[imgCnt] = imgCnt + np.random.randn(n_R, n_C)
    lP = RobustGaussianFittingLibrary.useMultiproc.fitLineTensor_MultiProc(inTensorX = dataX, 
                                                                            inTensorY = dataY,
                                                                            numRowSegs = 2,
                                                                            numClmSegs = 2,
                                                                            topKthPerc = 0.5,
                                                                            bottomKthPerc = 0.4,
                                                                            MSSE_LAMBDA = 3.0,
                                                                            showProgress = True)
    plt.imshow(lP[0]), plt.show()
    plt.imshow(lP[1]), plt.show()
    plt.imshow(lP[2]), plt.show()
    
def test_fitValueSmallSample():
    print('test_fitValueSmallSample')
    inliers = np.random.randn(3)
    outliers = np.array([10])
    testData = np.hstack((inliers, outliers))
    np.random.shuffle(testData)
    print('testing RobustSingleGaussianVecPy')
    mP = RobustGaussianFittingLibrary.fitValue(testData, topKthPerc = 0.5, bottomKthPerc=0.4, MSSE_LAMBDA=3.0)
    print(mP)
    #inds = np.where(np.fabs(testData)<50)
    #print('(' + str(testData[inds].mean())+ ', ' + str(testData[inds].std()) + ')')
    print('inliers mean ' + str(inliers.mean()) + ' inliers std ' + str(inliers.std()))                                   
    
if __name__ == '__main__':
    print('PID ->' + str(os.getpid()))
    test_fitBackgroundTensor()
    test_fitBackgroundTensor_multiproc()
    test_RobustAlgebraicLineFittingPy()
    test_fitBackground()
    test_fitValue2Skewed()
    test_for_textProgBar()
    test_removeIslands()
    test_fitValueSmallSample()
    test_bigTensor2SmallsInds()
    test_RobustAlgebraicPlaneFittingPy()
    test_SginleGaussianVec()
    test_flatField()
    test_fitValue2Skewed_sweep_over_N()
    test_fitValueTensor_MultiProc()
    test_fitLineTensor_MultiProc()
    visOrderStat()
    exit()