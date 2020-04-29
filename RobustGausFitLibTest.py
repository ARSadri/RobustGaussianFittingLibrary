import RobustGausFitLibPy as RGFLib
import numpy as np
import time
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

def naiveHist(vec, mP):
    plt.figure(figsize=[10,8])
    hist,bin_edges = np.histogram(vec, 100)
    plt.bar(bin_edges[:-1], hist, width = 3, color='#0504aa',alpha=0.7)
    x = np.linspace(vec.min(), vec.max(), 1000)
    y = hist.max() * np.exp(-(x-mP[0])*(x-mP[0])/(2*mP[1]*mP[1])) 
    plt.plot(x,y, 'r')
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value',fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.title('Normal Distribution Histogram',fontsize=15)
    plt.show()
    
def test_TLS_AlgebraicPlaneFittingPY():
    N = 100
    inX = np.random.rand(N) - 0.5
    inY = np.random.rand(N) - 0.5
    inZ = inX + inY + 0.01*np.random.randn(N)
    mP = RGFLib.TLS_AlgebraicPlaneFittingPY(inX, inY, inZ)
    print(mP)
    
def test_bigTensor2SmallsInds():
    a = (100*np.random.randn(20,16,11)).astype('int')
    rowClmInds, segInds = RGFLib.bigTensor2SmallsInds(a.shape, 2,3)
    print(rowClmInds)

def test_RobustAlgebraicPlaneFittingPy():
    N = 500
    numOut = 20
    inX = 100*np.random.rand(N)-50
    inY = 100*np.random.rand(N)-50
    inZ = 1*inX - 2 * inY + 50*np.random.randn(N) + 50
    inZ[((N-1)*np.random.rand(numOut)).astype('int')] = 100*np.random.rand(numOut) +500
    mP = RGFLib.RobustAlgebraicPlaneFittingPy(inX, inY, inZ, 0.8, 0.5)
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
    mP = RGFLib.RobustAlgebraicLineFittingPy(inX, inY, 0.5, 0.3)
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
    
def test_RMGImagePy():
    
    inImage = 100*np.random.randn(1024, 1024).astype('float32')
    
    inMask = np.ones(inImage.shape, dtype = 'uint8')
    inMask[-1, :] = 0
    inMask[ 0, :] = 0
    inMask[ :, 0] = 0
    inMask[:, -1] = 0
    
    for ccnt in range(inImage.shape[1]):
        for rcnt in range(inImage.shape[0]):
            inImage[rcnt, ccnt] += (rcnt**2+ccnt**2)**0.5
    
    mP = RGFLib.RMGImagePy(inImage, inMask, winX = 102, winY = 102) \
        + RGFLib.RMGImagePy(inImage, inMask, winX = 60, winY = 60) \
        + RGFLib.RMGImagePy(inImage, inMask, winX = 20, winY = 20)
    mP = mP/3

    plt.imshow(inImage*inMask)
    plt.show()
    plt.imshow(mP[0,...])
    plt.show()
    plt.imshow(mP[1,...])
    plt.show()


def test_RSGImagesInTensorPy():
    imgDimX = 100
    imgDimY = 100
    Xax = np.arange(imgDimX)
    Yax = np.arange(imgDimY)
    inX, inY = np.meshgrid(Xax, Yax)
    img1 = np.random.randn(1, imgDimX,imgDimY)
    mP = RGFLib.RobustAlgebraicPlaneFittingPy(inX = inX.flatten(), 
                                       inY = inY.flatten(),
                                       inZ = img1.flatten())
    print(mP)
    img2 = 10+np.random.randn(1, imgDimX,imgDimY)
    mP = RGFLib.RobustAlgebraicPlaneFittingPy(inX = inX.flatten(), 
                                       inY = inY.flatten(),
                                       inZ = img2.flatten())
    print(mP)
    img3 = 100+10*np.random.randn(1, imgDimX,imgDimY)
    mP = RGFLib.RobustAlgebraicPlaneFittingPy(inX = inX.flatten(), 
                                       inY = inY.flatten(),
                                       inZ = img3.flatten())
    print(mP)
    inTensor = np.concatenate((img1, img2, img3))
    print('input Tensor shape is: ', str(inTensor.shape))
    modelParamsMap = RSGImagesInTensorPy(inTensor)
    print(modelParamsMap)


def test_RSGImagesInTensorPy_multiproc():
    img1 = np.random.randn(1, 185,388)
    img2 = 10+np.random.randn(1, 185,388)
    img3 = 100+10*np.random.randn(1, 185,388)
    inTensor = np.concatenate((img1, img2, img3))
    print('input Tensor shape is: ', str(inTensor.shape))
    modelParamsMap = RGFLib.RSGImagesInTensorPy_multiproc(inTensor,
                                  numRowSegs = 9,
                                    numClmSegs = 9)
    plt.imshow(modelParamsMap[0,0,:,:])
    plt.show()
    plt.imshow(modelParamsMap[1,0,:,:])
    plt.show()
    plt.imshow(modelParamsMap[0,1,:,:])
    plt.show()
    plt.imshow(modelParamsMap[1,1,:,:])
    plt.show()
    plt.imshow(modelParamsMap[0,2,:,:])
    plt.show()
    plt.imshow(modelParamsMap[1,2,:,:])
    plt.show()
  
def test_visOrderStat():
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

def test_SginleGaussianTensor():    
    testData = np.random.randn(90000)
    RNN2 = 4.4 + np.random.randn(10000)
    testData = np.concatenate((testData, RNN2)).flatten()
    print('testing RobustMultiGaussiansVecPy')
    mP1, mP2 = RGFLib.RobustMultiGaussiansVecPy(testData, MSSE_LAMBDA=3.0)
    print(mP1)
    print(mP2)
    
    SIGMA = 10
    RNN1 = SIGMA*np.random.randn(5000-500-30, 18, 38)
    RNN2 = 5*SIGMA + 5*SIGMA*np.random.randn(500, 18, 38)
    RNU = 30*SIGMA+SIGMA*np.random.randn(30, 18, 38)

    testData = np.concatenate((RNN1, RNN2))
    testData = np.concatenate((testData, RNU))
    
    print('testing RobustMultiGaussiansTensorPy')
    nowtime = time.time()
    modelParamsMap1, bckGNDParamsMap1 = RGFLib.RobustMultiGaussiansTensorPy(testData, MSSE_LAMBDA=3.0)
    print(time.time() - nowtime)
    print(modelParamsMap1 - bckGNDParamsMap1)
    
    print('testing RobustMultiGaussiansTensorPy_MultiProc')
    nowtime = time.time()
    modelParamsMap2, bckGNDParamsMap2 = RGFLib.RobustMultiGaussiansTensorPy_MultiProc(testData,
                                                        topKthPercentile = 0.9, 
                                                        bottomKthPercentile=0.8,
                                                        MSSE_LAMBDA=3.0)
    print(time.time() - nowtime)
    print(modelParamsMap2 - bckGNDParamsMap2)
    #print(bckGNDParamsMap2)
    #print(bckGNDParamsMap2[0,...] + 4.4*bckGNDParamsMap2[1,...])
    
if __name__ == '__main__':
    #choose a test from above
    test_RMGImagePy()
