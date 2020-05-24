import numpy as np
import ctypes
import os
from multiprocessing import Process, Queue, cpu_count
from robustLib.textProgBar import textProgBar

import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

##########################################################################################
############################# a header for Ctypes functions ##############################
dir_path = os.path.dirname(os.path.realpath(__file__))
RobustGausFitCLib = ctypes.cdll.LoadLibrary(dir_path + '/RobustGausFitLib.so')

'''
void islandRemoval(unsigned char* inMask, unsigned char* outMask, 
					  unsigned int X, unsigned int Y, 
					  unsigned int islandSizeThreshold)
'''
RobustGausFitCLib.islandRemoval.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
  
'''
void indexCheck(float* inTensor, float* targetLoc, unsigned int X, unsigned int Y, unsigned int Z)
'''
RobustGausFitCLib.indexCheck.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_int, ctypes.c_float]

'''
float MSSE(float *error, unsigned int vecLen, float MSSE_LAMBDA, unsigned int k)
'''
RobustGausFitCLib.MSSE.restype = ctypes.c_float
RobustGausFitCLib.MSSE.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_float, ctypes.c_int ]

'''
void RobustSingleGaussianVec(float *vec, float *modelParams, float theta, unsigned int N,
		float topKthPerc, float bottomKthPerc, float MSSE_LAMBDA, unsigned char optIters)
'''
RobustGausFitCLib.RobustSingleGaussianVec.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_float, ctypes.c_int, ctypes.c_float, 
                ctypes.c_float, ctypes.c_float, ctypes.c_uint8]

'''
void RobustAlgebraicLineFitting(float* x, float* y, float* mP, unsigned int N,
							  float topKthPerc, float bottomKthPerc, float MSSE_LAMBDA)
'''                
RobustGausFitCLib.RobustAlgebraicLineFitting.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]


'''
void RobustAlgebraicLineFittingTensor(float *inTensorX, float *inTensorY, 
                                        float *modelParamsMap, unsigned int N,
                                        unsigned int X, unsigned int Y, 
                            float topKthPerc, float bottomKthPerc, float MSSE_LAMBDA)
'''
RobustGausFitCLib.RobustAlgebraicLineFittingTensor.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                ctypes.c_float, ctypes.c_float, ctypes.c_float]                
                
'''
void RobustSingleGaussianTensor(float *inTensor, float *modelParamsMap,
    unsigned int N, unsigned int X,
    unsigned int Y, float topKthPerc, float bottomKthPerc, float MSSE_LAMBDA))
'''
RobustGausFitCLib.RobustSingleGaussianTensor.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float, ctypes.c_float]

'''
void RobustAlgebraicPlaneFitting(float* x, float* y, float* z, float* mP,
							unsigned int N, float topKthPerc, float bottomKthPerc, 
							float MSSE_LAMBDA, unsigned char stretch2CornersOpt)
'''                            
RobustGausFitCLib.RobustAlgebraicPlaneFitting.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_float, ctypes.c_float, 
                ctypes.c_float, ctypes.c_uint8]
                
'''
void RSGImage(float* inImage, unsigned char* inMask, float *modelParamsMap,
				unsigned int winX, unsigned int winY,
				unsigned int X, unsigned int Y, 
				float topKthPerc, float bottomKthPerc, 
				float MSSE_LAMBDA, unsigned char stretch2CornersOpt,
				unsigned char numModelParams, unsigned char optIters)

'''                
RobustGausFitCLib.RSGImage.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_uint32, ctypes.c_uint32, 
                ctypes.c_uint32, ctypes.c_uint32, 
                ctypes.c_float, ctypes.c_float, 
                ctypes.c_float, ctypes.c_uint8, 
                ctypes.c_uint8, ctypes.c_uint8]
       
'''
void RSGImage_by_Image_Tensor(float* inImage_Tensor, unsigned char* inMask_Tensor, 
						float *model_mean, float *model_std,
						unsigned int winX, unsigned int winY,
						unsigned int N, unsigned int X, unsigned int Y, 
						float topKthPerc, float bottomKthPerc, 
						float MSSE_LAMBDA, unsigned char stretch2CornersOpt,
						unsigned char numModelParams, unsigned char optIters)
'''
RobustGausFitCLib.RSGImage_by_Image_Tensor.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_uint32, ctypes.c_uint32, 
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_float, ctypes.c_float, 
                ctypes.c_float, ctypes.c_uint8, 
                ctypes.c_uint8, ctypes.c_uint8]
          
################################## end of Ctypes functions ###############################
##########################################################################################

###########################################################################################
################################### Basic functions #######################################

def islandRemovalPy(inMask, 
            islandSizeThreshold = 1):
    outMask = np.zeros(inMask.shape, dtype='uint8')
    RobustGausFitCLib.islandRemoval(1 - inMask.astype('uint8'),
                                   outMask,
                                   inMask.shape[0],
                                   inMask.shape[1],
                                   islandSizeThreshold)
    return(outMask+inMask)  
    
def indexCheck():
    inTensor = np.zeros((3,4), dtype='float32')
    for rCnt in range(3):
        for cCnt in range(4):
            inTensor[rCnt, cCnt] = rCnt + 10*cCnt
    targetLoc = np.zeros(2, dtype='float32')
    RobustGausFitCLib.indexCheck(inTensor, targetLoc, 3, 4, 21.0)
    print(targetLoc)

def sGHist(inVec, mP, SNR_ACCEPT=3.0):
    tmpL  = (inVec[  (inVec<=mP[0]-SNR_ACCEPT*mP[1]) & (inVec>=mP[0]-4*SNR_ACCEPT*mP[1])  ]).copy()
    tmpM  = (inVec[(inVec>mP[0]-SNR_ACCEPT*mP[1]) & (inVec<mP[0]+SNR_ACCEPT*mP[1])]).copy()
    tmpH  = (inVec[  (inVec>=mP[0]+SNR_ACCEPT*mP[1]) & (inVec<=mP[0]+4*SNR_ACCEPT*mP[1]) ]).copy()
    _xlimMin = tmpM.min()
    _xlimMax = tmpM.max()
    plt.figure()
    if (tmpL.any()):
        hist,bin_edges = np.histogram(tmpL, tmpL.shape[0])
        plt.bar(bin_edges[:-1], hist, width = 0.1*tmpL.std()/SNR_ACCEPT, color='b',alpha=0.5)
        _xlimMin = tmpL.min()
    hist,bin_edges = np.histogram(tmpM, 40)
    tmpMmax = hist.max()
    plt.bar(bin_edges[:-1], hist, width = 0.5*tmpM.std()/SNR_ACCEPT, color='g',alpha=0.5)
    if (tmpH.any()):
        hist,bin_edges = np.histogram(tmpH, tmpH.shape[0])
        plt.bar(bin_edges[:-1], hist, width = 0.1*tmpH.std()/SNR_ACCEPT, color='r',alpha=0.5)
        _xlimMax = tmpH.max()
    x = np.linspace(mP[0]-SNR_ACCEPT*mP[1], mP[0]+SNR_ACCEPT*mP[1], 1000)
    y = tmpMmax * np.exp(-(x-mP[0])*(x-mP[0])/(2*mP[1]*mP[1])) 
    plt.plot(x,y, 'm')
    plt.xlim(_xlimMin, _xlimMax)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value',fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.title('Normal Distribution Histogram',fontsize=15)
    plt.show()

def sGHist_multi_mP(inVec, mP, SNR=3.0):
    numModels = mP.shape[1]
    flag = np.zeros(inVec.size)
    plt.figure()
    for mCnt in range(numModels):
        flag[(inVec>=mP[0, mCnt]-SNR*mP[1, mCnt]) & (inVec<=mP[0, mCnt]+SNR*mP[1, mCnt])] = mCnt + 1
        modelVec = inVec[flag == mCnt + 1].copy()
        hist,bin_edges = np.histogram(modelVec, 40)
        tmpMmax = hist.max()
        plt.bar(bin_edges[:-1], hist, width = mP[1, mCnt]/SNR,alpha=0.5)
        x = np.linspace(mP[0, mCnt]-SNR*mP[1, mCnt], mP[0, mCnt]+SNR*mP[1, mCnt], 1000)
        y = tmpMmax * np.exp(-((x-mP[0, mCnt])**2)/(2*mP[1, mCnt]**2)) 
        plt.plot(x,y)
    
    modelVec = inVec[flag == 0]
    #hist,bin_edges = np.histogram(modelVec, modelVec.shape[0])
    #plt.bar(bin_edges[:-1], hist, width =, color='g',alpha=0.5)
    plt.bar(modelVec, np.ones(modelVec.size), color='g',alpha=0.5)
    plt.show()
    
def bigTensor2SmallsInds(inTensor_shape, numRowSegs, numClmSegs):
    """
    This function gives indices by which a large tensor is broken into smaller segments,
    the input shape is FxRxC and the output would be indices that makes up a tensor Fx(R/rs)x(C/cs)
    Output:
        rowClmInds (rs x cs, 4) where rowClmInds (rs x cs, 0) is the row start index
                                    rowClmInds (rs x cs, 1) is the row end index
                                    rowClmInds (rs x cs, 2) is the coloumn start index
                                    rowClmInds (rs x cs, 3) is the coloumn end index
                                    rs x cs will be the segment number going in row direction first.
    Note: because we use linspace, obviously, one of the parts will be larger in case the sizes don't match
    """
    #print('meshgrid indices for shape: ' + str(inTensor_shape))
    #print('divided into ' + str(numRowSegs) + ' row segments and into '+ str(numClmSegs) + ' clm segments')
    rowClmInds = np.zeros((numRowSegs*numClmSegs, 4), dtype='int')
    rowStarts = np.linspace(0, inTensor_shape[1], numRowSegs+1, dtype='int')
    rowEnds = rowStarts
    rowStarts = rowStarts[:-1]
    rowEnds = rowEnds[1:]
    clmStarts = np.linspace(0, inTensor_shape[2], numClmSegs+1, dtype='int')
    clmEnds = clmStarts
    clmStarts = clmStarts[:-1]
    clmEnds = clmEnds[1:]
    segCnt = 0
    segInds = np.zeros((numRowSegs* numClmSegs, 2), dtype='int')
    for rcnt in range(numRowSegs):
        for ccnt in range(numClmSegs):
            rowClmInds[segCnt, 0] = rowStarts[rcnt]
            rowClmInds[segCnt, 1] = rowEnds[rcnt]
            rowClmInds[segCnt, 2] = clmStarts[ccnt]
            rowClmInds[segCnt, 3] = clmEnds[ccnt]
            segInds[segCnt, 0] = rcnt
            segInds[segCnt, 1] = ccnt
            segCnt += 1

    #print('meshgrid number of parts: ' + str(segCnt))
    return(rowClmInds, segInds)
            
def MSSEPy(inVec, 
            MSSE_LAMBDA = 3.0, k = 12):
    return RobustGausFitCLib.MSSE((inVec.copy()).astype('float32'),
                                       inVec.shape[0],
                                       float(MSSE_LAMBDA), k)

###########################################################################################
################################### Robust average  #######################################
                                       
def RobustSingleGaussianVecPy(inVec,
                              topKthPerc = 0.5,
                              bottomKthPerc=0.45,
                              MSSE_LAMBDA = 3.0,
                              modelValueInit = 0,
                              optimizerNumIteration = 10):
    """
    finds the parameters of a single gaussian structure through FLKOS [DICTA'08]
    The Perc and sample size can be the same as suggested in MCNC [CVIU '18]
    if used to remove a large stryctyre, a few points can be left behind as in [cybernetics '19]
    Arguments:
        inVec (numpy.1darray): a float32 input vector
        MSSE_LAMBDA: how far by std, a Guassian is a Guassian, must be above 2 for MSSE.
    Returns:
        numpy.1darray of 2 elements, average and standard deviation of the guassian
    """
    modelParams = np.zeros(2, dtype='float32')
    RobustGausFitCLib.RobustSingleGaussianVec((inVec.copy()).astype('float32'),
                                                   modelParams, modelValueInit,
                                                   inVec.shape[0],
                                                   topKthPerc,
                                                   bottomKthPerc,
                                                   MSSE_LAMBDA,
                                                   optimizerNumIteration)
    return (modelParams)

def RobustSingleGaussianTensorPy(inTensor,
                              topKthPerc = 0.5,
                              bottomKthPerc=0.45,
                              MSSE_LAMBDA = 3.0):
    modelParamsMap = np.zeros((2, inTensor.shape[1], inTensor.shape[2]), dtype='float32')
    RobustGausFitCLib.RobustSingleGaussianTensor((inTensor.copy()).astype('float32'),
                                                        modelParamsMap,
                                                        inTensor.shape[0],
                                                        inTensor.shape[1],
                                                        inTensor.shape[2],
                                                        topKthPerc,
                                                        bottomKthPerc,
                                                        MSSE_LAMBDA)
    return (modelParamsMap)

def RobustSingleGaussiansTensorPy_MultiProcFunc(aQ, 
                            partCnt, inTensor,
                            topKthPerc, bottomKthPerc, MSSE_LAMBDA):

    modelParams = RobustSingleGaussianTensorPy(inTensor=inTensor,
                        topKthPerc=topKthPerc,
                        bottomKthPerc=bottomKthPerc,
                        MSSE_LAMBDA = MSSE_LAMBDA)
    aQ.put(list([partCnt, modelParams]))

def RobustSingleGaussiansTensorPy_MultiProc(inTensor,
                              numRowSegs = 1,
                              numClmSegs = 1,
                              topKthPerc = 0.5,
                              bottomKthPerc = 0.4,
                              MSSE_LAMBDA = 3.0):

    rowClmInds, segInds = bigTensor2SmallsInds(inTensor.shape, numRowSegs, numClmSegs)

    numSegs = rowClmInds.shape[0]

    myCPUCount = cpu_count()-1
    aQ = Queue()
    numBusyCores = 0
    numProc = numSegs
    numWiating = numSegs
    numDone = 0
    partCnt = 0
    firstProcessed = 0
    modelParamsMap = np.zeros((2, inTensor.shape[1], inTensor.shape[2]), dtype='float32')
    while(numDone<numProc):
        if (not aQ.empty()):
            aQElement = aQ.get()
            _partCnt = aQElement[0]
            modelParamsMap[:,
                           rowClmInds[_partCnt, 0]: rowClmInds[_partCnt, 1],
                           rowClmInds[_partCnt, 2]: rowClmInds[_partCnt, 3] ] = aQElement[1]
            numDone += 1
            numBusyCores -= 1
            if(firstProcessed==0):
                pBar = textProgBar(numProc-1, title = 'Calculationg background')
                firstProcessed = 1
            else:
                pBar.go(1)

        if((numWiating>0) & (numBusyCores < myCPUCount)):
            
            Process(target=RobustSingleGaussiansTensorPy_MultiProcFunc,
                            args=(aQ, partCnt,
                            np.squeeze(inTensor[                                            :,
                                                rowClmInds[partCnt, 0]:rowClmInds[partCnt, 1],
                                                rowClmInds[partCnt, 2]:rowClmInds[partCnt, 3] ]),
                            topKthPerc,
                            bottomKthPerc,
                            MSSE_LAMBDA)).start()
            partCnt += 1

            numWiating -= 1
            numBusyCores += 1
    del pBar
    return (modelParamsMap)
    
################################################################################################
################################### Line fitting library #######################################
    
def RobustAlgebraicLineFittingPy(inX, inY,
                            topKthPerc = 0.5,
                            bottomKthPerc=0.45,
                            MSSE_LAMBDA = 3.0):
    modelParams = np.zeros(3, dtype='float32')
    RobustGausFitCLib.RobustAlgebraicLineFitting((inX.copy()).astype('float32'),
                                            (inY.copy()).astype('float32'),
                                            modelParams, 
                                            inX.shape[0],
                                            topKthPerc,
                                            bottomKthPerc,
                                            MSSE_LAMBDA)
    return (modelParams)

def RobustAlgebraicLineFittingTensorPy(inX, inY,
                            topKthPerc = 0.5,
                            bottomKthPerc = 0.45,
                            MSSE_LAMBDA = 3.0):
    modelParams = np.zeros((3, inX.shape[1], inX.shape[2]), dtype='float32')
    RobustGausFitCLib.RobustAlgebraicLineFittingTensor( (inX.copy()).astype('float32'),
                                                        (inY.copy()).astype('float32'),
                                                        modelParams, 
                                                        inX.shape[0],
                                                        inX.shape[1],
                                                        inX.shape[2],
                                                        topKthPerc,
                                                        bottomKthPerc,
                                                        MSSE_LAMBDA)
    return (modelParams)

def RobustAlgebraicLineFittingTensorPy_MultiProcFunc(aQ, partCnt, 
                                                        inX, inY,
                                                        topKthPerc,
                                                        bottomKthPerc,
                                                        MSSE_LAMBDA):

    modelParams = RobustAlgebraicLineFittingTensorPy(inX, inY,
                                                    topKthPerc,
                                                    bottomKthPerc,
                                                    MSSE_LAMBDA)
    aQ.put(list([partCnt, modelParams]))

def RobustAlgebraicLineFittingTensorPy_MultiProc(inTensorX, inTensorY,
                              numRowSegs = 1,
                              numClmSegs = 1,
                              topKthPerc = 0.5,
                              bottomKthPerc = 0.4,
                              MSSE_LAMBDA = 3.0):

    rowClmInds, _ = bigTensor2SmallsInds(inTensorX.shape, numRowSegs, numClmSegs)

    numSegs = rowClmInds.shape[0]

    myCPUCount = cpu_count()-1
    aQ = Queue()
    numBusyCores = 0
    numProc = numSegs
    numWiating = numSegs
    numDone = 0
    partCnt = 0
    
    modelParamsMap = np.zeros((3, inTensorX.shape[1], inTensorX.shape[2]), dtype='float32')
    while(numDone<numProc):
        if (not aQ.empty()):
            aQElement = aQ.get()
            _partCnt = aQElement[0]
            modelParamsMap[:,
                           rowClmInds[_partCnt, 0]: rowClmInds[_partCnt, 1],
                           rowClmInds[_partCnt, 2]: rowClmInds[_partCnt, 3] ] = aQElement[1]
            numDone += 1
            numBusyCores -= 1

        if((numWiating>0) & (numBusyCores < myCPUCount)):
            
            Process(target=RobustAlgebraicLineFittingTensorPy_MultiProcFunc,
                            args=(aQ, partCnt,
                            np.squeeze(inTensorX[                                            :,
                                                rowClmInds[partCnt, 0]:rowClmInds[partCnt, 1],
                                                rowClmInds[partCnt, 2]:rowClmInds[partCnt, 3] ]),
                            np.squeeze(inTensorY[                                            :,
                                                rowClmInds[partCnt, 0]:rowClmInds[partCnt, 1],
                                                rowClmInds[partCnt, 2]:rowClmInds[partCnt, 3] ]),
                            topKthPerc,
                            bottomKthPerc,
                            MSSE_LAMBDA)).start()
            partCnt += 1

            numWiating -= 1
            numBusyCores += 1

    return (modelParamsMap)
    
################################################################################################
################################### background estimation library ##############################
    
def RobustAlgebraicPlaneFittingPy(inX, inY, inZ,
                            topKthPerc = 0.5,
                            bottomKthPerc = 0.25,
                            MSSE_LAMBDA = 3.0,
                            stretch2CornersOpt = 2):
    modelParams = np.zeros(4, dtype='float32')
    RobustGausFitCLib.RobustAlgebraicPlaneFitting((inX.copy()).astype('float32'),
                                            (inY.copy()).astype('float32'),
                                            (inZ.copy()).astype('float32'),
                                            modelParams, 
                                            inZ.shape[0],
                                            topKthPerc,
                                            bottomKthPerc,
                                            MSSE_LAMBDA, 
                                            stretch2CornersOpt)
    return (modelParams)

def RMGImagePy(inImage, 
                inMask = None,
                winX = None,
                winY = None,
                topKthPerc = 0.5,
                bottomKthPerc = 0.45,
                MSSE_LAMBDA = 3.0,
                stretch2CornersOpt = 0,
                numModelParams = 4,
                optIters = 12):
    stretch2CornersOpt = np.uint8(stretch2CornersOpt)
    if(inMask is None):
        inMask = np.ones((inImage.shape[0], inImage.shape[1]), dtype='uint8')
        
    if(winX is None):
        winX = inImage.shape[0]
    if(winY is None):
        winY = inImage.shape[1]
    modelParamsMap = np.zeros((2, inImage.shape[0], inImage.shape[1]), dtype='float32')

    RobustGausFitCLib.RSGImage(inImage.astype('float32'),
                                inMask,
                                modelParamsMap,
                                winX,
                                winY,
                                inImage.shape[0],
                                inImage.shape[1],
                                topKthPerc,
                                bottomKthPerc,
                                MSSE_LAMBDA,
                                stretch2CornersOpt,
                                numModelParams,
                                optIters)
    return (modelParamsMap)

def RSGImage_by_Image_TensorPy(inImage_Tensor, 
                inMask_Tensor = None,
                winX = None,
                winY = None,
                topKthPerc = 0.5,
                bottomKthPerc = 0.45,
                MSSE_LAMBDA = 3.0,
                stretch2CornersOpt = 2,
                numModelParams = 4,
                optIters = 12):
                
    stretch2CornersOpt = np.uint8(stretch2CornersOpt)
    if(inMask_Tensor is None):
        inMask_Tensor = np.ones(inImage_Tensor.shape, dtype='uint8')
    if(winX is None):
        winX = inImage_Tensor.shape[1]
    if(winY is None):
        winY = inImage_Tensor.shape[2]
    model_mean = np.zeros(inImage_Tensor.shape, dtype='float32')
    model_std  = np.zeros(inImage_Tensor.shape, dtype='float32')

    RobustGausFitCLib.RSGImage_by_Image_Tensor(inImage_Tensor.astype('float32'),
                                                inMask_Tensor.astype('uint8'),
                                                model_mean,
                                                model_std,
                                                winX,
                                                winY,
                                                inImage_Tensor.shape[0],
                                                inImage_Tensor.shape[1],
                                                inImage_Tensor.shape[2],
                                                topKthPerc,
                                                bottomKthPerc,
                                                MSSE_LAMBDA,
                                                stretch2CornersOpt,
                                                numModelParams,
                                                optIters)
    
    return ( np.array([model_mean, model_std]))
    
def RSGImage_by_Image_TensorPy_multiprocFunc(queue, imgCnt,
                            inImage_Tensor, 
                            inMask_Tensor,
                            winX,
                            winY,
                            topKthPerc,
                            bottomKthPerc,
                            MSSE_LAMBDA,
                            stretch2CornersOpt,
                            numModelParams,
                            optIters):
    modelParamsMap = RSGImage_by_Image_TensorPy(inImage_Tensor, 
                                                inMask_Tensor,
                                                winX,
                                                winY,
                                                topKthPerc,
                                                bottomKthPerc,
                                                MSSE_LAMBDA,
                                                stretch2CornersOpt,
                                                numModelParams,
                                                optIters)
    queue.put(list([imgCnt, modelParamsMap]))
    
def RSGImage_by_Image_TensorPy_multiproc(inDataSet, inMask = None, 
                                        winX = None, winY = None,
                                        topKthPerc = 0.5,
                                        bottomKthPerc = 0.4,
                                        MSSE_LAMBDA = 3.0,
                                        stretch2CornersOpt = 0,
                                        numModelParams = 4,
                                        optIters = 10):
    
    f_N = inDataSet.shape[0]
    r_N = inDataSet.shape[1]
    c_N = inDataSet.shape[2]
    
    if(inMask is None):
        inMask = np.zeros(inDataSet.shape, dtype='uint8')
    if(winX is None):
        winX = r_N
    if(winY is None):
        winY = c_N
    
    modelParamsMapTensor = np.zeros((2, f_N, r_N, c_N), dtype='float32')

    queue = Queue()
    mycpucount = cpu_count() - 1
    print('Multiprocessing ' + str(f_N) + ' frames...')
    numProc = f_N
    numSubmitted = 0
    numProcessed = 0
    numBusyCores = 0
    firstProcessed = 0
    default_stride = np.maximum(mycpucount, int(np.floor(numProc/mycpucount)))
    while(numProcessed<numProc):
        if (not queue.empty()):
            qElement = queue.get()
            _imgCnt = qElement[0]
            _tmpResult = qElement[1]
            _stride = _tmpResult.shape[1]
            modelParamsMapTensor[:, _imgCnt:_imgCnt+_stride, :, :] = _tmpResult
            numBusyCores -= 1
            numProcessed += _stride
            if(firstProcessed==0):
                pBar = textProgBar(numProc-_stride, title = 'Calculationg background')
                firstProcessed = 1
            else:
                pBar.go(_stride)

        if((numSubmitted<numProc) & (numBusyCores < mycpucount)):
            stride = np.minimum(default_stride, numProc - numSubmitted)
            Process(target = RSGImage_by_Image_TensorPy_multiprocFunc, 
                    args=(queue, numSubmitted, 
                            inDataSet[numSubmitted:numSubmitted+stride, :, :],
                            inMask[numSubmitted:numSubmitted+stride, :, :],
                            winX,
                            winY,
                            topKthPerc,
                            bottomKthPerc,
                            MSSE_LAMBDA,
                            stretch2CornersOpt,
                            numModelParams,
                            optIters)).start()
            numSubmitted += stride
            numBusyCores += 1
    del pBar
    return(modelParamsMapTensor)
