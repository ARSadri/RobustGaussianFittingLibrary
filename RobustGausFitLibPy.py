import numpy as np
import ctypes
from multiprocessing import Process, Queue, cpu_count

import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

##########################################################################################
############################# a header for Ctypes functions ##############################
RobustGausFitLib = ctypes.cdll.LoadLibrary("./RobustGausFitLib.so")

'''
void indexCheck(float* inTensor, float* targetLoc, unsigned int X, unsigned int Y, unsigned int Z)
'''
RobustGausFitLib.indexCheck.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_int, ctypes.c_float]

'''
void TLS_AlgebraicPlaneFitting(float* x, float* y, float* z, float* mP, unsigned int N)
'''
RobustGausFitLib.TLS_AlgebraicPlaneFitting.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int]

'''
float MSSE(float *error, unsigned int vecLen, float MSSE_LAMBDA, unsigned int k)
'''
RobustGausFitLib.MSSE.restype = ctypes.c_float
RobustGausFitLib.MSSE.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_float, ctypes.c_int ]

'''
void RobustSingleGaussianVec(float *vec, float *modelParams,
    unsigned int N, float topKthPerc, float bottomKthPerc, float MSSE_LAMBDA))
'''
RobustGausFitLib.RobustSingleGaussianVec.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]

'''
void RobustAlgebraicLineFitting(float* x, float* y, float* mP, unsigned int N,
							  float topKthPerc, float bottomKthPerc, float MSSE_LAMBDA)
'''                
RobustGausFitLib.RobustAlgebraicLineFitting.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]

'''
void RobustSingleGaussianTensor(float *inTensor, float *modelParamsMap,
    unsigned int N, unsigned int X,
    unsigned int Y, float topKthPerc, float bottomKthPerc, float MSSE_LAMBDA))
'''
RobustGausFitLib.RobustSingleGaussianTensor.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float, ctypes.c_float]

'''
void RobustAlgebraicPlaneFitting(float* x, float* y, float* z, float* mP,
							unsigned int N, float topKthPerc, float bottomKthPerc, 
							float MSSE_LAMBDA, unsigned char stretch2CornersOpt)
'''                            
RobustGausFitLib.RobustAlgebraicPlaneFitting.argtypes = [
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
				float topKthPerc, float bottomKthPerc, float MSSE_LAMBDA, 
                unsigned char stretch2CornersOpt)
'''                
RobustGausFitLib.RSGImage.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_int, 
                ctypes.c_int, ctypes.c_int, ctypes.c_float, 
                ctypes.c_float, ctypes.c_float, ctypes.c_uint8]
                
'''
void RSGImagesInTensor(float *inTensor, unsigned char* inMask, 
					float *modelParamsMap, unsigned int N,
					unsigned int X, unsigned int Y, float topKthPerc,
					float bottomKthPerc, float MSSE_LAMBDA, unsigned char stretch2CornersOpt)
'''
RobustGausFitLib.RSGImagesInTensor.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
                np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_uint8]
                
################################## end of Ctypes functions ###############################
##########################################################################################

def indexCheck():
    inTensor = np.zeros((3,4), dtype='float32')
    for rCnt in range(3):
        for cCnt in range(4):
            inTensor[rCnt, cCnt] = rCnt + 10*cCnt
    targetLoc = np.zeros(2, dtype='float32')
    RobustGausFitLib.indexCheck(inTensor, targetLoc, 3, 4, 21.0)
    print(targetLoc)

def TLS_AlgebraicPlaneFittingPY(inX, inY, inZ):
    mP = np.zeros(3, dtype = 'float32')
    RobustGausFitLib.TLS_AlgebraicPlaneFitting( inX.copy().astype('float32'),
                                                inY.copy().astype('float32'),
                                                inZ.copy().astype('float32'),
                                                mP,
                                                inZ.shape[0])
    return(mP)

def sGHist(inVec, mP, SNR_ACCEPT=3.0):
    tmpL  = (inVec[  (inVec<=mP[0]-SNR_ACCEPT*mP[1]) & (inVec>=mP[0]-4*SNR_ACCEPT*mP[1])  ]).copy()
    tmpM  = (inVec[(inVec>mP[0]-SNR_ACCEPT*mP[1]) & (inVec<mP[0]+SNR_ACCEPT*mP[1])]).copy()
    tmpH  = (inVec[  (inVec>=mP[0]+SNR_ACCEPT*mP[1]) & (inVec<=mP[0]+4*SNR_ACCEPT*mP[1]) ]).copy()
    _xlimMin = tmpM.min()
    _xlimMax = tmpM.max()
    plt.figure()
    if (tmpL.any()):
        hist,bin_edges = np.histogram(tmpL, 40)
        plt.bar(bin_edges[:-1], hist, width = 0.5*tmpL.std()/SNR_ACCEPT, color='b',alpha=0.5)
        _xlimMin = tmpL.min()
    hist,bin_edges = np.histogram(tmpM, 40)
    tmpMmax = hist.max()
    plt.bar(bin_edges[:-1], hist, width = 0.5*tmpM.std()/SNR_ACCEPT, color='g',alpha=0.5)
    if (tmpH.any()):
        hist,bin_edges = np.histogram(tmpH, 40)
        plt.bar(bin_edges[:-1], hist, width = 0.5*tmpH.std()/SNR_ACCEPT, color='r',alpha=0.5)
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
    print('meshgrid indices for shape: ' + str(inTensor_shape))
    print('divided into ' + str(numRowSegs) + ' row segments and into '+ str(numClmSegs) + ' clm segments')
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

    print('meshgrid number of parts: ' + str(segCnt))
    return(rowClmInds, segInds)
            
def MSSEPy(inVec, 
            MSSE_LAMBDA = 3.0, k = 12):
    return RobustGausFitLib.MSSE((inVec.copy()).astype('float32'),
                                       inVec.shape[0],
                                       float(MSSE_LAMBDA), k)

def RobustSingleGaussianVecPy(inVec,
                              topKthPerc = 0.5,
                              bottomKthPerc=0.45,
                              MSSE_LAMBDA = 3.0):
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
    RobustGausFitLib.RobustSingleGaussianVec((inVec.copy()).astype('float32'),
                                                   modelParams, inVec.shape[0],
                                                   topKthPerc,
                                                   bottomKthPerc,
                                                   MSSE_LAMBDA)
    return (modelParams)

def RobustAlgebraicPlaneFittingPy(inX, inY, inZ,
                            topKthPerc = 0.5,
                            bottomKthPerc = 0.25,
                            MSSE_LAMBDA = 3.0,
                            stretch2CornersOpt = 2):
    modelParams = np.zeros(4, dtype='float32')
    RobustGausFitLib.RobustAlgebraicPlaneFitting((inX.copy()).astype('float32'),
                                            (inY.copy()).astype('float32'),
                                            (inZ.copy()).astype('float32'),
                                            modelParams, 
                                            inZ.shape[0],
                                            topKthPerc,
                                            bottomKthPerc,
                                            MSSE_LAMBDA, 
                                            stretch2CornersOpt)
    return (modelParams)

def RobustAlgebraicLineFittingPy(inX, inY,
                            topKthPerc = 0.5,
                            bottomKthPerc=0.45,
                            MSSE_LAMBDA = 3.0):
    modelParams = np.zeros(3, dtype='float32')
    RobustGausFitLib.RobustAlgebraicLineFitting((inX.copy()).astype('float32'),
                                            (inY.copy()).astype('float32'),
                                            modelParams, 
                                            inX.shape[0],
                                            topKthPerc,
                                            bottomKthPerc,
                                            MSSE_LAMBDA)
    return (modelParams)
    
def RobustSingleGaussianTensorPy(inTensor,
                              topKthPerc = 0.5,
                              bottomKthPerc=0.45,
                              MSSE_LAMBDA = 3.0):
    modelParamsMap = np.zeros((2, inTensor.shape[1], inTensor.shape[2]), dtype='float32')
    RobustGausFitLib.RobustSingleGaussianTensor((inTensor.copy()).astype('float32'),
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
    print('inTensor shape -> ' + str(inTensor.shape)+ ', parts->' + str(numSegs))
    print('rowClmInds shape->' + str(rowClmInds.shape))
    
    modelParamsMap = np.zeros((2, inTensor.shape[1], inTensor.shape[2]), dtype='float32')
    print('starting ' +str(numProc) + ' processes with ' + str(myCPUCount) + ' CPUs')
    while(numDone<numProc):
        if (not aQ.empty()):
            aQElement = aQ.get()
            _partCnt = aQElement[0]
            modelParamsMap[:,
                           rowClmInds[_partCnt, 0]: rowClmInds[_partCnt, 1],
                           rowClmInds[_partCnt, 2]: rowClmInds[_partCnt, 3] ] = aQElement[1]
            print('-> ' + str(int(100*numDone/numProc)) + '%')
            numDone += 1
            numBusyCores -= 1

        if((numWiating>0) & (numBusyCores < myCPUCount)):
            
            Process(target=patchSTDModelPy_multiprocFunc,
                            args=(aQ, partCnt,
                            np.squeeze(inTensor[                                            :,
                                                rowClmInds[partCnt, 0]:rowClmInds[partCnt, 1],
                                                rowClmInds[partCnt, 2]:rowClmInds[partCnt, 3] ]),
                            topKthPerc,
                            bottomKthPerc,
                            MSSE_LAMBDA)).start()
            partCnt += 1
            print(str(int(100*numWiating/numProc)) +'% ->')

            numWiating -= 1
            numBusyCores += 1
            if(numWiating==0):
                print('all processes are submitted!')

    return (modelParamsMap)

################################### background estimation library ############
def RMGImagePy(inImage, inMask = None,
              winX = None,
              winY = None,
              topKthPerc = 0.5,
              bottomKthPerc = 0.45,
              MSSE_LAMBDA = 3.0,
              stretch2CornersOpt = 2):
    stretch2CornersOpt = np.uint8(stretch2CornersOpt)
    if(inMask is None):
        inMask = np.ones((inImage.shape[0], inImage.shape[1]), dtype='uint8')
        
    if(winX is None):
        winX = inImage.shape[0]
    if(winY is None):
        winY = inImage.shape[1]
    modelParamsMap = np.zeros((2, inImage.shape[0], inImage.shape[1]), dtype='float32')
    RobustGausFitLib.RSGImage(inImage.astype('float32'),
                                inMask,
                                modelParamsMap,
                                winX,
                                winY,
                                inImage.shape[0],
                                inImage.shape[1],
                                topKthPerc,
                                bottomKthPerc,
                                MSSE_LAMBDA,
                                stretch2CornersOpt)
    return (modelParamsMap)

def RSGImagesInTensorPy(inTensor, inMask = None,
                      topKthPerc = 0.5,
                      bottomKthPerc = 0.4,
                      MSSE_LAMBDA = 3.0,
                      stretch2CornersOpt = 2):
    
    if(inMask is None):
        inMask = np.ones((inTensor.shape[1], inTensor.shape[2]), dtype='uint8')

    modelParamsMap = np.zeros((4, inTensor.shape[0]), dtype='float32')
    
    RobustGausFitLib.RSGImagesInTensor((inTensor.copy()).astype('float32'),
                                    inMask,
                                    modelParamsMap,
                                    inTensor.shape[0],
                                    inTensor.shape[1],
                                    inTensor.shape[2],
                                    topKthPerc,
                                    bottomKthPerc,
                                    MSSE_LAMBDA,
                                    stretch2CornersOpt)
    return (modelParamsMap)

def RSGImagesInTensorPy_multiprocFunc(aQ, partCnt, inTensor, inMask,
                  topKthPerc, bottomKthPerc, MSSE_LAMBDA):
    modelParamsMap = RSGImagesInTensorPy(inTensor=inTensor, inMask=inMask,
                        topKthPerc=topKthPerc,
                        bottomKthPerc=bottomKthPerc,
                        MSSE_LAMBDA = MSSE_LAMBDA)
    aQ.put(list([partCnt, modelParamsMap]))

def RSGImagesInTensorPy_multiproc(inTensor, inMask = None,
                              topKthPerc = 0.8,
                              bottomKthPerc = 0.5,
                              numRowSegs = 1,
                              numClmSegs = 1,
                              MSSE_LAMBDA = 3.0):

    if(inMask is None):
        inMask = np.ones((inTensor.shape[1], inTensor.shape[2]), dtype='uint8')
                              
    rowClmInds, segInds = bigTensor2SmallsInds(inTensor.shape, numRowSegs, numClmSegs)

    numSegs = rowClmInds.shape[0]

    myCPUCount = cpu_count()-1
    aQ = Queue()
    numBusyCores = 0
    numProc = numSegs
    numWiating = numSegs
    numDone = 0
    partCnt = 0

    modelParamsMap = np.zeros((2, inTensor.shape[0],
                               inTensor.shape[1], inTensor.shape[2]), dtype='float32')
    print('starting ' +str(numProc) + ' processes with ' + str(myCPUCount) + ' CPUs')
    while(numDone<numProc):
        if ((not aQ.empty()) & ( (numWiating==0) | (numBusyCores >= myCPUCount) )):
            aQElement = aQ.get()
            _partCnt = aQElement[0]
            patchModelParams = aQElement[1]
            patchModelParams= np.tile(patchModelParams,
                                        (rowClmInds[_partCnt, 1] - rowClmInds[_partCnt, 0],
                                        rowClmInds[_partCnt, 3] - rowClmInds[_partCnt, 2],
                                        1,1))
            patchModelParams = np.swapaxes(patchModelParams, 0, 2)
            patchModelParams = np.swapaxes(patchModelParams, 1, 3)
            
            x = np.arange(0, rowClmInds[_partCnt, 1]-rowClmInds[_partCnt, 0], dtype='int')
            y = np.arange(0, rowClmInds[_partCnt, 3]-rowClmInds[_partCnt, 2], dtype='int')
            Y, X = np.meshgrid(y, x)
            xvals = np.tile(X, (inTensor.shape[0],1,1))
            yvals = np.tile(Y, (inTensor.shape[0],1,1))
            bckEst = np.zeros((2, inTensor.shape[0], 
                                    rowClmInds[_partCnt, 1]-rowClmInds[_partCnt, 0], 
                                    rowClmInds[_partCnt, 3]-rowClmInds[_partCnt, 2]))
            # background mean
            bckEst[0,...] = np.squeeze(patchModelParams[0, ...]*xvals + \
                                       patchModelParams[1, ...]*yvals + \
                                       patchModelParams[2, ...])
            # background std
            bckEst[1,...] = patchModelParams[3, ...]
            modelParamsMap[:, :, 
                           rowClmInds[_partCnt, 0]: rowClmInds[_partCnt, 1],
                           rowClmInds[_partCnt, 2]: rowClmInds[_partCnt, 3] ] = bckEst
            print('-> ' + str(int(100*numDone/numProc)) + '%')
            numDone += 1
            numBusyCores -= 1
            if(numBusyCores>myCPUCount*0.75):
                continue;    # empty the queue

        if((numWiating>0) & (numBusyCores < myCPUCount)):
            Process(target=RSGImagesInTensorPy_multiprocFunc,
                            args=(aQ, partCnt,
                            np.squeeze(inTensor[                                            :,
                                                rowClmInds[partCnt, 0]:rowClmInds[partCnt, 1],
                                                rowClmInds[partCnt, 2]:rowClmInds[partCnt, 3] ]),
                            inMask,
                            topKthPerc,
                            bottomKthPerc,
                            MSSE_LAMBDA)).start()
            partCnt += 1
            print(str(int(100*numWiating/numProc)) +'% ->')

            numWiating -= 1
            numBusyCores += 1
            if(numWiating==0):
                print('all processes are submitted!')

    return (modelParamsMap)
