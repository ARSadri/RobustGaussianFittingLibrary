#################################################################################################
# This file is part of RobustGaussianFittingLibrary, a free library WITHOUT ANY WARRANTY        # 
# Copyright: 2019-2020 Deutsches Elektronen-Synchrotron                                         # 
#################################################################################################

import numpy as np
from multiprocessing import Process, Queue, cpu_count
from .basic import fitValueTensor, fitLineTensor, fitBackgroundTensor, fitBackgroundRadially
from .misc import textProgBar

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

################################################################################################
################################### Value fitting library #######################################

def fitValueTensor_MultiProcFunc(aQ, 
                                partCnt, inTensor, inMask,
                                topKthPerc, bottomKthPerc, MSSE_LAMBDA):

    modelParams = fitValueTensor(inTensor=inTensor, 
                                 inMask = inMask,
                                 topKthPerc=topKthPerc,
                                 bottomKthPerc=bottomKthPerc,
                                 MSSE_LAMBDA = MSSE_LAMBDA)
    aQ.put(list([partCnt, modelParams]))

def fitValueTensor_MultiProc(inTensor,
                             inMask = None,
                             numRowSegs = 1,
                             numClmSegs = 1,
                             topKthPerc = 0.5,
                             bottomKthPerc = 0.4,
                             MSSE_LAMBDA = 3.0,
                             showProgress = False):
    """"Does fitValueTensor in RGFLib.py using multiprocessing
    Input arguments
    ~~~~~~~~~~~~~~~
        inTensor: n_F x n_R x n_C Tensor of n_R x n_C vectors, each with size n_F, float32
        inMask: n_F x n_R x n_C Tensor of n_R x n_C vectors, each with size n_F, uint8
        numRowSegs, numClmSegs: if you have 80 processors, and the image is 512x128, then set them to 7, 11. This way, the patches are almost equal and the processes are spread among the cores. It has no mathematical value.
        MSSE_LAMBDA : How far (normalized by STD of the Gaussian) from the 
                        mean of the Gaussian, data is considered inlier.
        topKthPerc: A rough but certain guess of portion of inliers, between 0 and 1, e.g. 0.5. 
                    Choose the topKthPerc to be as high as you are sure the portion of data is inlier.
                    if you are not sure at all, refer to the note above this code.
                    default : 0.5
        bottomKthPerc: We'd like to make a sample out of worst inliers from data points that are
                       between bottomKthPerc and topKthPerc of sorted residuals.
                       set it to 0.9*topKthPerc, if N is number of data points, then make sure that
                       (topKthPerc - bottomKthPerc)*N>4, 
                       it is best if bottomKthPerc*N>12 then MSSE makes sense
                       otherwise the code may return non-robust results.         
    Output
    ~~~~~~
        2 x n_R x n_C float32 values, out[0] is mean and out[1] is the STDs for each element

    """
    if(inMask is None):
        inMask = np.ones(shape = inTensor.shape, dtype = 'uint8')
    
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
            if(showProgress):
                if(firstProcessed==0):
                    pBar = textProgBar(numProc-1, title = 'Calculationg Values')
                    firstProcessed = 1
                else:
                    pBar.go(1)

        if((numWiating>0) & (numBusyCores < myCPUCount)):
            
            Process(target=fitValueTensor_MultiProcFunc,
                            args=(aQ, partCnt,
                            np.squeeze(inTensor[                                            :,
                                                rowClmInds[partCnt, 0]:rowClmInds[partCnt, 1],
                                                rowClmInds[partCnt, 2]:rowClmInds[partCnt, 3] ]),
                            np.squeeze(inMask[                                            :,
                                                rowClmInds[partCnt, 0]:rowClmInds[partCnt, 1],
                                                rowClmInds[partCnt, 2]:rowClmInds[partCnt, 3] ]),
                            topKthPerc,
                            bottomKthPerc,
                            MSSE_LAMBDA)).start()
            partCnt += 1

            numWiating -= 1
            numBusyCores += 1
    if(showProgress):
        del pBar
    return (modelParamsMap)

################################################################################################
################################### Line fitting library #######################################

def fitLineTensor_MultiProcFunc(aQ, partCnt, 
                                inX, inY,
                                topKthPerc,
                                bottomKthPerc,
                                MSSE_LAMBDA):

    modelParams = fitLineTensor(inX, inY,
                                topKthPerc,
                                bottomKthPerc,
                                MSSE_LAMBDA)
    aQ.put(list([partCnt, modelParams]))

def fitLineTensor_MultiProc(inTensorX, inTensorY,
                              numRowSegs = 1,
                              numClmSegs = 1,
                              topKthPerc = 0.5,
                              bottomKthPerc = 0.4,
                              MSSE_LAMBDA = 3.0,
                              showProgress = False):
    """"Does fitLineTensor in RGFLib.py using multiprocessing
    Input arguments
    ~~~~~~~~~~~~~~~
        inX: Tensor of data points x, n_F x n_R x n_C
        inY: vector of data points y, n_F x n_R x n_C
        numRowSegs, numClmSegs: if you have 80 processors, and the image is 512x128, then set them to 7, 11. This way, the patches are almost equal and the processes are spread among the cores. It has no mathematical value.
        MSSE_LAMBDA : How far (normalized by STD of the Gaussian) from the 
                        mean of the Gaussian, data is considered inlier.
        topKthPerc: A rough but certain guess of portion of inliers, between 0 and 1, e.g. 0.5. 
                    Choose the topKthPerc to be as high as you are sure the portion of data is inlier.
                    if you are not sure at all, refer to the note above this code.
                    default : 0.5
        bottomKthPerc: We'd like to make a sample out of worst inliers from data points that are
                       between bottomKthPerc and topKthPerc of sorted residuals.
                       set it to 0.9*topKthPerc, if N is number of data points, then make sure that
                       (topKthPerc - bottomKthPerc)*N>4, 
                       it is best if bottomKthPerc*N>12 then MSSE makes sense
                       otherwise the code may return non-robust results.         
    Output
    ~~~~~~
        3 x n_R x n_C, a, Rmean, RSTD fpr each pixel 

    """

    rowClmInds, _ = bigTensor2SmallsInds(inTensorX.shape, numRowSegs, numClmSegs)

    numSegs = rowClmInds.shape[0]

    myCPUCount = cpu_count()-1
    aQ = Queue()
    numBusyCores = 0
    numProc = numSegs
    numWiating = numSegs
    numDone = 0
    partCnt = 0
    firstProcessed = 0
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
            if(showProgress):
                if(firstProcessed==0):
                    pBar = textProgBar(numProc-1, title = 'Calculationg line parameters')
                    firstProcessed = 1
                else:
                    pBar.go(1)


        if((numWiating>0) & (numBusyCores < myCPUCount)):
            
            Process(target=fitLineTensor_MultiProcFunc,
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
    if(showProgress):
        del pBar
    return (modelParamsMap)
    
################################################################################################
################################### background estimation library ##############################
    
def fitBackgroundTensor_multiprocFunc(aQ, imgCnt,
                            inImage_Tensor, 
                            inMask_Tensor,
                            winX,
                            winY,
                            topKthPerc,
                            bottomKthPerc,
                            MSSE_LAMBDA,
                            stretch2CornersOpt,
                            numModelParams,
                            optIters,
                            numStrides):
    modelParamsMap = fitBackgroundTensor(inImage_Tensor, 
                                                inMask_Tensor,
                                                winX,
                                                winY,
                                                topKthPerc,
                                                bottomKthPerc,
                                                MSSE_LAMBDA,
                                                stretch2CornersOpt,
                                                numModelParams,
                                                optIters,
                                                numStrides)
    aQ.put(list([imgCnt, modelParamsMap]))
    
def fitBackgroundTensor_multiproc(inDataSet, inMask = None, 
                                        winX = None, winY = None,
                                        topKthPerc = 0.5,
                                        bottomKthPerc = 0.3,
                                        MSSE_LAMBDA = 3.0,
                                        stretch2CornersOpt = 0,
                                        numModelParams = 4,
                                        optIters = 12,
                                        showProgress = False,
                                        numStrides = 2):
    """"Does fitBackgroundTensor in RGFLib.py using multiprocessing
    Input arguments
    ~~~~~~~~~~~~~~~
        inImage_Tensor: n_F x n_R x n_C input Tensor, each image has size n_R x n_C
        inMask_Tensor: same size of inImage_Tensor
        MSSE_LAMBDA : How far (normalized by STD of the Gaussian) from the 
                        mean of the Gaussian, data is considered inlier.
        optIters: number of iterations of FLKOS for this fitting
            value 0: returns total mean and total STD
            value 1: returns topKthPerc percentile and the scale by MSSE.
            value 8 and above is recommended for optimization according 
                    to Newton method
            default : 12
        topKthPerc: A rough but certain guess of portion of inliers, between 0 and 1, e.g. 0.5. 
                    Choose the topKthPerc to be as high as you are sure the portion of data is inlier.
                    if you are not sure at all, refer to the note above this code.
                    default : 0.5
        bottomKthPerc: We'd like to make a sample out of worst inliers from data points that are
                       between bottomKthPerc and topKthPerc of sorted residuals.
                       set it to 0.9*topKthPerc, if N is number of data points, then make sure that
                       (topKthPerc - bottomKthPerc)*N>4, 
                       it is best if bottomKthPerc*N>12 then MSSE makes sense
                       otherwise the code may return non-robust results.     
        numStrides: Convolve the filter this number of times. For example, if the image is 32 by 32
                    and winX and Y are 16 and numStrides is 1, from 0 to 15 and 15 to 31,
                    will be analysed. But if numStrides is 2, from 0 to 15, 10 to 25 and 15 to 31
                    will be analysed and averaged. This means that the method will run 7 times.
    Output
    ~~~~~~
        2 x n_F x n_R x n_C where out[0] would be background mean and out[1] would be STD for each pixel in the Tensor.

    """
    
    f_N = inDataSet.shape[0]
    r_N = inDataSet.shape[1]
    c_N = inDataSet.shape[2]
    
    if(inMask is None):
        inMask = np.ones(inDataSet.shape, dtype='uint8')
    if(winX is None):
        winX = r_N
    if(winY is None):
        winY = c_N
    
    modelParamsMapTensor = np.zeros((2, f_N, r_N, c_N), dtype='float32')

    aQ = Queue()
    mycpucount = cpu_count() - 1
    if(showProgress):
        print('Multiprocessing background ' + str(f_N) + ' frames...')
    numProc = f_N
    numSubmitted = 0
    numProcessed = 0
    numBusyCores = 0
    firstProcessed = 0
    default_stride = int(np.ceil(numProc/(2*mycpucount)))
    while(numProcessed<numProc):
        if (not aQ.empty()):
            qElement = aQ.get()
            _imgCnt = qElement[0]
            _tmpResult = qElement[1]
            _stride = _tmpResult.shape[1]
            modelParamsMapTensor[:, _imgCnt:_imgCnt+_stride, :, :] = _tmpResult
            numBusyCores -= 1
            numProcessed += _stride
            if(showProgress):
                if(firstProcessed==0):
                    pBar = textProgBar(numProc-_stride, title = 'Calculationg background')
                    firstProcessed = 1
                else:
                    pBar.go(_stride)

        if((numSubmitted<numProc) & (numBusyCores < mycpucount)):
            stride = np.minimum(default_stride, numProc - numSubmitted)
            Process(target = fitBackgroundTensor_multiprocFunc, 
                    args=(aQ, numSubmitted, 
                            inDataSet[numSubmitted:numSubmitted+stride, :, :],
                            inMask[numSubmitted:numSubmitted+stride, :, :],
                            winX,
                            winY,
                            topKthPerc,
                            bottomKthPerc,
                            MSSE_LAMBDA,
                            stretch2CornersOpt,
                            numModelParams,
                            optIters,
                            numStrides)).start()
            numSubmitted += stride
            numBusyCores += 1
    if(showProgress):
        del pBar
    return(modelParamsMapTensor)

def fitBackgroundRadiallyTensor_multiprocFunc(aQ, inImg, inMask, 
                                              minRes, includeCenter, maxRes,
                                              shellWidth, numStrides, 
                                              finiteSampleBias, 
                                              topKthPerc,
                                              bottomKthPerc,
                                              MSSE_LAMBDA,
                                              optIters,
                                              imgCnt):
    mP = fitBackgroundRadially(inImg, inMask, 
                               minRes=minRes, includeCenter=includeCenter, maxRes=maxRes,
                               shellWidth=shellWidth, numStrides=numStrides,
                               finiteSampleBias = finiteSampleBias,
                               topKthPerc = topKthPerc,
                               bottomKthPerc = bottomKthPerc,
                               MSSE_LAMBDA = MSSE_LAMBDA,
                               optIters = optIters)
    aQ.put(list([imgCnt, mP]))

def fitBackgroundRadiallyTensor_multiproc(inImg_Tensor,
                                          inMask_Tensor = None,
                                          minRes = 1,
                                          includeCenter = 0,
                                          maxRes = None,
                                          shellWidth = 2,
                                          numStrides = 2,
                                          finiteSampleBias = 400,
                                          topKthPerc = 0.5,
                                          bottomKthPerc = 0.35,
                                          MSSE_LAMBDA = 3.0,
                                          optIters = 12,                         
                                          showProgress = False):
    """ using Multiprocessing in python, 
        fit a value to the ring around the image and fine tune it by convolving the resolution shells
        by number of stride and calculate the value of the background of the ring
        and STD at the location of each pixel.
    
    Input arguments
    ~~~~~~~~~~~~~~~
        inImg_Tensor: a 3D float32 numpy array as the tensor of n_F images: n_f x n_R x n_C
        inMask_Tensor: same size as inImg_Tensor, with data type 'uint8',
                        where 0 is bad and 1 is good. The masked pixels have not effect
                        in the calculation of the parameters of the plane fit to background.
                        However, the value of the background at their location can be found.
        minRes: minimum distance to the center of the image
            default: 0
        includeCenter: if you'd like to set the minimum to a higher value and yet get the
                        circle inside the minimum resolution as one area, set this to one.
                        this is particularly useful when shellWidth=1, then the area within
                        radius 6 will have size of more than 200, the finiteSampleBias pf monteCarlo.
                        So you can set the minRes to 6, set includeCenter to 1 and shellWidth to 1.
            default: 0
        maxRes: maximum distance to the center of the image
        shellWidth : the ring around the center can have a width and a value will be fitted to
                all calue in the ring.
        finiteSampleBias : size of an area on a ring will be downsampled evenly to no more than finiteSampleBias
            default : twice monte carlo finite sample bias 2x200
        optIters: number of iterations of FLKOS for this fitting
            value 0: returns total mean and total STD
            value 1: returns topKthPerc percentile and the scale by MSSE.
            value 8 and above is recommended for optimization according 
                    to Newton method
            default : 12
        MSSE_LAMBDA : How far (normalized by STD of the Gaussian) from the 
                        mean of the Gaussian, data is considered inlier.
                        default: 3.0
        topKthPerc: A rough but certain guess of portion of inliers, between 0 and 1, e.g. 0.5. 
                    Choose the topKthPerc to be as high as you are sure the portion of data is inlier.
                    if you are not sure at all, refer to the note above this code.
                    default : 0.5
        bottomKthPerc: We'd like to make a sample out of worst inliers from data points that are
                       between bottomKthPerc and topKthPerc of sorted residuals.
                       set it to 0.9*topKthPerc, if N is number of data points, then make sure that
                       (topKthPerc - bottomKthPerc)*N>4, 
                       it is best if bottomKthPerc*N>12 then MSSE makes sense
                       otherwise the code may return non-robust results.
        numStrides: by giving a shellWidth>1, one can desire convolving the shell over radius by
            number of strides.
        showProgress: shows progress, default: False
    Output
    ~~~~~~
        numpy array with 3 parameters for each pixel : 2 x n_F x n_R, n_C : Rmean and RSTD.
    """


    if(inMask_Tensor is None):
        inMask_Tensor = np.ones(inImg_Tensor.shape, dtype='uint8')

    n_F = inImg_Tensor.shape[0]
    n_R = inImg_Tensor.shape[1]
    n_C = inImg_Tensor.shape[2]

    if(showProgress):
        print('Use radial info to find abnormal behaviour', flush=True)
    imgAbMap = np.zeros((2, n_F, n_R, n_C), dtype='float32')
    if(showProgress):
        print('inImg_Tensor shape-->', inImg_Tensor.shape)
    
    myCPUCount = cpu_count()-1
    aQ = Queue()
    numProc = n_F
    procID = 0
    numProcessed = 0
    numBusyCores = 0
    if(showProgress):
        print('starting ' +str(numProc) + ' processes with ' + str(myCPUCount) + ' CPUs')
    while(numProcessed<numProc):
        if (not aQ.empty()):
            aQElement = aQ.get()
            _imgCnt = aQElement[0]
            imgAbMap[:, _imgCnt, :, :] = aQElement[1]
            numProcessed += 1
            numBusyCores -= 1
            if(showProgress):
                if(numProcessed == 1):
                    pBar = textProgBar(numProc-1, title = 'Multiprocessing results progress bar')
                if(numProcessed > 1):
                    pBar.go()

        if((procID<numProc) & (numBusyCores < myCPUCount)):
            Process(target = fitBackgroundRadiallyTensor_multiprocFunc, 
                    args = (aQ,
                            inImg_Tensor[procID],
                            inMask_Tensor[procID],
                            minRes,
                            includeCenter,
                            maxRes,
                            shellWidth, 
                            numStrides,
                            finiteSampleBias,
                            topKthPerc,
                            bottomKthPerc,
                            MSSE_LAMBDA,
                            optIters,
                            procID)).start()
            procID += 1
            numBusyCores += 1
    if(showProgress):
        del pBar    
    
    return(imgAbMap)
