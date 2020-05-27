from robustLib.RGFLib.RobustGausFitLibCTypes import RobustGausFitCLib
import numpy as np

def islandRemovalPy(inMask, 
            islandSizeThreshold = 1):
    outMask = np.zeros(inMask.shape, dtype='uint8')
    RobustGausFitCLib.islandRemoval(1 - inMask.astype('uint8'),
                                   outMask,
                                   inMask.shape[0],
                                   inMask.shape[1],
                                   islandSizeThreshold)
    return(outMask+inMask)  
 
def MSSEPy(inVec, 
            MSSE_LAMBDA = 3.0, k = 12):
    return RobustGausFitCLib.MSSE((inVec.copy()).astype('float32'),
                                       inVec.shape[0],
                                       float(MSSE_LAMBDA), k)

###########################################################################################
################################### Robust AVG and STD ####################################
                                       
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

##########################################################################################
################################### Line fitting library #################################
    
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

##########################################################################################
############################### background estimation library ############################
    
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
