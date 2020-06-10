""" fit a Gaussian to recover vector value, lines, plane,...
Input arguments
~~~~~~~~~~~~~~~
    MSSE_LAMBDA
    topKthPerc
    bottomKthPerc : set it to 0.95*topKthPerc and if smaller than 8 set it to 8, this number must be more than 12
Output
~~~~~~
    usually the mean and std of the Gaussian
"""
from RGFLib.cWrapper import RGFCLib
import numpy as np
 
def MSSE(inVec, MSSE_LAMBDA = 3.0, k = 12):
    """ A C implementation of MSSE'99
        
        Input arguments
        ~~~~~~~~~~~~~~~
        inVec : the residuals verctor
        MSSE_LAMBDA : How far from mean of the Gaussian, data is inlier normalized by STD of the Gaussian
        k : minimum number of inliers, 12 is the least.
        
        Output
        ~~~~~~
        a scalar, STD of the Gaussian. If you'd like to know its relatino to $\lambda$, have a look at the function visOrderStat in the tests.py 
    """
    return RGFCLib.MSSE((inVec.copy()).astype('float32'),
                        inVec.shape[0],
                        float(MSSE_LAMBDA), k)

###########################################################################################
################################### Robust AVG and STD ####################################
                                       
def fitValue(inVec,
              topKthPerc = 0.5,
              bottomKthPerc = 0.45,
              MSSE_LAMBDA = 3.0,
              modelValueInit = 0,
              optimizerNumIteration = 10):
    """Fit a Gaussian to input vector robustly:
    The function returns the parameters of a single gaussian structure through FLKOS [DICTA'08]
    The default values are suggested in MCNC [CVIU '18]
    if used to remove a large structure, a few points can be left behind as in [cybernetics '19]
    **Note**: This function will only help if you do not have a clustering problem.
    Input arguments
    ~~~~~~~~~~~~~~~
        inVec (numpy.1darray): a float32 input vector
        MSSE_LAMBDA: how far by std, a Guassian is a Guassian, must be above 2 for MSSE.
        topKthPerc, bottomKthPerc: a float32 scalar, roughly guess maximum size of the structure between 0 and 1, the bottomKthPerc is the minimum of it.
            Choose the topKthPerc to be as high as you are sure the portion of data is inlier.
            After you chose the bottomKthPerc, make sure that the remaining number of data points is more than 12.
            For example: Choose the bottomKthPerc so that the sample size remains above 5 [TPAMI16].
            If number of data points is 100, and you are sure that 
    Output
    ~~~~~~
        tuple of two numpy.1darrays, robust average and standard deviation of the guassian
    """
    modelParams = np.zeros(2, dtype='float32')
    RGFCLib.RobustSingleGaussianVec((inVec.copy()).astype('float32'),
                                                   modelParams, modelValueInit,
                                                   inVec.shape[0],
                                                   topKthPerc,
                                                   bottomKthPerc,
                                                   MSSE_LAMBDA,
                                                   optimizerNumIteration)
    return (modelParams[0], modelParams[1])

def fitValueTensor(inTensor,
                  topKthPerc = 0.5,
                  bottomKthPerc=0.45,
                  MSSE_LAMBDA = 3.0):
    """ fit a Gaussian to every vector inside a Tensor, robustly.
    Input arguments
    ~~~~~~~~~~~~~~~
        inTensor: n_F x n_R x n_C Tensor of n_R x n_C vectors, each with size n_F, float32
    Output
    ~~~~~~
        2 x n_R x n_C float32 values, out[0] is mean and out[1] is the STDs for each element
    """
    modelParamsMap = np.zeros((2, inTensor.shape[1], inTensor.shape[2]), dtype='float32')
    RGFCLib.RobustSingleGaussianTensor((inTensor.copy()).astype('float32'),
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
    
def fitLine(inX, inY,
            topKthPerc = 0.5,
            bottomKthPerc=0.45,
            MSSE_LAMBDA = 3.0):
    """ fit a line assuming a Gaussian noise to data points with x and y.
    The line is supposed to be y = ax + Normal(Rmean, RSTD^2)
    Input arguments
    ~~~~~~~~~~~~~~~
        inX: vector of data points x
        inY: vector of data points y
    Output
    ~~~~~~
        3 parameters, a, Rmean, RSTD (your line would be out[0]x + out[1]) 
    """
    
    modelParams = np.zeros(3, dtype='float32')
    RGFCLib.RobustAlgebraicLineFitting((inX.copy()).astype('float32'),
                                            (inY.copy()).astype('float32'),
                                            modelParams, 
                                            inX.shape[0],
                                            topKthPerc,
                                            bottomKthPerc,
                                            MSSE_LAMBDA)
    
    return (modelParams)

def fitLineTensor(inX, inY,
                    topKthPerc = 0.5,
                    bottomKthPerc = 0.45,
                    MSSE_LAMBDA = 3.0):
    """fit a line to every pixel in a Tensor
    Input arguments
    ~~~~~~~~~~~~~~~
        inX: Tensor of data points x, n_F x n_R x n_C
        inY: vector of data points y, n_F x n_R x n_C
    Output
    ~~~~~~
        3 x n_R x n_C, a, Rmean, RSTD fpr each pixel 
    """    
    
    modelParams = np.zeros((3, inX.shape[1], inX.shape[2]), dtype='float32')
    
    RGFCLib.RobustAlgebraicLineFittingTensor( (inX.copy()).astype('float32'),
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
    
def fitPlane(inX, inY, inZ,
                            topKthPerc = 0.5,
                            bottomKthPerc = 0.25,
                            MSSE_LAMBDA = 3.0,
                            stretch2CornersOpt = 2):
    """ fit a plane assuming a Gaussian noise to data points with x, y and z.
    The plane is supposed to be z = ax + by + Normal(Rmean, RSTD^2)
    Input arguments
    ~~~~~~~~~~~~~~~
        inX: vector of data points x
        inY: vector of data points y
        inZ: vector of data points z
    Output
    ~~~~~~
        4 parameters, a, b, Rmean, RSTD (your plane would be out[0]x + out[1]y + out[2]) 
    """

    modelParams = np.zeros(4, dtype='float32')
    RGFCLib.RobustAlgebraicPlaneFitting((inX.copy()).astype('float32'),
                                            (inY.copy()).astype('float32'),
                                            (inZ.copy()).astype('float32'),
                                            modelParams, 
                                            inZ.shape[0],
                                            topKthPerc,
                                            bottomKthPerc,
                                            MSSE_LAMBDA, 
                                            stretch2CornersOpt)
    return (modelParams)

def fitBackground(inImage, 
                inMask = None,
                winX = None,
                winY = None,
                topKthPerc = 0.5,
                bottomKthPerc = 0.45,
                MSSE_LAMBDA = 3.0,
                stretch2CornersOpt = 0,
                numModelParams = 4,
                optIters = 12):
    """ fit a plane to the background of the image and calculate the value of the background plane and STD at the location of each pixel.
    
    Input arguments
    ~~~~~~~~~~~~~~~
        inImage: a 2D float32 numpy array as the image
        inMask: where 0 is bad and 1 is good. The masked pixels have not effect in the calculation of the parameters of the plane fit to background. However, the value of the background at their location can be found.
        winX: size of the window to fit the plane to in rows
        winY: size of the window to fit the plane to in coloumns
        stretch2CornersOpt: An option that helps approximating towards segmentaed planes
            default is zero that does not stretch the model to corner and can be degenerate.
            set it to 4 for a reasonable performance.
        numModelParams: takes either 0, which gives a horizontal plane or 4 which gives an algebraic plane.
        optIters: number of iterations of FLKOS for this fitting
    Output
    ~~~~~~
        2 parameters for each pixel : 2 x n_R, n_C : Rmean and RSTD.
    """
    
    stretch2CornersOpt = np.uint8(stretch2CornersOpt)
    if(inMask is None):
        inMask = np.ones((inImage.shape[0], inImage.shape[1]), dtype='uint8')
        
    if(winX is None):
        winX = inImage.shape[0]
    if(winY is None):
        winY = inImage.shape[1]
    modelParamsMap = np.zeros((2, inImage.shape[0], inImage.shape[1]), dtype='float32')

    RGFCLib.RSGImage(inImage.astype('float32'),
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

def fitBackgroundTensor(inImage_Tensor, 
                inMask_Tensor = None,
                winX = None,
                winY = None,
                topKthPerc = 0.5,
                bottomKthPerc = 0.45,
                MSSE_LAMBDA = 3.0,
                stretch2CornersOpt = 2,
                numModelParams = 4,
                optIters = 12):
    """ fit a plane to each image in the input Tensor and reportbackground values and STD for each pixel for each plane
    
    Input arguments
    ~~~~~~~~~~~~~~~
        inImage_Tensor: n_F x n_R x n_C input Tensor, each image has size n_R x n_C
        inMask_Tensor: same size of inImage_Tensor
    Output
    ~~~~~~
        2 x n_F x n_R x n_C where out[0] would be background mean and out[1] would be STD for each pixel in the Tensor.
    """
    
                
    stretch2CornersOpt = np.uint8(stretch2CornersOpt)
    if(inMask_Tensor is None):
        inMask_Tensor = np.ones(inImage_Tensor.shape, dtype='uint8')
    if(winX is None):
        winX = inImage_Tensor.shape[1]
    if(winY is None):
        winY = inImage_Tensor.shape[2]
    model_mean = np.zeros(inImage_Tensor.shape, dtype='float32')
    model_std  = np.zeros(inImage_Tensor.shape, dtype='float32')

    RGFCLib.RSGImage_by_Image_Tensor(inImage_Tensor.astype('float32'),
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