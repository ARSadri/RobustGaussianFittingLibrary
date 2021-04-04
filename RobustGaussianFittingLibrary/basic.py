#################################################################################################
# This file is part of RobustGaussianFittingLibrary, a free library WITHOUT ANY WARRANTY        # 
# Copyright: 2019-2020 Deutsches Elektronen-Synchrotron                                         # 
#################################################################################################

""" fit a Gaussian to recover vector value, lines, plane,...
Input arguments
~~~~~~~~~~~~~~~
    MSSE_LAMBDA : How far (normalized by STD of the Gaussian) from the 
                        mean of the Gaussian, data is considered inlier.
                        default: 3.0
    topKthPerc: A rough but certain guess of portion of inliers, between 0 and 1, e.g. 0.5. 
                    Choose the topKthPerc to be as high as you are sure the portion of data is inlier.
                    if you are not sure at all, you have a problem of estimating structure size
                    That can be solved by the MCNC method which develops covariance of data. 
                    One simpler solution (that maybe slower and less accurate) is to try many and 
                    ensemble the models by their median.
                default : 0.5
    bottomKthPerc : set it to 0.9*topKthPerc, 
                    if N is number of data points, then make sure that
                    (topKthPerc - bottomKthPerc)*N>p+4 [RuwanAliTPAMI16], where p is 
                    number of parameters of the model, 
                    p_valuefitting = 1
                    p_linefitting = 2
                    p_planefitting = 3
                    it is best if bottomKthPerc*N>12 then MSSE makes sense
                    otherwise the code returns non-robust results.
Output
~~~~~~
    usually the mean and std of the Gaussian
"""
from .cWrapper import RGFCLib
import numpy as np
 
def MSSE(inVec, MSSE_LAMBDA = 3.0, k = 12, minimumResidual = 0):
    """ A C implementation of MSSE'99
        
        Input arguments
        ~~~~~~~~~~~~~~~
        inVec : the residuals verctor
        MSSE_LAMBDA : How far (normalized by STD of the Gaussian) from the 
                        mean of the Gaussian, data is considered inlier.
                        default: 3.0
        k : minimum number of inliers, 12 is the least.
        minimumResidual : minimum fitting error to initialize MSSE (dtype = 'float32')
                          default : 0
        
        Output
        ~~~~~~
        a scalar, STD of the Gaussian. If you'd like to know 
        its relatino to $\lambda$, have a look at the function 
        visOrderStat in the tests.py 
    """
    return RGFCLib.MSSE((inVec.copy()).astype('float32'),
                        inVec.shape[0],
                        float(MSSE_LAMBDA), k, minimumResidual)

def MSSEWeighted(inVec, inWeights = None, 
                 MSSE_LAMBDA = 3.0, k = 12, minimumResidual = 0):
    """ A C implementation of MSSE'99
        
        Input arguments
        ~~~~~~~~~~~~~~~
        inVec : the residuals verctor
        inWeights : the weights for residuals
        MSSE_LAMBDA : How far (normalized by STD of the Gaussian) from the 
                        mean of the Gaussian, data is considered inlier.
                        default: 3.0
        k : minimum number of inliers, 12 is the least.
        minimumResidual : minimum fitting error to initialize MSSE (dtype = 'float32')
                          default : 0
        
        Output
        ~~~~~~
        a scalar, STD of the Gaussian. If you'd like to know 
        its relatino to $\lambda$, have a look at the function 
        visOrderStat in the tests.py 
    """
    if ( inWeights is None):
        inWeights = np.ones(shape = inVec.shape, dtype=inVec.dtype)
    return RGFCLib.MSSEWeighted((inVec.copy()).astype('float32'),
                                (inWeights.copy()).astype('float32'),
                                inVec.shape[0],
                                float(MSSE_LAMBDA), k, minimumResidual)

###########################################################################################
################################### Robust AVG and STD ####################################
                                       
def fitValue(inVec, inWeights = None,
             topKthPerc = 0.5,
             bottomKthPerc = 0.35,
             MSSE_LAMBDA = 3.0,
             modelValueInit = 0,
             optIters = 8,
             minimumResidual = 0,
             downSampledSize = np.iinfo("uint32").max):
    """Fit a Gaussian to input vector robustly:
    The function returns the parameters of a single gaussian structure through FLKOS [DICTA'08]
    The default values are suggested in MCNC [CVIU '18]
    if used to remove a large structure, a few points can be left behind as in [cybernetics '19]
    **Note**: This function will only help if you do not have a clustering problem.
    Input arguments
    ~~~~~~~~~~~~~~~
        inVec (numpy.1darray): a float32 input vector
        inWeights : same size as inVec with positive float32 weights for errors
        MSSE_LAMBDA : How far (normalized by STD of the Gaussian) from the 
                        mean of the Gaussian, data is considered inlier.
                        default: 3.0
        optIters: number of iterations of FLKOS for this fitting
            value 0: returns total mean and total STD
            value 1: returns topKthPerc percentile and the scale by MSSE.
            value 8 and above is recommended for optimization according 
                    to Newton method
            default : 8
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
        minimumResidual : minimum fitting error to initialize MSSE (dtype = 'float32')
                          default : 0
        downSampledSize: the data will be downsampled regualrly starting from
            position 0 to have this length. This is used for finding the 
            parameter model and not to find the noise scale. The entire
            inVec will be used to find the noise scale. If you'd like to use
            less part of data for edtimation of the noise which is not recommended
            then down sample the whole thing before you send it to this function
            and set the downSampledSize to inf.
            default: np.iinfo('uint32').max
    Output
    ~~~~~~
        tuple of two numpy.1darrays, robust average and standard deviation of the guassian
    """
    if (inWeights is None):
        inWeights = np.ones(shape = inVec.shape, dtype ='float32')
    modelParams = np.zeros(2, dtype='float32')
    RGFCLib.fitValue((inVec.copy()).astype('float32'),
                     (inWeights.copy()).astype('float32'),
                     modelParams, 
                     modelValueInit,
                     inVec.shape[0],
                     topKthPerc,
                     bottomKthPerc,
                     MSSE_LAMBDA,
                     optIters,
                     minimumResidual,
                     downSampledSize)
    return modelParams

def fitValue2Skewed(inVec, 
                    inWeights = None,
                    topKthPerc = 0.5,
                    bottomKthPerc = 0.35,
                    MSSE_LAMBDA = 3.0,
                    modelValueInit = 0,
                    optIters = 8,
                    minimumResidual = 0):
    """Fit a skewed bell shaped unimodal sharp density robustly:
    The function works exactly the same as the fitValue, it fits 
    a Gaussian to inVec robustly. Except that it accepts weights 
    as well and returns the average and standard deviation of a 
    skewed density, it reports the bigger STD of two sides as 
    standard deviation, and the median of inliers as the mode.
    Input arguments
    ~~~~~~~~~~~~~~~
        inVec (numpy.1darray): a float32 input vector of values 
            to fit the model to
        inWeight (numpy.1darray): a float32 input vector of weights 
            for each data point, doesn't need to sum to 1
        MSSE_LAMBDA : How far (normalized by STD of the Gaussian) from the 
            mean of the Gaussian, data is considered inlier.
            default: 3.0
        optIters: number of iterations of FLKOS for this fitting
            value 0: returns total mean and total STD
            value 1: returns topKthPerc percentile and the scale by MSSE.
            value 8 and above is recommended for optimization similar 
                to Newton method
            default : 8
        topKthPerc: A rough but certain guess of portion of inliers, 
            between 0 and 1, e.g. 0.5. 
            Choose the topKthPerc to be as high as you are 
            sure the portion of data is inlier.
            if you are not sure at all, refer to the note above this code.
            default : 0.5
        bottomKthPerc: We'd like to make a sample out of worst 
            inliers from data points that are
            between bottomKthPerc and topKthPerc of sorted residuals.
            set it to 0.9*topKthPerc, if N is number of data points,
            then make sure that
            (topKthPerc - bottomKthPerc)*N>4, 
            it is best if bottomKthPerc*N>12 then MSSE makes sense
            otherwise the code may return non-robust results.
        minimumResidual : minimum fitting error to initialize MSSE 
            (dtype = 'float32')
            default : 0
    Output
    ~~~~~~
        tuple of two numpy.1darrays, robust mode and standard deviation of the density from the longer tail
    """
    if(inWeights is None):
        inWeights = np.ones(inVec.shape, dtype='float32')
    modelParams = np.zeros(2, dtype='float32')
    RGFCLib.fitValue2Skewed((inVec.copy()).astype('float32'),
                              (inWeights.copy()).astype('float32'),
                               modelParams, 
                               modelValueInit,
                               inVec.shape[0],
                               topKthPerc,
                               bottomKthPerc,
                               MSSE_LAMBDA,
                               optIters,
                               minimumResidual)
    return (modelParams[0], modelParams[1])
    
def medianOfFits(inVec, 
                 inWeights = None,
                 topkMax = 0.5,
                 topkMin = 0.35,
                 numSamples = 50,
                 samplePerc = 0.1,
                 MSSE_LAMBDA = 3.0,
                 modelValueInit = 0,
                 optIters = 12,
                 minimumResidual = 0):
    """Fit a skewed bell shaped unimodal sharp density robustly:
    The function works exactly the same as the fitValue, it fits a Gaussian to inVec robustly. Except that it accepts weights as well and returns the average and standard deviation of a skewed density, it reports the bigger STD of two sides as standard deviation, and the median of inliers as the mode.
    Input arguments
    ~~~~~~~~~~~~~~~
        inVec (numpy.1darray): a float32 input vector of values to fit the model to
        inWeight (numpy.1darray): a float32 input vector of weights for each data point, doesn't need to sum to 1
        MSSE_LAMBDA : How far (normalized by STD of the Gaussian) from the 
                        mean of the Gaussian, data is considered inlier.
                        default: 3.0
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
        minimumResidual : minimum fitting error to initialize MSSE (dtype = 'float32')
                          default : 0
    Output
    ~~~~~~
        tuple of two numpy.1darrays, robust mode and standard deviation of the density from the longer tail
    """
    if(inWeights is None):
        inWeights = np.ones(inVec.shape, dtype='float32')
    modelParams = np.zeros(2, dtype='float32')
    RGFCLib.medianOfFits((inVec.copy()).astype('float32'),
                         (inWeights.copy()).astype('float32'),
                         modelParams, 
                         modelValueInit,
                         inVec.shape[0],
                         topkMin,
                         topkMax,
                         numSamples,
                         samplePerc,
                         MSSE_LAMBDA,
                         optIters,
                         minimumResidual)
    return np.array([modelParams[0], modelParams[1]])

def fitValueTensor(inTensor,
                   inWeights = None,
                   topKthPerc = 0.5,
                   bottomKthPerc=0.35,
                   MSSE_LAMBDA = 3.0,
                   optIters = 12,
                   minimumResidual = 0.0,
                   downSampledSize = np.iinfo('uint32').max):
    """ fit a Gaussian to every vector inside a Tensor, robustly.
    Input arguments
    ~~~~~~~~~~~~~~~
        inTensor: n_F x n_R x n_C Tensor of n_R x n_C vectors, each with size n_F, float32
        inWeights: n_F x n_R x n_C Tensor of n_R x n_C vectors, each with size n_F, float32
        MSSE_LAMBDA : How far (normalized by STD of the Gaussian) from the 
                        mean of the Gaussian, data is considered inlier.
                        default: 3.0
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
        minimumResidual : minimum fitting error to initialize MSSE (dtype = 'float32')
                          default : 0
        downSampledSize: the data will be downsampled regualrly starting from
            position 0 to have this length. This is used for finding the 
            parameter model and not to find the noise scale. The entire
            inVec will be used to find the noise scale. If you'd like to use
            less part of data for edtimation of the noise which is not recommended
            then down sample the whole thing before you send it to this function
            and set the downSampledSize to inf.
            default: np.iinfo('uint32').max                          
    Output
    ~~~~~~
        2 x n_R x n_C float32 values, out[0] is mean and out[1] is the STDs for each element
    """
    if(inWeights is None):
        inWeights = np.ones(shape = inTensor.shape, dtype = 'float32')
    modelParamsMap = np.zeros((2, inTensor.shape[1], inTensor.shape[2]), dtype='float32')
    assert(inTensor.shape == inWeights.shape)
    RGFCLib.fitValueTensor((inTensor.copy()).astype('float32'),
                           (inWeights.copy()).astype('float32'),
                           modelParamsMap,
                           inTensor.shape[0],
                           inTensor.shape[1],
                           inTensor.shape[2],
                           topKthPerc,
                           bottomKthPerc,
                           MSSE_LAMBDA,
                           optIters,
                           minimumResidual,
                           downSampledSize)
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
                  bottomKthPerc = 0.35,
                  MSSE_LAMBDA = 3.0):
    """fit a line to every pixel in a Tensor
    Input arguments
    ~~~~~~~~~~~~~~~
        inX: Tensor of data points x, n_F x n_R x n_C
        inY: vector of data points y, n_F x n_R x n_C
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
             bottomKthPerc = 0.35,
             MSSE_LAMBDA = 3.0,
             stretch2CornersOpt = 0,
             modelParamsInitial = None,
             minimumResidual = 0,
             optIters = 10):
    """ fit a plane assuming a Gaussian noise to data points with x, y and z.
    The plane is supposed to be z = ax + by + Normal(Rmean, RSTD^2)
    Input arguments
    ~~~~~~~~~~~~~~~
        inX: vector of data points x
        inY: vector of data points y
        inZ: vector of data points z
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
    Output
    ~~~~~~
        4 parameters, a, b, Rmean, RSTD (your plane would be out[0]x + out[1]y + out[2]) 
    """

    modelParams = np.zeros(4, dtype='float32')
    if(modelParamsInitial is None):
        modelParamsInitial = modelParams.copy()
    RGFCLib.RobustAlgebraicPlaneFitting((inX.copy()).astype('float32'),
                                            (inY.copy()).astype('float32'),
                                            (inZ.copy()).astype('float32'),
                                            modelParams, 
                                            modelParamsInitial, 
                                            inZ.shape[0],
                                            topKthPerc,
                                            bottomKthPerc,
                                            MSSE_LAMBDA, 
                                            stretch2CornersOpt,
                                            minimumResidual,
                                            optIters)
    return (modelParams)

def fitBackground(inImage, 
                  inMask = None,
                  winX = None,
                  winY = None,
                  topKthPerc = 0.5,
                  bottomKthPerc = 0.35,
                  MSSE_LAMBDA = 3.0,
                  stretch2CornersOpt = 0,
                  numModelParams = 4,
                  optIters = 12,
                  numStrides = 0,
                  minimumResidual = 0):
    """ fit a plane to the background of the image uainf convolving the window by number of strides
        and calculate the value of the background plane and STD at the location of each pixel.
    
    Input arguments
    ~~~~~~~~~~~~~~~
        inImage: a 2D float32 numpy array as the image
        inMask: where 0 is bad and 1 is good. The masked pixels have not effect in the calculation of the parameters of the plane fit to background. However, the value of the background at their location can be found.
        winX: size of the window to fit the plane to in rows
        winY: size of the window to fit the plane to in coloumns
        stretch2CornersOpt: An option that helps approximating towards segmentaed planes
            default is zero that does not stretch the model to corner and can be degenerate.
            set it to 4 for a reasonable performance.
        numModelParams: takes either 1, which gives a horizontal plane or 4 which gives an algebraic plane.
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
        numStrides: Convolve the filter this number of times. For example, if the image is 32 by 32
                    and winX and Y are 16 and numStrides is 1, from 0 to 15 and 15 to 31,
                    will be analysed. But if numStrides is 2, from 0 to 15, 10 to 25 and 15 to 31
                    will be analysed and averaged and so on ...
        minimumResidual: Minimum fitting error if available

    Output
    ~~~~~~
        numpy array with 2 parameters for each pixel : 2 x n_R, n_C : Rmean and RSTD.
    """
    
    stretch2CornersOpt = np.uint8(stretch2CornersOpt)
    if(inMask is None):
        inMask = np.ones((inImage.shape[0], inImage.shape[1]), dtype='uint8')
        
    if(winX is None):
        winX = inImage.shape[0]
    if(winY is None):
        winY = inImage.shape[1]

    n_R = inImage.shape[0]
    n_C = inImage.shape[1]
        
    bckParam = np.zeros((2, n_R, n_C), dtype='float32')
    RGFCLib.RSGImage(inImage.astype('float32'),
                     inMask,
                     bckParam,
                     winX,
                     winY,
                     inImage.shape[0],
                     inImage.shape[1],
                     topKthPerc,
                     bottomKthPerc,
                     MSSE_LAMBDA,
                     stretch2CornersOpt,
                     numModelParams,
                     optIters,
                     minimumResidual)
    
    _sums = np.ones((2, n_R, n_C), dtype='uint8')
    if(numStrides>1):
        wSDListRows = np.linspace(0, winX, numStrides+2, dtype='uint8')[1:-1]
        wSDListClms = np.linspace(0, winY, numStrides+2, dtype='uint8')[1:-1]
        for wSDRow in wSDListRows:
            for wSDClm in wSDListClms:
                _inImage = inImage[wSDRow:n_R-winX+wSDRow ,wSDClm:n_C-winY+wSDClm].copy()
                _inMask = inMask[wSDRow:n_R-winX+wSDRow ,wSDClm:n_C-winY+wSDClm].copy()
                modelParamsMap = np.zeros((2, _inImage.shape[0], _inImage.shape[1]), dtype='float32')
                RGFCLib.RSGImage(_inImage.astype('float32'),
                                 _inMask,
                                 modelParamsMap,
                                 winX,
                                 winY,
                                 _inImage.shape[0],
                                 _inImage.shape[1],
                                 topKthPerc,
                                 bottomKthPerc,
                                 MSSE_LAMBDA,
                                 stretch2CornersOpt,
                                 numModelParams,
                                 optIters,
                                 minimumResidual)
                bckParam[:,wSDRow:n_R-winX+wSDRow ,wSDClm:n_C-winY+wSDClm] += modelParamsMap
                _sums[:,wSDRow:n_R-winX+wSDRow ,wSDClm:n_C-winY+wSDClm] += 1
    return(bckParam / _sums)

def fitBackgroundTensor(inImage_Tensor, 
                        inMask_Tensor = None,
                        winX = None,
                        winY = None,
                        topKthPerc = 0.5,
                        bottomKthPerc = 0.3,
                        MSSE_LAMBDA = 3.0,
                        stretch2CornersOpt = 0,
                        numModelParams = 4,
                        optIters = 12,
                        numStrides = 0,
                        minimumResidual = 0):
    """ fit a plane by convolving the model to each image in the input Tensor and report background values and STD for each pixel for each plane
    
    Input arguments
    ~~~~~~~~~~~~~~~
        inImage_Tensor: n_F x n_R x n_C input Tensor, each image has size n_R x n_C
        inMask_Tensor: same size of inImage_Tensor
        MSSE_LAMBDA : How far (normalized by STD of the Gaussian) from the 
                        mean of the Gaussian, data is considered inlier.
                        default: 3.0
        numModelParams: takes either 1, which gives a horizontal plane or 4 which gives an algebraic plane.
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
        minimumResidual : minimum fitting error if available
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
    
    n_F = inImage_Tensor.shape[0]
    n_R = inImage_Tensor.shape[1]
    n_C = inImage_Tensor.shape[2]
        
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
                                    optIters,
                                    minimumResidual)

    bckParam = np.array([model_mean, model_std])    
    _sums = np.ones((2, n_F, n_R, n_C), dtype='uint8')
    if(numStrides>0):
        wSDListRows = np.linspace(0, winX, numStrides+2, dtype='uint8')[1:-1]
        wSDListClms = np.linspace(0, winY, numStrides+2, dtype='uint8')[1:-1]
        for wSDRow in wSDListRows:
            for wSDClm in wSDListClms:
                _inImage_Tensor = inImage_Tensor[:, wSDRow:n_R-winX+wSDRow ,wSDClm:n_C-winY+wSDClm].copy()
                _inMask_Tensor = inMask_Tensor[:, wSDRow:n_R-winX+wSDRow ,wSDClm:n_C-winY+wSDClm].copy()
    
                model_mean = np.zeros(_inImage_Tensor.shape, dtype='float32')
                model_std  = np.zeros(_inImage_Tensor.shape, dtype='float32')
            
                RGFCLib.RSGImage_by_Image_Tensor(_inImage_Tensor.astype('float32'),
                                                 _inMask_Tensor.astype('uint8'),
                                                 model_mean,
                                                 model_std,
                                                 winX,
                                                 winY,
                                                 _inImage_Tensor.shape[0],
                                                 _inImage_Tensor.shape[1],
                                                 _inImage_Tensor.shape[2],
                                                 topKthPerc,
                                                 bottomKthPerc,
                                                 MSSE_LAMBDA,
                                                 stretch2CornersOpt,
                                                 numModelParams,
                                                 optIters,
                                                 minimumResidual)
    
                bckParam[:,:,wSDRow:n_R-winX+wSDRow ,wSDClm:n_C-winY+wSDClm] += np.array([model_mean, model_std])
                _sums[:,:,wSDRow:n_R-winX+wSDRow ,wSDClm:n_C-winY+wSDClm] += 1
    return(bckParam / _sums)

def fitBackgroundRadially(inImage, 
                          inMask = None,
                          minRes = 1,
                          includeCenter = 0,
                          maxRes = None,
                          shellWidth = 2,
                          topKthPerc = 0.5,
                          bottomKthPerc = 0.35,
                          MSSE_LAMBDA = 3.0,
                          optIters = 12,
                          numStrides = 0,
                          finiteSampleBias = 200,
                          minimumResidual = 0,
                          return_vecMP = False):
    """ fit a value to the ring around the image and fine tune it by convolving the resolution shells
        by number of stride and calculate the value of the background of the ring
        and STD at the location of each pixel.
    
    Input arguments
    ~~~~~~~~~~~~~~~
        inImage: a 2D float32 numpy array as the image
        inMask: where 0 is bad and 1 is good. The masked pixels have not effect in the calculation of the parameters of the plane fit to background. However, the value of the background at their location can be found.
        minRes: minimum distance to the center of the image
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
        minimumResidual: the minimum residual for MSSE to initialize with, similar to RANSAC,
            default: 0
        return_vecMP: True if you'd like to receive a vector with radial info as a vector
            

    Output
    ~~~~~~
        modelParametersMap: numpy array with 2 parameters for each pixel : 2 x n_R, n_C : Rmean and RSTD. In case return_vecMP is set to True, you get a tuple in the output, the first element will be numpy array modelParametersMap which is the same size as the image and 
        the second element will be a vector of size the maximum radius with two elements,
        average and standard deviation.
    """
    
    if(inMask is None):
        inMask = np.ones((inImage.shape[0], inImage.shape[1]), dtype='uint8')
        
    if(maxRes is None):
        maxRes = int(((inImage.shape[0]/2)**2 + (inImage.shape[1]/2)**2)**0.5)

    n_R = inImage.shape[0]
    n_C = inImage.shape[1]

    vecMP = np.zeros((2, int(np.ceil(np.sqrt(n_R*n_R/4+n_C*n_C/4)))), dtype='float32')
    _vecMP = np.zeros((2, int(np.ceil(np.sqrt(n_R*n_R/4+n_C*n_C/4)))), dtype='float32')

    bckParam = np.zeros((2, n_R, n_C), dtype='float32')
    if(shellWidth<1):
        shellWidth = 1
    modelParamsMap = bckParam.copy()
    _sums = 0
    shellStrideList = np.linspace(0, shellWidth, numStrides + 1, dtype='uint8')
    for shellStride in shellStrideList:
        RGFCLib.fitBackgroundRadially(inImage.astype('float32'),
                                      inMask,
                                      modelParamsMap,
                                      _vecMP,
                                      minRes+shellStride,
                                      maxRes,
                                      shellWidth,
                                      includeCenter,
                                      finiteSampleBias,
                                      inImage.shape[0],
                                      inImage.shape[1],
                                      topKthPerc,
                                      bottomKthPerc,
                                      MSSE_LAMBDA,
                                      optIters,
                                      minimumResidual)
        bckParam += modelParamsMap
        vecMP += _vecMP
        _sums += 1
    if(return_vecMP):
        return(bckParam/_sums, vecMP/_sums)
    else:
        return(bckParam/_sums)

def fitValue_by_meanShift(inVec, minSNR = 3.0, MS_numIter = 8):
    MS_mu = inVec.mean()
    MS_std = inVec.std()
    for _ in range(MS_numIter):
        rth = MS_mu + minSNR*MS_std
        lth = MS_mu - minSNR*MS_std
        vec = inVec[(lth<inVec) & (inVec<rth)].copy()
        MS_mu = vec.mean()
        MS_std = vec.std()
    return(np.array([MS_mu, MS_std]))

def fitPlane_by_meanShift(inX, inY, inZ, minSNR = 3.0, MS_numIter = 8):
    res2 = 0*inX.copy()
    for _ in range(MS_numIter):
        mP = fitPlane(inX[res2<=minSNR**2],
                      inY[res2<=minSNR**2],
                      inZ[res2<=minSNR**2],
                      1, 0, MSSE_LAMBDA=0)
        res2 = (inZ - (mP[0]*inX + mP[1]*inY + mP[2]))**2/mP[3]**2
    return(mP)
