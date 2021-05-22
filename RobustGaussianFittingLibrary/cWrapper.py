"""
------------------------------------------------------
This file is part of RobustGaussianFittingLibrary,
a free library WITHOUT ANY WARRANTY
Copyright: 2017-2020 LaTrobe University Melbourne,
           2019-2020 Deutsches Elektronen-Synchrotron
------------------------------------------------------
"""

""" A ctypes wrapper for the Robust Gaussian Fitting Library C file
Nothing to look for in this file, its just a wrapper
"""
import numpy as np
import ctypes
import os
import fnmatch

dir_path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + '..' + os.path.sep
fileNameTemplate = 'RGFLib*.so'
flist = fnmatch.filter(os.listdir(dir_path + os.path.sep), fileNameTemplate)
if(len(flist)==0):	#for those who use make
	dir_path = os.path.dirname(os.path.realpath(__file__))
	fileNameTemplate = 'RGFLib*.so'
	flist = fnmatch.filter(os.listdir(dir_path + os.path.sep), fileNameTemplate)
	
RGFCLib = ctypes.cdll.LoadLibrary(dir_path + os.path.sep + flist[0])

'''
void islandRemoval(unsigned char* inMask, unsigned char* labelMap, 
					  unsigned int X, unsigned int Y, 
					  unsigned int islandSizeThreshold)
'''
RGFCLib.islandRemoval.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags='C_CONTIGUOUS'),
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
  
'''
void indexCheck(float* inTensor, float* targetLoc, unsigned int X, unsigned int Y, unsigned int Z)
'''
RGFCLib.indexCheck.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int, ctypes.c_int, ctypes.c_float]

'''
float MSSE(float *error, unsigned int vecLen, float MSSE_LAMBDA, unsigned int k, float minimumResidual)
'''
RGFCLib.MSSE.restype = ctypes.c_float
RGFCLib.MSSE.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_uint, ctypes.c_float, ctypes.c_uint, ctypes.c_float ]

'''
float MSSEWeighted(float* error, float* weights, unsigned int vecLen, 
                   float MSSE_LAMBDA, unsigned int k, float minimumResidual)
'''
RGFCLib.MSSEWeighted.restype = ctypes.c_float
RGFCLib.MSSEWeighted.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_uint, ctypes.c_float, ctypes.c_uint, ctypes.c_float ]

'''
void fitValue(float* inVec,
			  float* inWeights,
			  float* modelParams,
			  float theta,
			  unsigned int inN,
              float topkPerc,
			  float botkPerc,
              float MSSE_LAMBDA,
			  unsigned char optIters,
              float minimumResidual,
			  unsigned int downSampledSize);
'''
RGFCLib.fitValue.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_float, 
                ctypes.c_int, 
                ctypes.c_float, 
                ctypes.c_float, 
                ctypes.c_float, 
                ctypes.c_uint8, 
                ctypes.c_float, 
                ctypes.c_int]

'''
void fitValue2Skewed(float* inVec,
			         float* inWeights,
			         float* modelParams,
			         float theta,
			         unsigned int inN,
					 float topkPerc,
			         float botkPerc,
                     float MSSE_LAMBDA,
			         unsigned char optIters,
                     float minimumResidual,
			         unsigned int downSampledSize);
'''                    
RGFCLib.fitValue2Skewed.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_float, 
                ctypes.c_int,
                ctypes.c_float, 
                ctypes.c_float, 
                ctypes.c_float, 
                ctypes.c_uint8, 
                ctypes.c_float, 
                ctypes.c_int]


'''
void medianOfFits(float *vec, float *weights, 
                  float *modelParams, float theta, unsigned int N,
                  float topkMin, float topkMax, unsigned int numSamples, float samplePerc,
                  float MSSE_LAMBDA, unsigned char optIters, float minimumResidual) 
'''
RGFCLib.medianOfFits.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_float, ctypes.c_uint32, 
                ctypes.c_float, ctypes.c_float, ctypes.c_uint32, ctypes.c_float,   
                ctypes.c_float, ctypes.c_uint8, ctypes.c_float]                

'''
void RobustAlgebraicLineFitting(float* x, float* y, float* mP, unsigned int N,
							  float topKthPerc, float bottomKthPerc, float MSSE_LAMBDA)
'''                
RGFCLib.RobustAlgebraicLineFitting.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]


'''
void RobustAlgebraicLineFittingTensor(float *inTensorX, float *inTensorY, 
                                        float *modelParamsMap, unsigned int N,
                                        unsigned int X, unsigned int Y, 
                            float topKthPerc, float bottomKthPerc, float MSSE_LAMBDA)
'''
RGFCLib.RobustAlgebraicLineFittingTensor.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                ctypes.c_float, ctypes.c_float, ctypes.c_float]                
                
'''
void fitValueTensor(float* inTensor, float* inWeights, float* modelParamsMap,
					unsigned int N, unsigned int X, unsigned int Y,
					float topkPerc, float botkPerc, float MSSE_LAMBDA,
					unsigned char optIters, float minimumResidual,
					unsigned int downSampledSize);
'''
RGFCLib.fitValueTensor.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_float, ctypes.c_float, ctypes.c_float, 
                ctypes.c_uint8, ctypes.c_float, ctypes.c_uint32]

'''
void RobustAlgebraicPlaneFitting(float* x, float* y, float* z, float* mP, float* mP_Init,
							unsigned int N, float topkPerc, float botkPerc,
							float MSSE_LAMBDA, unsigned char stretch2CornersOpt, 
							float minimumResidual, unsigned char optIters)
'''
RGFCLib.RobustAlgebraicPlaneFitting.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int, ctypes.c_float, ctypes.c_float, 
                ctypes.c_float, ctypes.c_uint8, ctypes.c_float, ctypes.c_uint8]
                
'''
void RSGImage(float* inImage, unsigned char* inMask, float *modelParamsMap,
				unsigned int winX, unsigned int winY,
				unsigned int X, unsigned int Y,
				float topkPerc, float botkPerc,
				float MSSE_LAMBDA, unsigned char stretch2CornersOpt,
				unsigned char numModelParams, unsigned char optIters,
                float minimumResidual)
'''                
RGFCLib.RSGImage.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_uint32, ctypes.c_uint32, 
                ctypes.c_uint32, ctypes.c_uint32, 
                ctypes.c_float, ctypes.c_float, 
                ctypes.c_float, ctypes.c_uint8, 
                ctypes.c_uint8, ctypes.c_uint8, ctypes.c_float]
       
'''
void RSGImage_by_Image_Tensor(float* inImage_Tensor, unsigned char* inMask_Tensor,
						float *model_mean, float *model_std,
						unsigned int winX, unsigned int winY,
						unsigned int N, unsigned int X, unsigned int Y,
						float topkPerc, float botkPerc,
						float MSSE_LAMBDA, unsigned char stretch2CornersOpt,
						unsigned char numModelParams, unsigned char optIters,
                        float minimumResidual)
'''
RGFCLib.RSGImage_by_Image_Tensor.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_uint32, ctypes.c_uint32, 
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_float, ctypes.c_float, 
                ctypes.c_float, ctypes.c_uint8, 
                ctypes.c_uint8, ctypes.c_uint8, ctypes.c_float]

'''
void fitBackgroundRadially(float* inImage, unsigned char* inMask,
                           float* modelParamsMap, float* vecMP,
                           unsigned int minRes,
                           unsigned int maxRes,
                           unsigned int shellWidth,
                           unsigned int stride,
                           unsigned int X_Cent,
                           unsigned int Y_Cent,
                           unsigned char includeCenter,
                           unsigned int finiteSampleBias,
                           unsigned int X, unsigned int Y,
                           float topkPerc, float botkPerc,
                           float MSSE_LAMBDA,
                           unsigned char optIters,
                           float minimumResidual);
'''

RGFCLib.fitBackgroundRadially.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, 
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_uint8, ctypes.c_uint32, 
                ctypes.c_uint32, ctypes.c_uint32, 
                ctypes.c_float, ctypes.c_float, 
                ctypes.c_float, ctypes.c_uint8, ctypes.c_float]

'''
void fitBackgroundCylindrically(float* inTensor,
								unsigned char* inMask,
                                float* modelParamsMap,
								float* vecMP,
                                unsigned int minRes,
                                unsigned int maxRes,
                                unsigned int shellWidth,
                                unsigned char includeCenter,
                                unsigned int finiteSampleBias,
								unsigned int N,
                                unsigned int X,
								unsigned int Y,
                                float topkPerc,
								float botkPerc,
                                float MSSE_LAMBDA,
                                unsigned char optIters,
						        float minimumResidual)
'''
RGFCLib.fitBackgroundCylindrically.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, 
                ctypes.c_uint8, ctypes.c_uint32, 
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, 
                ctypes.c_float, ctypes.c_float, 
                ctypes.c_float, ctypes.c_uint8, ctypes.c_float]
