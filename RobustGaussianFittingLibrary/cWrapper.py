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

dir_path = os.path.dirname(
    os.path.realpath(__file__)) + os.path.sep + '..' + os.path.sep
fileNameTemplate = 'RGFLib*.so'
flist = fnmatch.filter(os.listdir(dir_path + os.path.sep), fileNameTemplate)
if(len(flist)==0):	#for those who use make
	dir_path = os.path.dirname(os.path.realpath(__file__))
	fileNameTemplate = 'RGFLib*.so'
	flist = fnmatch.filter(os.listdir(dir_path + os.path.sep), fileNameTemplate)
	
RGFCLib = ctypes.cdll.LoadLibrary(dir_path + os.path.sep + flist[0])

'''
void islandRemoval(char* inMask, char* labelMap, 
					  int X, int Y, 
					  int islandSizeThreshold)
'''
RGFCLib.islandRemoval.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_int8, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_int8, flags='C_CONTIGUOUS'),
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
  
'''
void indexCheck(float* inTensor, float* targetLoc, int X, int Y, int Z)
'''
RGFCLib.indexCheck.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int, ctypes.c_int, ctypes.c_float]

'''
float MSSE(float *error, int vecLen, float MSSE_LAMBDA, int k, float minimumResidual)
'''
RGFCLib.MSSE.restype = ctypes.c_float
RGFCLib.MSSE.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float ]

'''
float MSSEWeighted(float* error, float* weights, int vecLen, 
                   float MSSE_LAMBDA, int k, float minimumResidual)
'''
RGFCLib.MSSEWeighted.restype = ctypes.c_float
RGFCLib.MSSEWeighted.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float ]

'''
void fitValue(float* inVec,
			  float* inWeights,
			  float* modelParams,
			  float theta,
			  int inN,
              float likelyRatio,
			  float certainRatio,
              float MSSE_LAMBDA,
			  char optIters,
              float minimumResidual,
			  int downSampledSize);
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
                ctypes.c_int8, 
                ctypes.c_float, 
                ctypes.c_int]

'''
void fitValue2Skewed(float* inVec,
			         float* inWeights,
			         float* modelParams,
			         float theta,
			         int inN,
					 float likelyRatio,
			         float certainRatio,
                     float MSSE_LAMBDA,
			         char optIters,
                     float minimumResidual,
			         int downSampledSize);
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
                ctypes.c_int8, 
                ctypes.c_float, 
                ctypes.c_int]


'''
void medianOfFits(float *vec, float *weights, 
                  float *modelParams, float theta, int N,
                  float likelyRatio_min, float likelyRatio_max, int numSamples, float sampleRatio,
                  float MSSE_LAMBDA, char optIters, float minimumResidual) 
'''
RGFCLib.medianOfFits.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_float, ctypes.c_int32, 
                ctypes.c_float, ctypes.c_float, ctypes.c_int32, ctypes.c_float,   
                ctypes.c_float, ctypes.c_int8, ctypes.c_float]                

'''
void RobustAlgebraicLineFitting(float* x, float* y, float* mP, int N,
							  float likelyRatio, float certainRatio, float MSSE_LAMBDA)
'''                
RGFCLib.RobustAlgebraicLineFitting.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]


'''
void RobustAlgebraicLineFittingTensor(float *inTensorX, float *inTensorY, 
                                        float *modelParamsMap, int N,
                                        int X, int Y, 
                            float likelyRatio, float certainRatio, float MSSE_LAMBDA)
'''
RGFCLib.RobustAlgebraicLineFittingTensor.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float, ctypes.c_float]                
                
'''
void fitValueTensor(float* inTensor, float* inWeights, float* modelParamsMap,
					int N, int X, int Y,
					float likelyRatio, float certainRatio, float MSSE_LAMBDA,
					char optIters, float minimumResidual,
					int downSampledSize);
'''
RGFCLib.fitValueTensor.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                ctypes.c_float, ctypes.c_float, ctypes.c_float, 
                ctypes.c_int8, ctypes.c_float, ctypes.c_int32]

'''
void RobustAlgebraicPlaneFitting(float* x, float* y, float* z, float* mP, float* mP_Init,
							int N, float likelyRatio, float certainRatio,
							float MSSE_LAMBDA, char stretch2CornersOpt, 
							float minimumResidual, char optIters)
'''
RGFCLib.RobustAlgebraicPlaneFitting.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int, ctypes.c_float, ctypes.c_float, 
                ctypes.c_float, ctypes.c_int8, ctypes.c_float, ctypes.c_int8]
                
'''
void RSGImage(float* inImage, char* inMask, float *modelParamsMap,
				int winX, int winY,
				int X, int Y,
				float likelyRatio, float certainRatio,
				float MSSE_LAMBDA, char stretch2CornersOpt,
				char numModelParams, char optIters,
                float minimumResidual)
'''                
RGFCLib.RSGImage.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_int8, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int32, ctypes.c_int32, 
                ctypes.c_int32, ctypes.c_int32, 
                ctypes.c_float, ctypes.c_float, 
                ctypes.c_float, ctypes.c_int8, 
                ctypes.c_int8, ctypes.c_int8, ctypes.c_float]
       
'''
void RSGImage_by_Image_Tensor(float* inImage_Tensor, char* inMask_Tensor,
						float *model_mean, float *model_std,
						int winX, int winY,
						int N, int X, int Y,
						float likelyRatio, float certainRatio,
						float MSSE_LAMBDA, char stretch2CornersOpt,
						char numModelParams, char optIters,
                        float minimumResidual)
'''
RGFCLib.RSGImage_by_Image_Tensor.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_int8, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int32, ctypes.c_int32, 
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                ctypes.c_float, ctypes.c_float, 
                ctypes.c_float, ctypes.c_int8, 
                ctypes.c_int8, ctypes.c_int8, ctypes.c_float]

'''
void fitBackgroundRadially(float* inImage, char* inMask,
                           float* modelParamsMap, float* vecMP,
                           int minRes,
                           int maxRes,
                           int shellWidth,
                           int stride,
                           int X_Cent,
                           int Y_Cent,
                           char includeCenter,
                           int finiteSampleBias,
                           int X, int Y,
                           float likelyRatio, float certainRatio,
                           float MSSE_LAMBDA,
                           char optIters,
                           float minimumResidual);
'''

RGFCLib.fitBackgroundRadially.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_int8, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, 
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                ctypes.c_int8, ctypes.c_int32, 
                ctypes.c_int32, ctypes.c_int32, 
                ctypes.c_float, ctypes.c_float, 
                ctypes.c_float, ctypes.c_int8, ctypes.c_float]

'''
void fitBackgroundCylindrically(float* inTensor,
								char* inMask,
                                float* modelParamsMap,
								float* vecMP,
                                int minRes,
                                int maxRes,
                                int shellWidth,
                                char includeCenter,
                                int finiteSampleBias,
								int N,
                                int X,
								int Y,
                                float likelyRatio,
								float certainRatio,
                                float MSSE_LAMBDA,
                                char optIters,
						        float minimumResidual)
'''
RGFCLib.fitBackgroundCylindrically.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_int8, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, 
                ctypes.c_int8, ctypes.c_int32, 
                ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, 
                ctypes.c_float, ctypes.c_float, 
                ctypes.c_float, ctypes.c_int8, ctypes.c_float]
