import numpy as np
import ctypes
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
RobustGausFitCLib = ctypes.cdll.LoadLibrary(dir_path + '/RobustGausFitLib.so')


'''
void islandRemoval(unsigned char* inMask, unsigned char* outMask, 
					  unsigned int X, unsigned int Y, 
					  unsigned int islandSizeThreshold)
'''
RobustGausFitCLib.islandRemoval.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags='C_CONTIGUOUS'),
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
  
'''
void indexCheck(float* inTensor, float* targetLoc, unsigned int X, unsigned int Y, unsigned int Z)
'''
RobustGausFitCLib.indexCheck.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int, ctypes.c_int, ctypes.c_float]

'''
float MSSE(float *error, unsigned int vecLen, float MSSE_LAMBDA, unsigned int k)
'''
RobustGausFitCLib.MSSE.restype = ctypes.c_float
RobustGausFitCLib.MSSE.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int, ctypes.c_float, ctypes.c_int ]

'''
void RobustSingleGaussianVec(float *vec, float *modelParams, float theta, unsigned int N,
		float topKthPerc, float bottomKthPerc, float MSSE_LAMBDA, unsigned char optIters)
'''
RobustGausFitCLib.RobustSingleGaussianVec.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_float, ctypes.c_int, ctypes.c_float, 
                ctypes.c_float, ctypes.c_float, ctypes.c_uint8]

'''
void RobustAlgebraicLineFitting(float* x, float* y, float* mP, unsigned int N,
							  float topKthPerc, float bottomKthPerc, float MSSE_LAMBDA)
'''                
RobustGausFitCLib.RobustAlgebraicLineFitting.argtypes = [
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
RobustGausFitCLib.RobustAlgebraicLineFittingTensor.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                ctypes.c_float, ctypes.c_float, ctypes.c_float]                
                
'''
void RobustSingleGaussianTensor(float *inTensor, float *modelParamsMap,
    unsigned int N, unsigned int X,
    unsigned int Y, float topKthPerc, float bottomKthPerc, float MSSE_LAMBDA))
'''
RobustGausFitCLib.RobustSingleGaussianTensor.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float, ctypes.c_float]

'''
void RobustAlgebraicPlaneFitting(float* x, float* y, float* z, float* mP,
							unsigned int N, float topKthPerc, float bottomKthPerc, 
							float MSSE_LAMBDA, unsigned char stretch2CornersOpt)
'''                            
RobustGausFitCLib.RobustAlgebraicPlaneFitting.argtypes = [
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
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
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
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
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_uint8, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                ctypes.c_uint32, ctypes.c_uint32, 
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                ctypes.c_float, ctypes.c_float, 
                ctypes.c_float, ctypes.c_uint8, 
                ctypes.c_uint8, ctypes.c_uint8]
