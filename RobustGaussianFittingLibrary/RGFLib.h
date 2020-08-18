//#################################################################################################
//# This file is part of RobustGaussianFittingLibrary, a free library WITHOUT ANY WARRANTY        #
//# Copyright: 2017-2020 LaTrobe University Melbourne, 2019-2020 Deutsches Elektronen-Synchrotron #
//#################################################################################################

#ifndef RGFLIB_H
#define RGFLIB_H

#include <math.h>
#include <stdlib.h>

void islandRemoval(unsigned char* inMask, unsigned char* outMask,
 	 			   unsigned int X, unsigned int Y,
				   unsigned int islandSizeThreshold);

float MSSE(float *error, unsigned int vecLen, float MSSE_LAMBDA, unsigned int k);

void RobustSingleGaussianVec(float *vec, float *modelParams, float theta, unsigned int N,
		float topkPerc, float botkPerc, float MSSE_LAMBDA, unsigned char optIters);

void fitValue2Skewed(float *vec, float *weights,
					float *modelParams, float theta, unsigned int N,
					float topkPerc, float botkPerc,
					float MSSE_LAMBDA, unsigned char optIters);

void RobustAlgebraicLineFitting(float* x, float* y, float* mP,
							unsigned int N, float topkPerc, float botkPerc, float MSSE_LAMBDA);

void RobustAlgebraicLineFittingTensor(float *inTensorX, float *inTensorY,
									float *modelParamsMap, unsigned int N,
									unsigned int X, unsigned int Y,
									float topkPerc, float botkPerc,
									float MSSE_LAMBDA);

void RobustAlgebraicPlaneFitting(float* x, float* y, float* z, float* mP,
							unsigned int N, float topkPerc, float botkPerc,
							float MSSE_LAMBDA, unsigned char stretch2CornersOpt);

void RobustSingleGaussianTensor(float *inTensor, float *modelParamsMap, unsigned int N,
		unsigned int X, unsigned int Y, float topkPerc, float botkPerc, float MSSE_LAMBDA);

void RSGImage(float* inImage, unsigned char* inMask, float *modelParamsMap,
				unsigned int winX, unsigned int winY,
				unsigned int X, unsigned int Y,
				float topkPerc, float botkPerc,
				float MSSE_LAMBDA, unsigned char stretch2CornersOpt,
				unsigned char numModelParams, unsigned char optIters);

void RSGImage_by_Image_Tensor(float* inImage_Tensor, unsigned char* inMask_Tensor,
						float *model_mean, float *model_std,
						unsigned int winX, unsigned int winY,
						unsigned int N, unsigned int X, unsigned int Y,
						float topkPerc, float botkPerc,
						float MSSE_LAMBDA, unsigned char stretch2CornersOpt,
						unsigned char numModelParams, unsigned char optIters);

#endif