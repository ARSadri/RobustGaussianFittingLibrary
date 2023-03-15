/*
------------------------------------------------------
This file is part of RobustGaussianFittingLibrary,
a free library WITHOUT ANY WARRANTY
Copyright: 2017-2020 LaTrobe University Melbourne,
           2019-2020 Deutsches Elektronen-Synchrotron
------------------------------------------------------
*/

#ifndef RGFLIB_H
#define RGFLIB_H

#include <math.h>
#include <stdlib.h>

#define NEGATIVE_MAX -(1e+20)

void islandRemoval(char* inMask, char* outMask,
 	 			   int X, int Y,
				   int islandSizeThreshold);

float MSSE(float *error, int vecLen, float MSSE_LAMBDA,
		   int k, float minimumResidual);
float MSSEWeighted(float* error, float* weights, int vecLen,
                   float MSSE_LAMBDA, int k, float minimumResidual);
	
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

void medianOfFits(float *vec, float *weights, 
        float *modelParams, float theta, int N,
        float likelyRatio_min, float likelyRatio_max,
		  int numSamples, float sampleRatio,
        float MSSE_LAMBDA, char optIters,
		  float minimumResidual,
		  int downSampledSize);

void RobustAlgebraicLineFitting(float* x, float* y, float* mP,
							int N, float likelyRatio,
							float certainRatio, float MSSE_LAMBDA);

void RobustAlgebraicLineFittingTensor(float *inTensorX, float *inTensorY,
									float *modelParamsMap, int N,
									int X, int Y,
									float likelyRatio, float certainRatio,
									float MSSE_LAMBDA);

void RobustAlgebraicPlaneFitting(float* x, float* y, float* z,
								 float* mP, float* mP_Init,
							     int N, float likelyRatio,
								 float certainRatio,
								 float MSSE_LAMBDA,
								 char stretch2CornersOpt,
							     float minimumResidual,
								 char optIters);

void fitValueTensor(float* inTensor, float* inWeights, float* modelParamsMap,
		int N, int X, int Y,
		float likelyRatio, float certainRatio, float MSSE_LAMBDA,
		char optIters, float minimumResidual,
		int downSampledSize);

void RSGImage(float* inImage, char* inMask, float *modelParamsMap,
		      int winX, int winY,
		      int X, int Y,
		      float likelyRatio, float certainRatio,
		      float MSSE_LAMBDA, char stretch2CornersOpt,
		      char numModelParams, char optIters,
		      float minimumResidual);

void RSGImage_by_Image_Tensor(float* inImage_Tensor,
							  char* inMask_Tensor,
						      float *model_mean, float *model_std,
						      int winX, int winY,
						      int N, int X, int Y,
						      float likelyRatio, float certainRatio,
						      float MSSE_LAMBDA,
							  char stretch2CornersOpt,
						      char numModelParams,
							  char optIters,
						      float minimumResidual);

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
						        float minimumResidual);

#endif
