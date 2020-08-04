//#################################################################################################
//# This file is part of RobustGaussianFittingLibrary, a free library WITHOUT ANY WARRANTY        # 
//# Copyright: 2019-2020 Deutsches Elektronen-Synchrotron                                         # 
//#################################################################################################

/*
A Robust Gaussian Fitting Library
	for ourlier detection and background subtraction

This is a MEX C file for MATLAB as the
	RGFLib.c external interface

If you'd like to use it, compile it throught the test file.	
	
An example is made in RGFLibTest.m

Currently we will write a mex for the following function(s):

		void RobustSingleGaussianVec(float *vec, 
					float *modelParams, float theta, 
					unsigned int N,
					float topKthPerc, float bottomKthPerc, 
					float MSSE_LAMBDA, unsigned char optIters)

Written by Alireza Sadri, arsadri@gmail.com 
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "RGFLib.c"

//////////////////////////// MEX ///////////////////////////
#include <matrix.h>
#include <mex.h>   
#ifndef MWSIZE_MAX
	typedef int mwSize;
	typedef int mwIndex;
	typedef int mwSignedIndex;
	#define MWSIZE_MAX    2147483647UL
	#define MWINDEX_MAX   2147483647UL
	#define MWSINDEX_MAX  2147483647L
	#define MWSINDEX_MIN -2147483647L
	#define MWSIZE_MIN    0UL
	#define MWINDEX_MIN   0UL
#endif
/////////////////////////////// END OF MEX ////////////////////////

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
	unsigned int i;
	double *inVec;
	double *inWeights;
	double *initModel;
	double *N;
	double *topKthPerc;
	double *bottomKthPerc;
	double *MSSE_LAMBDA;
	double *optIters;

	//associate inputs
	inVec 			= mxGetPr(prhs[0]);
	inWeights	    = mxGetPr(prhs[1]);
	initModel		= mxGetPr(prhs[2]);
	N				= mxGetPr(prhs[3]);
	topKthPerc		= mxGetPr(prhs[4]);
	bottomKthPerc	= mxGetPr(prhs[5]);
	MSSE_LAMBDA		= mxGetPr(prhs[6]);
	optIters		= mxGetPr(prhs[7]);
	
	float *_inVec;
	float *_inWeights;
	float _initModel;
	unsigned int _N;
	float _topKthPerc;
	float _bottomKthPerc;
	float _MSSE_LAMBDA;
	unsigned char _optIters;
		
	_initModel = (float)(initModel[0]);
	_N = (unsigned int)(N[0]);
	_topKthPerc = (float)(topKthPerc[0]);
	_bottomKthPerc = (float)(bottomKthPerc[0]);
	_MSSE_LAMBDA = (float)(MSSE_LAMBDA[0]);
	_optIters = (unsigned char)(optIters[0]);
	
	_inVec=(float *) malloc( _N*sizeof(float));
	for(i=0; i<_N; i++)
		_inVec[i] = (float)inVec[i];

	_inWeights=(float *) malloc( _N*sizeof(float));
	for(i=0; i<_N; i++)
		_inWeights[i] = (float)inWeights[i];

	float _mP[2];
	_mP[0] = 0;
	_mP[1] = 0;
	
	fitValue2Skewed(_inVec, _inWeights, _mP, _initModel, _N,
					   _topKthPerc, _bottomKthPerc,
					   _MSSE_LAMBDA, _optIters);
	free(_inVec);
	free(_inWeights);
	
	plhs[0] = mxCreateDoubleScalar(1);
	*mxGetPr(plhs[0]) = _mP[0];

}
