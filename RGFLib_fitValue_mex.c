/*
A Robust Gaussian Fitting Library
	for ourlier detection and background subtraction

This is a MEX C file for MATLAB as the
	RGFLib.c external interface

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
	double *initModel;
	double *N;
	double *topKthPerc;
	double *bottomKthPerc;
	double *MSSE_LAMBDA;
	double *optIters;

	//associate inputs
	inVec 			= mxGetPr(prhs[0]);
	initModel		= mxGetPr(prhs[1]);
	N				= mxGetPr(prhs[2]);
	topKthPerc		= mxGetPr(prhs[3]);
	bottomKthPerc	= mxGetPr(prhs[4]);
	MSSE_LAMBDA		= mxGetPr(prhs[5]);
	optIters		= mxGetPr(prhs[6]);
    
	float *_inVec;
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

	float _mP[2];
	_mP[0] = 0;
	_mP[1] = 0;
	
	RobustSingleGaussianVec(_inVec, _mP, _initModel, _N,
							_topKthPerc, _bottomKthPerc, 
							_MSSE_LAMBDA, _optIters);

	free(_inVec);
	plhs[0] = mxCreateDoubleScalar(1);
	*mxGetPr(plhs[0]) = _mP[0];	
	plhs[1] = mxCreateDoubleScalar(1);
	*mxGetPr(plhs[1]) = _mP[1];	
	
	return;
}
