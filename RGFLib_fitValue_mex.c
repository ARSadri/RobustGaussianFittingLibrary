/*
 * A Robust Gaussian Fitting Library
 *      for ourlier detection and background subtraction
 *
 * This is a MEX C file for MATLAB as the
 *      RGFLib.c external interface
 *
 * An example is made in RGFLibTest.m
 *
 * Currently we will write a mex for the following function(s):
 *
 * 		void RobustSingleGaussianVec(float *vec, 
 *									float *modelParams, float theta, 
 *									unsigned int N,
 *									float topKthPerc, float bottomKthPerc, 
 *									float MSSE_LAMBDA, unsigned char optIters)
 *
 * Written by Alireza Sadri, arsadri@gmail.com 
*/

//////////////////////////// MEX ///////////////////////////
#include <matrix.h>
#include <mex.h>   
/* Definitions to keep compatibility with earlier versions of ML */
#ifndef MWSIZE_MAX
	typedef int mwSize;
	typedef int mwIndex;
	typedef int mwSignedIndex;
	# define MWSIZE_MAX    2147483647UL
	# define MWINDEX_MAX   2147483647UL
	# define MWSINDEX_MAX  2147483647L
	# define MWSINDEX_MIN -2147483647L
	#define MWSIZE_MIN    0UL
	#define MWINDEX_MIN   0UL
#endif
/////////////////////////////// END OF MEX ////////////////////////

#include "RPFLib.c"

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{

	float *inVec;
	float *initModel;
	unsigned int *N;
	float *topKthPerc;
	float *bottomKthPerc;
	float *MSSE_LAMBDA;
    unsigned char *optIters;

	//associate inputs
	inVec 			= mxGetPr(prhs[0]);
	initModel		= mxGetPr(prhs[1]);
	N				= mxGetPr(prhs[2]);
	topKthPerc		= mxGetPr(prhs[3]);
	bottomKthPerc	= mxGetPr(prhs[4]);
	MSSE_LAMBDA		= mxGetPr(prhs[5]);
	optIters		= mxGetPr(prhs[6]);
    
	float mP[2];
	mP[0]=0;
	mP[1]=0;
	
	RobustSingleGaussianVec(inVec, mP, initModel, N,
							topKthPerc, bottomKthPerc, 
							MSSE_LAMBDA, optIters);

	plhs[0] = mxCreatefloatScalar(1);
	*mxGetPr(plhs[0]) = mP[0];	
	plhs[1] = mxCreatefloatScalar(1);
	*mxGetPr(plhs[1]) = mP[1];	
	
	return;
}