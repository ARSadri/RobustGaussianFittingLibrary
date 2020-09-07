//#################################################################################################
//# This file is part of RobustGaussianFittingLibrary, a free library WITHOUT ANY WARRANTY        # 
//# Copyright: 2017-2020 LaTrobe University Melbourne, 2019-2020 Deutsches Elektronen-Synchrotron # 
//#################################################################################################
// gcc -fPIC -shared -o RobustGausFitLib.c RobustGausFitLib.so

#include "RGFLib.h"

void dfs(unsigned char* inMask, unsigned char* mask, 
			 int x,  int y, unsigned int X, unsigned int Y,
			unsigned int label, unsigned int* islandSize, 
			unsigned int* rowsToMask, unsigned int* clmsToMask){
				
	unsigned int nx, ny, i;
	//int dx[8] = {-1, 0, 1, 1, 1, 0,-1,-1};
	//int dy[8] = { 1, 1, 1, 0,-1,-1,-1, 0};
	int dx[4] = {    0,    1,    0,   -1};
	int dy[4] = {    1,    0,   -1,    0};

    mask[y + x*Y] = label;
	rowsToMask[islandSize[0]] = x;
	clmsToMask[islandSize[0]] = y;
	islandSize[0]++;
    for(i=0; i<4;i++){
        nx = x+dx[i];
		ny = y+dy[i];
		if((nx<0) || (nx>=X) || (ny<0) || (ny>=Y))
			continue;
        if( (inMask[ny + nx*Y]>0) && (mask[ny + nx*Y]==0) )
			dfs(inMask, mask, nx,ny, X, Y, label, islandSize, rowsToMask, clmsToMask);
    }
}

void islandRemoval(unsigned char* inMask, unsigned char* outMask, 
 	 			   unsigned int X, unsigned int Y,
				   unsigned int islandSizeThreshold) {
    unsigned int label = 1;
	int i, j, cnt;
	unsigned int islandSize[1];
	
    unsigned int* rowsToMask;
	unsigned int* clmsToMask;
	unsigned char* mask;
	
	mask = (unsigned char*) malloc(X*Y * sizeof(unsigned char));
	rowsToMask = (unsigned int*) malloc(X*Y * sizeof(unsigned int));
	clmsToMask = (unsigned int*) malloc(X*Y * sizeof(unsigned int));
	
	for(i=0; i< X; i++) {
        for(j=0; j < Y; j++) {
			mask[j + i*Y] = 0;
        }
	}
	
    for(i=0; i< X;i++) {
        for(j=0; j<Y; j++) {
            if((inMask[j + i*Y]>0) && (mask[j + i*Y]==0) ) {
				islandSize[0] = 0;
                dfs(inMask, mask, i, j, X, Y, label, islandSize, rowsToMask, clmsToMask);
				
				if(islandSize[0] <= islandSizeThreshold)
					for(cnt=0; cnt< islandSize[0]; cnt++)
							outMask[clmsToMask[cnt] + rowsToMask[cnt]*Y] = 1;
				label++;
			}
        }
    }
			
	free(rowsToMask);
	free(clmsToMask);
	free(mask);
}

void indexCheck(float* inTensor, float* targetLoc,
		unsigned int X, unsigned int Y, float Z) {
	unsigned int rCnt, cCnt;
	for(rCnt = 0; rCnt < X; rCnt++)
		for(cCnt = 0; cCnt < Y; cCnt++)
			if(inTensor[cCnt + rCnt*Y]==Z) {
				targetLoc[0] = rCnt;
				targetLoc[1] = cCnt;
			}
}

struct sortStruct {
    float vecData;
    unsigned int indxs;
};

int _partition( struct sortStruct dataVec[], int l, int r) {
   float pivot;
   int i, j;
   struct sortStruct t;

   pivot = dataVec[l].vecData;
   i = l; j = r+1;
   while(1)	{
		do ++i; while( dataVec[i].vecData <= pivot && i <= r );
		do --j; while( dataVec[j].vecData > pivot );
		if( i >= j ) break;
		t = dataVec[i];
		dataVec[i] = dataVec[j];
		dataVec[j] = t;
   }
   t = dataVec[l];
   dataVec[l] = dataVec[j];
   dataVec[j] = t;
   return j;
}

void quickSort( struct sortStruct dataVec[], int l, int r) {
   int j;
   if( l < r ) {
		j = _partition( dataVec, l, r);
		quickSort( dataVec, l, j-1);
		quickSort( dataVec, j+1, r);
   }
}

float MSSE(float* error, unsigned int vecLen, float MSSE_LAMBDA, unsigned int k) {
	unsigned int i;
	float estScale, cumulative_sum, cumulative_sum_perv;
	struct sortStruct* sortedSqError;

	if((vecLen < k) || (vecLen<=0))
		return(-1);
	
	sortedSqError = (struct sortStruct*) malloc(vecLen * sizeof(struct sortStruct));
	for (i = 0; i < vecLen; i++) {
		sortedSqError[i].vecData  = error[i]*error[i];
		sortedSqError[i].indxs = i;
	}
	quickSort(sortedSqError,0,vecLen-1);

	cumulative_sum = 0;
	for (i = 0; i < k; i++)	//Note: finite sample bias of MSSE [RezaJMIJV'06]
		cumulative_sum += sortedSqError[i].vecData;
	cumulative_sum_perv = cumulative_sum;
	for (i = k; i < vecLen; i++) {
		// in (i-1), the 1 is the dimension of model
		if ( MSSE_LAMBDA*MSSE_LAMBDA * cumulative_sum < (i-1)*sortedSqError[i].vecData )		
			break;
		cumulative_sum_perv = cumulative_sum;
		cumulative_sum += sortedSqError[i].vecData ;
	}
	estScale = fabs(sqrt(cumulative_sum_perv / ((float)i - 1)));
	free(sortedSqError);
	return estScale;
}

void RobustSingleGaussianVec(float* vec, float* modelParams, float theta, unsigned int N,
		float topkPerc, float botkPerc, float MSSE_LAMBDA, unsigned char optIters) {

	float avg, tmp, estScale;
	unsigned int i, iter;

	float* residual;

	struct sortStruct* errorVec;
	errorVec = (struct sortStruct*) malloc(N * sizeof(struct sortStruct));
	
	if(N==0) {
		modelParams[0] = 0;
		modelParams[1] = 0;
		return;
	}
	else if(N==1) {
		modelParams[0] = vec[0];
		modelParams[1] = 0;
		return;
	}
	else if(N==2) {
		optIters = 0;
	}
	
	if(N<12)
		botkPerc = 0;
	
	if(optIters>0) {
		for (i = 0; i < N; i++) {
			errorVec[i].vecData = vec[i];
			errorVec[i].indxs = i;
		}
		quickSort(errorVec,0,N-1);
		theta = errorVec[(int)(N*topkPerc)].vecData;

		for(iter=0; iter<optIters; iter++) {
			for (i = 0; i < N; i++) {
				errorVec[i].vecData  = fabs(vec[i] - theta);
				errorVec[i].indxs = i;
			}
			quickSort(errorVec,0,N-1);
			theta = 0;
			tmp = 0;
			for(i=(int)(N*botkPerc); i<(int)(N*topkPerc); i++) {
				theta += vec[errorVec[i].indxs];
				tmp++;
			}
			theta = theta / tmp;
		}
	
	
		avg = 0;
		for(i=0; i<(int)(N*topkPerc); i++)
			avg += vec[errorVec[i].indxs];
		avg = avg / (int)(N*topkPerc);

		residual = (float*) malloc(N * sizeof(float));
		for (i = 0; i < N; i++)
			residual[i]  = vec[i] - avg;
		estScale = MSSE(residual, N, MSSE_LAMBDA, (int)(N*topkPerc));
		free(residual);
	}
	else {
		avg = 0;
		for(i=0; i<N; i++)
			avg += vec[i];
		avg = avg / N;
		estScale = 0;
		for(i=0; i<N; i++)
			estScale += (vec[i]-avg)*(vec[i]-avg);
		estScale = sqrt(estScale/N);
	}
	modelParams[0] = avg;
	modelParams[1] = estScale;

	free(errorVec);
}

float MSSEWeighted(float* error, float* weights, unsigned int vecLen, float MSSE_LAMBDA, unsigned int k) {
	unsigned int i, q;
	float estScale, cumulative_sum, tmp;
	struct sortStruct* sortedSqError;

	if((vecLen < k) || (vecLen<=0))
		return(-1);

	sortedSqError = (struct sortStruct*) malloc(vecLen * sizeof(struct sortStruct));

	for (i = 0; i < vecLen; i++) {
		sortedSqError[i].vecData  = error[i]*error[i];
		sortedSqError[i].indxs = i;
	}
	quickSort(sortedSqError,0,vecLen-1);

	cumulative_sum = 0;
	for (i = 0; i < k; i++)	//finite sample bias of MSSE [RezaJMIJV'06]
		cumulative_sum += sortedSqError[i].vecData;
	for (i = k; i < vecLen; i++) {
		if ( MSSE_LAMBDA*MSSE_LAMBDA * cumulative_sum < (i-1)*sortedSqError[i].vecData )		// in (i-1), the 1 is the dimension of model
			break;
		cumulative_sum += sortedSqError[i].vecData ;
	}

	estScale = 0;
	tmp = 0;
	for(q=0; q<i; q++) {
		estScale += (weights[sortedSqError[q].indxs])*sortedSqError[q].vecData;
		tmp += weights[sortedSqError[q].indxs];
	}
	estScale = sqrt((i/(float)(i-1))*estScale / tmp);

	free(sortedSqError);
	return estScale;
}

void fitValue2Skewed(float* vec, float* weights, 
					float* modelParams, float theta, unsigned int N,
					float topkPerc, float botkPerc,
					float MSSE_LAMBDA, unsigned char optIters) {

	float tmp, tmpH, tmpL, estScale, theta_new;
	float upperScale, lowerScale, errAtTopk;
	int i, topk, botk, iter, numPointsSide;
		
	topk = (int)(N*topkPerc);
	botk = (int)(N*botkPerc);
	
	if(N==0) {
		modelParams[0] = 0;
		modelParams[1] = 0;
		return;
	}
	else if(N==1) {
		modelParams[0] = vec[0];
		modelParams[1] = 0;
		return;
	}
	else if(N==2) {
		optIters = 0;
	}
	else if(N<12) {
		if(topk<N/2)
			topk = (int)(N/2)+1;
		botk = 0;
	}
	if(optIters>0) {
		float *residual;
		residual = (float*) malloc(N * sizeof(float));
		
		struct sortStruct* errorVec;
		errorVec = (struct sortStruct*) malloc(N * sizeof(struct sortStruct));
		for(iter=0; iter<optIters; iter++) {
			
			tmpH = 0;
			tmpL = 0;
			if(iter>=optIters/2) {
				// lets do a symmetric fitting, half of the sample data points must come from either side
				for (i = 0; i < N; i++) {
					errorVec[i].vecData = fabs(vec[i] - theta);
					errorVec[i].indxs = i;
				}
				quickSort(errorVec,0,N-1);
				errAtTopk = errorVec[topk-1].vecData;

				//////////////////////////////////////////////////////////////////
				theta_new = 0;
				numPointsSide = 0;
				for (i = 0; i < N; i++) {
					if((vec[i] >= theta) && (vec[i] <=  theta + errAtTopk)) {
						errorVec[i].vecData  = vec[i] - theta;
						numPointsSide++;
					}
					else {
						errorVec[i].vecData  = 1e+9;
					}
					errorVec[i].indxs = i;
				}
				if((int)(numPointsSide*topkPerc)>0) {
					quickSort(errorVec,0,N-1);
					for(i=(int)(numPointsSide*botkPerc); i<(int)(numPointsSide*topkPerc); i++) {
						theta_new += weights[errorVec[i].indxs]*vec[errorVec[i].indxs];
						tmpH += weights[errorVec[i].indxs];
					}
				}
				numPointsSide = 0;
				for (i = 0; i < N; i++) {
					if((vec[i] < theta) && (vec[i] >= theta - errAtTopk) ) {
						errorVec[i].vecData  = theta - vec[i];
						numPointsSide++;
					}
					else {
						errorVec[i].vecData  = 1e+9;
					}
					errorVec[i].indxs = i;
				}
				if((int)(numPointsSide*topkPerc)>0) {
					quickSort(errorVec,0,N-1);
					for(i=(int)(numPointsSide*botkPerc); i<(int)(numPointsSide*topkPerc); i++) {
						theta_new += weights[errorVec[i].indxs]*vec[errorVec[i].indxs];
						tmpL += weights[errorVec[i].indxs];
					}
				}
			}
			tmp = tmpL + tmpH;

			//////////////////////////////////////////////////////////////////
			if((tmpL==0) || (tmpH==0)) {
				theta_new = 0;
				tmp = 0;
				for (i = 0; i < N; i++) {
					errorVec[i].vecData = fabs(vec[i] - theta);
					errorVec[i].indxs = i;
				}
				quickSort(errorVec,0,N-1);
				theta = 0;
				tmp = 0;
				for(i=botk; i<topk; i++) {
					theta_new += weights[errorVec[i].indxs]*vec[errorVec[i].indxs];
					tmp += weights[errorVec[i].indxs];
				}
			}
			if(tmp==0) {
				theta_new = 0;
				tmp = 0;
				for(i=botk; i<topk; i++) {
					theta_new += vec[errorVec[i].indxs];
					tmp += 1;
				}
			}
			theta = theta_new / tmp;
		}
		
		////////////////////////////////////////////////////////////////
		/////////// calculate the mean of symmetric inliers  ///////////
		////////////////////////////////////////////////////////////////
		for (i = 0; i < N; i++)
			residual[i]  = fabs(theta - vec[i]);
		estScale = MSSEWeighted(residual, weights, N, MSSE_LAMBDA, topk);
		free(residual);
		modelParams[0] = theta;
		modelParams[1] = estScale;


		////////////////////////////////////////////////////////////////
		//////////////// calculate the std by inliers  /////////////////
		////////////////////////////////////////////////////////////////
		upperScale = fabs((theta + 3.0*estScale) - modelParams[0])/3.0;
		lowerScale = fabs(modelParams[0] - (theta - 3.0*estScale))/3.0;
		if(lowerScale<upperScale)
			modelParams[1] = upperScale;	//hopefully never negative
		else
			modelParams[1] = lowerScale;
		
		free(errorVec);
	}
	else {
		theta = 0;
		for(i=0; i<N; i++)
			theta += vec[i];
		theta = theta / N;
		estScale = 0;
		for(i=0; i<N; i++)
			estScale += (vec[i]-theta)*(vec[i]-theta);
		estScale = sqrt(estScale/N);
		modelParams[0] = theta;
		modelParams[1] = estScale;
	}

}

void TLS_AlgebraicLineFitting(float* x, float* y, float* mP, unsigned int N) {
	unsigned int i;
	float xsum,x2sum,ysum,xysum, D; 
	xsum=0;
	x2sum=0;
	ysum=0;
	xysum=0;
    for (i=0;i<N;i++) {
         xsum += x[i];
         ysum += y[i];
        x2sum += x[i]*x[i];
        xysum += x[i]*y[i];
    }
	D = (N*x2sum-xsum*xsum);
	if(fabs(D)>0.0000001) {
		mP[0]=(N*xysum-xsum*ysum)/D;
		mP[1]=(x2sum*ysum-xsum*xysum)/D;
	}
	else {
		mP[0] = 0;
		mP[1] = 0;
	}
}

void lineAlgebraicModelEval(float* x, float* y_fit, float* mP, unsigned int N) {
    unsigned int i;
	for (i=0;i<N;i++)
        y_fit[i]=mP[0]*x[i]+mP[1];
}

void RobustAlgebraicLineFitting(float* x, float* y, float* mP,
							unsigned int N, float topkPerc, float botkPerc, float MSSE_LAMBDA) {
	float model[2];
	unsigned int i, iter, cnt;
	unsigned int sampleSize;
	float *residual;
	float* sample_x;
	float* sample_y;
	struct sortStruct* errorVec;
	errorVec = (struct sortStruct*) malloc(N * sizeof(struct sortStruct));

	sampleSize = (unsigned int)(N*topkPerc)- (unsigned int)(N*botkPerc);

	sample_x = (float*) malloc(sampleSize * sizeof(float));
	sample_y = (float*) malloc(sampleSize * sizeof(float));

	model[0]=0;
	model[1]=0;
	for(iter=0; iter<12; iter++) {
		
		for (i = 0; i < N; i++) {
			errorVec[i].vecData  = fabs(y[i] - (model[0]*x[i] + model[1]));	//could have called the function
			errorVec[i].indxs = i;
		}
		quickSort(errorVec,0,N-1);
		
		cnt = 0;
		for(i=(int)(N*botkPerc); i<(int)(N*topkPerc); i++) {
			sample_x[cnt] = x[errorVec[i].indxs];
			sample_y[cnt] = y[errorVec[i].indxs];
			cnt++;
		}
		TLS_AlgebraicLineFitting(sample_x, sample_y, model, sampleSize);
	}

	mP[0] = model[0];
	mP[1] = model[1];
	residual = (float*) malloc(N * sizeof(float));
	for (i = 0; i < N; i++)
		residual[i]  = y[i] - (model[0]*x[i] + model[1]) + ((float) rand() / (RAND_MAX))/4;	
		//Noise stabilizes MSSE
	mP[2] = MSSE(residual, N, MSSE_LAMBDA, (int)(N*topkPerc));

	free(sample_x);
	free(sample_y);
	free(residual);
	free(errorVec);
}

void RobustAlgebraicLineFittingTensor(float* inTensorX, float* inTensorY,
									float* modelParamsMap, unsigned int N,
									unsigned int X, unsigned int Y, 
									float topkPerc, float botkPerc,
									float MSSE_LAMBDA) {

	float* xVals;
	xVals = (float*) malloc(N * sizeof(float));
	float* yVals;
	yVals = (float*) malloc(N * sizeof(float));
	
	float mP[3];
	unsigned int rCnt, cCnt, i;

	for(rCnt=0; rCnt<X; rCnt++) {
		for(cCnt=0; cCnt<Y; cCnt++) {
			
			for(i=0; i<N; i++) {
				xVals[i]=inTensorX[cCnt + rCnt*Y + i*X*Y];
				yVals[i]=inTensorY[cCnt + rCnt*Y + i*X*Y];
			}
			
			RobustAlgebraicLineFitting(xVals, yVals, mP, N, topkPerc, botkPerc, MSSE_LAMBDA);
			modelParamsMap[cCnt + rCnt*Y + 0*X*Y] = mP[0];
			modelParamsMap[cCnt + rCnt*Y + 1*X*Y] = mP[1];
			modelParamsMap[cCnt + rCnt*Y + 2*X*Y] = mP[2];
		}
	}
	free(xVals);
	free(yVals);
}

void TLS_AlgebraicPlaneFitting(float* x, float* y, float* z, float* mP, unsigned int N) {
	unsigned int i;
	float x_mean, y_mean, z_mean, x_x_sum, y_y_sum, x_y_sum, x_z_sum, y_z_sum, a,b,c, D; 
	x_mean = 0;
	y_mean = 0;
	z_mean = 0;
	x_x_sum = 0;
	y_y_sum = 0;
	x_y_sum = 0;
	x_z_sum = 0;
	y_z_sum = 0;
    for (i=0;i<N;i++) {
		x_mean += x[i];
		y_mean += y[i];
		z_mean += z[i];
	}		  
	x_mean = x_mean/N;
	y_mean = y_mean/N;
	z_mean = z_mean/N;
	
    for (i=0;i<N;i++) {
		x_x_sum += (x[i] - x_mean) * (x[i] - x_mean);
		y_y_sum += (y[i] - y_mean) * (y[i] - y_mean);
		x_y_sum += (x[i] - x_mean) * (y[i] - y_mean);
		x_z_sum += (x[i] - x_mean) * (z[i] - z_mean);
		y_z_sum += (y[i] - y_mean) * (z[i] - z_mean);
    }
	D = x_x_sum*y_y_sum - x_y_sum * x_y_sum;
	if(fabs(D)>0.00001) {
		a = (y_y_sum * x_z_sum - x_y_sum * y_z_sum)/D;
		b = (x_x_sum * y_z_sum - x_y_sum * x_z_sum)/D;
	}
	else {
		a = 0;
		b = 0;
	}
    c = z_mean - a*x_mean - b*y_mean;
	mP[0] = a;
	mP[1] = b;
	mP[2] = c;
}

int stretch2CornersFunc(float* x, float* y, unsigned int N, 
						unsigned char stretch2CornersOpt, 
						unsigned int* sample_inds, unsigned int sampleSize) {
	return 0;
}

void RobustAlgebraicPlaneFitting(float* x, float* y, float* z, float* mP,
							unsigned int N, float topkPerc, float botkPerc,
							float MSSE_LAMBDA, unsigned char stretch2CornersOpt) {
	float model[3];
	unsigned int i, iter, cnt;
	unsigned int sampleSize;
	unsigned isStretchingPossible=0;
	float* residual;
	float* sample_x;
	float* sample_y;
	float* sample_z;
	unsigned int* sample_inds;
	struct sortStruct* errorVec;
	float* sortedX;
	float* sortedY;

	errorVec = (struct sortStruct*) malloc(N * sizeof(struct sortStruct));
	
	sortedX = (float*) malloc(N * sizeof(float));
	sortedY = (float*) malloc(N * sizeof(float));
	residual = (float*) malloc(N * sizeof(float));
	
	sampleSize = (unsigned int)(N*topkPerc)- (unsigned int)(N*botkPerc);
	sample_x = (float*) malloc(sampleSize * sizeof(float));
	sample_y = (float*) malloc(sampleSize * sizeof(float));
	sample_z = (float*) malloc(sampleSize * sizeof(float));
	sample_inds = (unsigned int*) malloc(sampleSize * sizeof(unsigned int));

	cnt = 0;
	for(i=(unsigned int)(N*botkPerc); i<(unsigned int)(N*topkPerc); i++)
		sample_inds[cnt++] = i;
	
	model[0] = 0;
	model[1] = 0;
	model[2] = 0;
	for(iter=0; iter<12; iter++) {
		
		for (i = 0; i < N; i++) {
			errorVec[i].vecData  = fabs(z[i] - (model[0]*x[i] + model[1]*y[i] + model[2]));	
			errorVec[i].indxs = i;
		}
		quickSort(errorVec,0,N-1);
		
		
		if(stretch2CornersOpt>0) {
			for(i=0; i<(unsigned int)(N*topkPerc); i++) {
				sortedX[i] = x[errorVec[i].indxs];
				sortedY[i] = y[errorVec[i].indxs];
			}
			isStretchingPossible = stretch2CornersFunc(sortedX, sortedY, 
													(unsigned int)(N*topkPerc),
													stretch2CornersOpt, 
													sample_inds, 
													sampleSize);
			if(isStretchingPossible==0) {
				cnt = 0;
				for(i=(unsigned int)(N*botkPerc); i<(unsigned int)(N*topkPerc); i++)
					sample_inds[cnt++] = i;
			}
		}

		for(i=0; i<sampleSize; i++) {
			sample_x[i] = x[errorVec[sample_inds[i]].indxs];
			sample_y[i] = y[errorVec[sample_inds[i]].indxs];
			sample_z[i] = z[errorVec[sample_inds[i]].indxs];
		}
		TLS_AlgebraicPlaneFitting(sample_x, sample_y, sample_z, model, sampleSize);
	}

	mP[0] = model[0];
	mP[1] = model[1];
	mP[2] = model[2];
	
	for (i = 0; i < N; i++)
		residual[i]  = z[i] - (model[0]*x[i] + model[1]*y[i] + model[2]);
	
	mP[3] = MSSE(residual, N, MSSE_LAMBDA, (int)(N*topkPerc));

	free(residual);
	free(errorVec);
	free(sortedX);
	free(sortedY);
	free(sample_inds);
	free(sample_x);
	free(sample_y);
	free(sample_z);
}

void RobustSingleGaussianTensor(float* inTensor, unsigned char* inMask,
				float* modelParamsMap, unsigned int N, unsigned int X, unsigned int Y,
				float topkPerc, float botkPerc, float MSSE_LAMBDA, unsigned char optIters) {

	float* vec;
	float* weights;

	float mP[2];
	unsigned int rCnt, cCnt, i, L;

	vec = (float*) malloc(N * sizeof(float));
	weights = (float*) malloc(N * sizeof(float));
	for(i=0; i<N; i++)
		weights[i]=1;


	for(rCnt=0; rCnt<X; rCnt++) {
		for(cCnt=0; cCnt<Y; cCnt++) {
			
			L = 0;
			for(i=0; i<N; i++)
				if(inMask[cCnt + rCnt*Y + i*X*Y])
					vec[L++]=inTensor[cCnt + rCnt*Y + i*X*Y];

			fitValue2Skewed(vec, weights, mP, 0, L, topkPerc, botkPerc, MSSE_LAMBDA, optIters);
			//RobustSingleGaussianVec(vec, mP, 0, L, topkPerc, botkPerc, MSSE_LAMBDA, 10);
			modelParamsMap[cCnt + rCnt*Y + 0*X*Y] = mP[0];
			modelParamsMap[cCnt + rCnt*Y + 1*X*Y] = mP[1];
		}
	}
	free(weights);
	free(vec);
}

void RSGImage(float* inImage, unsigned char* inMask, float* modelParamsMap,
				unsigned int winX, unsigned int winY,
				unsigned int X, unsigned int Y, 
				float topkPerc, float botkPerc,
				float MSSE_LAMBDA, unsigned char stretch2CornersOpt,
				unsigned char numModelParams, unsigned char optIters) {

	float* x;
	x = (float*) malloc(winX*winY * sizeof(float));
	float* y;
	y = (float*) malloc(winX*winY * sizeof(float));
	float* z;
	z = (float*) malloc(winX*winY * sizeof(float));
	unsigned int rCnt, cCnt, numElem, pRowStart, pRowEnd, pClmStart, pClmEnd;
	float mP[4];
	float mP_Oone[2];
	pRowStart = 0;
	pRowEnd = winX;
	while(pRowStart<X) {
		
		pClmStart = 0;
		pClmEnd = winY;
		while(pClmStart<Y) {

			if(numModelParams==1) {
				numElem = 0;
				for(rCnt = pRowStart; rCnt < pRowEnd; rCnt++)
					for(cCnt = pClmStart; cCnt < pClmEnd; cCnt++)
						if(inMask[cCnt + rCnt*Y]) {
							z[numElem] = inImage[cCnt + rCnt*Y];
							numElem++;
						}
				if((int) (botkPerc*numElem)>12) {
					RobustSingleGaussianVec(z, mP_Oone, 0, numElem, topkPerc, botkPerc, MSSE_LAMBDA, optIters);

					for(rCnt=pRowStart; rCnt<pRowEnd; rCnt++) {
						for(cCnt=pClmStart; cCnt<pClmEnd; cCnt++) {
							modelParamsMap[cCnt + rCnt*Y + 0*X*Y] = mP_Oone[0];
							modelParamsMap[cCnt + rCnt*Y + 1*X*Y] = mP_Oone[1];
						}
					}
				}
			}
		
			if(numModelParams==4) {
				numElem = 0;
				for(rCnt = pRowStart; rCnt < pRowEnd; rCnt++)
					for(cCnt = pClmStart; cCnt < pClmEnd; cCnt++)
						if(inMask[cCnt + rCnt*Y]) {
							x[numElem] = rCnt;
							y[numElem] = cCnt;
							z[numElem] = inImage[cCnt + rCnt*Y];
							numElem++;
						}
				if((int) (botkPerc*numElem)>12) {
					RobustAlgebraicPlaneFitting(x, y, z, mP, numElem, topkPerc,
												botkPerc, MSSE_LAMBDA, stretch2CornersOpt);

					for(rCnt=pRowStart; rCnt<pRowEnd; rCnt++) {
						for(cCnt=pClmStart; cCnt<pClmEnd; cCnt++) {
							modelParamsMap[cCnt + rCnt*Y + 0*X*Y] = mP[0]*rCnt + mP[1]*cCnt + mP[2];
							modelParamsMap[cCnt + rCnt*Y + 1*X*Y] = mP[3];
						}
					}
				}
			}
			
			pClmStart += winY;			
			pClmEnd += winY;
			if((pClmEnd>Y) && (pClmStart < Y)) {
				pClmEnd = Y;
				pClmStart = pClmEnd - winY;
			}
		}
		pRowStart += winX;
		pRowEnd += winX;
		if((pRowEnd>X) && (pRowStart<X)) {
			pRowEnd = X;
			pRowStart = pRowEnd - winX;
		}
	}
	free(x);
	free(y);
	free(z);
}

void RSGImage_by_Image_Tensor(float* inImage_Tensor, unsigned char* inMask_Tensor, 
						float* model_mean, float* model_std,
						unsigned int winX, unsigned int winY,
						unsigned int N, unsigned int X, unsigned int Y, 
						float topkPerc, float botkPerc,
						float MSSE_LAMBDA, unsigned char stretch2CornersOpt,
						unsigned char numModelParams, unsigned char optIters) {

    unsigned int frmCnt, rCnt, cCnt;
	float* modelParamsMap;
	modelParamsMap = (float*) malloc(2*X*Y * sizeof(float));
	float* inImage;
	inImage = (float*) malloc(X*Y * sizeof(float));
	unsigned char* inMask;
	inMask = (unsigned char*) malloc(X*Y * sizeof(unsigned char));
			
	for(frmCnt=0; frmCnt<N; frmCnt++) {
		
		for(rCnt=0; rCnt<X; rCnt++) {
			for(cCnt=0; cCnt<Y; cCnt++) {
				inImage[cCnt + rCnt*Y] = inImage_Tensor[cCnt + rCnt*Y + frmCnt*X*Y];
				inMask[cCnt + rCnt*Y]  = inMask_Tensor[cCnt + rCnt*Y + frmCnt*X*Y];
				modelParamsMap[cCnt + rCnt*Y + 0*X*Y] = 0;
				modelParamsMap[cCnt + rCnt*Y + 1*X*Y] = 0;
			}
		}
		
		RSGImage(inImage, inMask, modelParamsMap,
				 winX, winY, X, Y,
				 topkPerc, botkPerc,
				 MSSE_LAMBDA, stretch2CornersOpt,
				 numModelParams, optIters);
		
		for(rCnt=0; rCnt<X; rCnt++) {
			for(cCnt=0; cCnt<Y; cCnt++) {
				model_mean[cCnt + rCnt*Y + frmCnt*X*Y] = modelParamsMap[cCnt + rCnt*Y + 0*X*Y];
				model_std[cCnt + rCnt*Y + frmCnt*X*Y]  = modelParamsMap[cCnt + rCnt*Y + 1*X*Y];
			}
		}
	}		
	free(modelParamsMap);
	free(inImage);
	free(inMask);
}

void fitBackgroundRadially(float* inImage, unsigned char* inMask, float* modelParamsMap,
 						   unsigned int minRes, unsigned int maxRes, unsigned int shellWidth,
						   unsigned char includeCenter, unsigned int finiteSampleBias,
						   unsigned int X, unsigned int Y,
						   float topkPerc, float botkPerc, float MSSE_LAMBDA, unsigned char optIters) {

	int r, maxR, shellCnt, pixX, pixY, i, j,pixInd, new_numElem;
	int shell_low, shell_high, numElem;
	float theta, sCnt, jump, X_Cent, Y_Cent;
	float mP[2];
	float* inVec;
	
	if(minRes<1) minRes=1;

	X_Cent = (int)(ceil(X/2));
	Y_Cent = (int)(ceil(Y/2));

	maxR = (int)(ceil(sqrt(X*X/4 + Y*Y/4)));
	float *resShells;
	resShells = (float*) malloc(maxR * sizeof(float));
	for(i=0; i<maxR; i++)
		resShells[i]=0;

	shellCnt = 0;
	if(includeCenter) {
		resShells[shellCnt] = 1;
		shellCnt++;
	}
	resShells[shellCnt] = minRes;

	while(resShells[shellCnt]<maxRes) {
		shellCnt++;
		resShells[shellCnt] = resShells[shellCnt-1] + shellWidth;
	}
	if(resShells[shellCnt]>maxRes) {
		resShells[shellCnt] = maxRes;
	}

	for(i=0; i<shellCnt; i++) {
		shell_low = resShells[i];
		shell_high = resShells[i+1];
		inVec = (float*) malloc((int)ceil( 2*M_PI*(shell_high-shell_low+1)*(shell_high+shell_low)/2 ) * sizeof(float));
		numElem = 0;
		for(r=shell_low; r<shell_high; r++) {
			for(theta = 0; theta < 2*M_PI; theta += 1.0/r) {
				pixX = (int)((float)r*cos(theta) + X_Cent);
				pixY = (int)((float)r*sin(theta) + Y_Cent);
				pixInd = pixY + pixX*Y;
				if( (pixY>=0) && (pixY<Y) && (pixX>=0) && (pixX<X) && (inMask[pixInd]) ) {
					inVec[numElem] = inImage[pixInd];
					numElem++;
				}
			}
		}

		if(numElem > finiteSampleBias) {
			new_numElem = 0;
			jump = (float)numElem/((float)finiteSampleBias);
			for(sCnt=0; sCnt<numElem; sCnt += jump)
				inVec[new_numElem++] = inVec[(int)sCnt];
			numElem = new_numElem;
		}

		if(optIters>0) {
			RobustSingleGaussianVec(inVec, mP, 0, numElem,
					topkPerc, botkPerc, MSSE_LAMBDA, optIters);
		}		
		else {
			mP[0] = 0;
			for(j=0; j<numElem; j++)
				mP[0] += inVec[j];
			mP[0] /= numElem;
			mP[1] = 0;
			for(j=0; j<numElem; j++)
				mP[1] += (inVec[j] - mP[0])*(inVec[j] - mP[0]);
			mP[1] /= numElem;
			mP[1] = sqrt(mP[1]);
		}
		free(inVec);
		
		for(r=shell_low; r<shell_high; r++) {
			for(theta = 0; theta < 2*M_PI; theta += 1.0/r) {
				pixX = (int)((float)r*cos(theta) + X_Cent);
				pixY = (int)((float)r*sin(theta) + Y_Cent);
				pixInd = pixY + pixX*Y;
				if( (pixY>=0) && (pixY<Y) && (pixX>=0) && (pixX<X)) {
					modelParamsMap[pixInd + 0*X*Y] = mP[0];
					modelParamsMap[pixInd + 1*X*Y] = mP[1];
				}
				pixInd = pixY + (pixX+1)*Y;
				if( (pixY>=0) && (pixY<Y) && (pixX+1>=0) && (pixX+1<X)) {
					modelParamsMap[pixInd + 0*X*Y] = mP[0];
					modelParamsMap[pixInd + 1*X*Y] = mP[1];
				}
				pixInd = pixY+1 + pixX*Y;
				if( (pixY+1>=0) && (pixY+1<Y) && (pixX>=0) && (pixX<X)) {
					modelParamsMap[pixInd + 0*X*Y] = mP[0];
					modelParamsMap[pixInd + 1*X*Y] = mP[1];
				}
				pixInd = pixY+1 + (pixX+1)*Y;
				if( (pixY+1>=0) && (pixY+1<Y) && (pixX+1>=0) && (pixX+1<X)) {
					modelParamsMap[pixInd + 0*X*Y] = mP[0];
					modelParamsMap[pixInd + 1*X*Y] = mP[1];
				}
			}
		}
	}

	free(resShells);

}
