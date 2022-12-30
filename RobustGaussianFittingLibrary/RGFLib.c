/*
------------------------------------------------------
This file is part of RobustGaussianFittingLibrary,
a free library WITHOUT ANY WARRANTY
Copyright: 2017-2020 LaTrobe University Melbourne,
           2019-2020 Deutsches Elektronen-Synchrotron
------------------------------------------------------
*/

/*
 * References:
	https://research.ijcaonline.org/volume122/number21/pxc3905155.pdf
 *
 */

#include "RGFLib.h"
#include <stdio.h>

void dfs(unsigned char* inMask, unsigned int* mask, 
         int x,  int y, unsigned int X, unsigned int Y,
         unsigned int label, unsigned int* islandSize, 
         int* rowsToMask, int* clmsToMask){
                
    int nx, ny, i;
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
            dfs(inMask, mask, nx,ny, X, Y, label,
            		islandSize, rowsToMask, clmsToMask);
    }
}

void islandRemoval(unsigned char* inMask, unsigned char* labelMap, 
                   unsigned int X, unsigned int Y,
                   unsigned int islandSizeThreshold) {
    unsigned int label = 1;
    int i, j, cnt;
    unsigned int islandSize[1];
    
    int* rowsToMask;
    int* clmsToMask;
    unsigned int* mask;
    
    mask = (unsigned int*) malloc(X*Y * sizeof(unsigned int));
    rowsToMask = (int*) malloc(X*Y * sizeof(int));
    clmsToMask = (int*) malloc(X*Y * sizeof(int));
    
    for(i=0; i< X; i++) {
        for(j=0; j < Y; j++) {
            mask[j + i*Y] = 0;
        }
    }
    
    for(i=0; i< X;i++) {
        for(j=0; j<Y; j++) {
            if((inMask[j + i*Y]>0) && (mask[j + i*Y]==0) ) {
                islandSize[0] = 0;
                dfs(inMask, mask, i, j, X, Y, label,
                		islandSize, rowsToMask, clmsToMask);
                
                if(islandSize[0] <= islandSizeThreshold)
                    for(cnt=0; cnt< islandSize[0]; cnt++)
                            labelMap[clmsToMask[cnt] + rowsToMask[cnt]*Y] = 1;
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
   while(1)    {
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
                                     
void merge(struct sortStruct dataVec[], int l, int m, int r) { 
    // based on https://www.geeksforgeeks.org/c-program-for-merge-sort/
    int i, j, k; 
    int n1 = m - l + 1; 
    int n2 =  r - m; 
    struct sortStruct* L; //n1 
    struct sortStruct* R; //n2
    L = (struct sortStruct*) malloc(n1 * sizeof(struct sortStruct));
    R = (struct sortStruct*) malloc(n2 * sizeof(struct sortStruct));
    
    for (i = 0; i < n1; i++) 
        L[i] = dataVec[l + i]; 
    for (j = 0; j < n2; j++) 
        R[j] = dataVec[m + 1+ j]; 
  
    i = 0;
    j = 0; 
    k = l; 
    while (i < n1 && j < n2) { 
        if (L[i].vecData <= R[j].vecData) { 
            dataVec[k] = L[i]; 
            i++; 
        } 
        else { 
            dataVec[k] = R[j]; 
            j++; 
        } 
        k++; 
    } 

    while (i < n1) { 
        dataVec[k] = L[i]; 
        i++; 
        k++; 
    } 

    while (j < n2) { 
        dataVec[k] = R[j]; 
        j++; 
        k++; 
    } 
    free(L);
    free(R);
} 
  
void mergeSort(struct sortStruct dataVec[], int l, int r) { 
    if (l < r) { 
        int m = l+(r-l)/2; 
        mergeSort(dataVec, l, m); 
        mergeSort(dataVec, m+1, r); 
        merge(dataVec, l, m, r); 
    } 
} 
  
void MQSort(struct sortStruct dataVec[], int l, int r) {
    int q;
    if (l < r) {
        if(r-l<10)
            mergeSort(dataVec,l,r);
        else {
            q = _partition(dataVec,l,r);
            MQSort(dataVec,l,q-1);
            MQSort(dataVec,q+1,r);
        }
    }
}

float MSSE(float* error, unsigned int vecLen, float MSSE_LAMBDA, 
           unsigned int k, float minimumResidual) {
    
    unsigned int i;
    float estScale, cumulative_sum, cumulative_sum_perv;
    struct sortStruct* sortedSqError;

    if((vecLen < k) || (vecLen<=0))
        return(-1);
    
    sortedSqError =
    		(struct sortStruct*) malloc(vecLen * sizeof(struct sortStruct));
    for (i = 0; i < vecLen; i++) {
        sortedSqError[i].vecData  = error[i]*error[i];
        sortedSqError[i].indxs = i;
    }
    quickSort(sortedSqError,0,vecLen-1);
    
    i=0;
    cumulative_sum = 0;
    while( (i < vecLen) && ( (i<k) ||
    		(sortedSqError[i].vecData < minimumResidual*minimumResidual) ) ){
        cumulative_sum += sortedSqError[i].vecData;
        i++;
    }
    cumulative_sum_perv = cumulative_sum;
    while (i < vecLen){
                                              
        if ( MSSE_LAMBDA*MSSE_LAMBDA * cumulative_sum_perv <
        		(float)(i)*sortedSqError[i].vecData )
            break;
        cumulative_sum_perv = cumulative_sum;
        cumulative_sum += sortedSqError[i].vecData ;
        i++;
    }
    cumulative_sum_perv = cumulative_sum;
    //Officially i - rho, and rho = 1 mostly here
    estScale = fabs(sqrt(cumulative_sum_perv / (float)(i-1)));

    free(sortedSqError);
    return estScale;
}

float MSSEWeighted(float* error, float* weights, unsigned int vecLen,
                   float MSSE_LAMBDA, unsigned int k, float minimumResidual) {
    unsigned int i, q;
    float estScale, cumulative_sum, tmp;
    struct sortStruct* sortedSqWErrors;

    if((vecLen < k) || (vecLen<=0))
        return(-1);

    sortedSqWErrors =
    		(struct sortStruct*) malloc(vecLen * sizeof(struct sortStruct));

    for (i = 0; i < vecLen; i++) {
        sortedSqWErrors[i].vecData  = weights[i]*error[i]*weights[i]*error[i];
        sortedSqWErrors[i].indxs = i;
    }
    quickSort(sortedSqWErrors,0,vecLen-1);

    cumulative_sum = 0;
    i=0;
    while ((i < vecLen) &&( (i<k) | (sortedSqWErrors[i].vecData <
    									minimumResidual*minimumResidual) ) ){
        cumulative_sum += sortedSqWErrors[i].vecData;
        i++;
    }
    for (i = i; i < vecLen; i++) {
        if ( MSSE_LAMBDA*MSSE_LAMBDA * cumulative_sum <
        		(i-1)*sortedSqWErrors[i].vecData )
            break;
        cumulative_sum += sortedSqWErrors[i].vecData ;
    }

    estScale = 0;
    tmp = 0;
    for(q=0; q<i; q++) {
        estScale += sortedSqWErrors[q].vecData;
        tmp += weights[sortedSqWErrors[q].indxs];
    }
    estScale = sqrt((i/(float)(i-1))*estScale / tmp);

    free(sortedSqWErrors);
    return estScale;
}

void fitValue(float* inVec,
			  float* inWeights,
			  float* modelParams,
			  float theta,
			  unsigned int inN,
              float likelyRatio,
			  float certainRatio,
              float MSSE_LAMBDA,
			  unsigned char optIters,
              float minimumResidual,
			  unsigned int downSampledSize) {

    float wAVG, estScale, dsCnt;
    unsigned int i, iter, ds_N;

    float *vec;
    vec = (float*) malloc(inN * sizeof(float));
    float *weights;
    weights = (float*) malloc(inN * sizeof(float));
    unsigned int N = 0;
    for (i = 0; i < inN; i++) {
    	if(inWeights[i]>0){
			vec[N] = inVec[i];
			weights[N] = inWeights[i];
			N++;
    	}
    }

    if ((downSampledSize>N) || (downSampledSize<5)) {
    	downSampledSize = N;
    }

    struct sortStruct* errorVec;
    errorVec = (struct sortStruct*) malloc(downSampledSize * sizeof(struct sortStruct));

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
    if(N<5) {
        certainRatio = 0;
        MSSE_LAMBDA = 0;
    }

    if(optIters>0) {
    	if(theta == modelParams[0])
    		theta = NEGATIVE_MAX;

        for(iter=0; iter<optIters; iter++) {
        	ds_N = 0;
        	dsCnt = 0;
			i = 0;
        	while (i < N) {
                errorVec[ds_N].vecData  = weights[i]*fabs(vec[i] - theta);
                errorVec[ds_N].indxs = i;
                ds_N++;
                dsCnt += (float)N/(float)downSampledSize;
                i = (unsigned int) dsCnt;
            }
            quickSort(errorVec,0,ds_N-1);

            theta = 0;
            wAVG = 0;
            for(i=(int)(ds_N*certainRatio); i<(int)(ds_N*likelyRatio); i++) {
                theta += vec[errorVec[i].indxs];
                wAVG += weights[errorVec[i].indxs];
            }
            if(wAVG>0)
            	theta = theta / wAVG;
            else
            	theta = 0;
        }

		float* residual;
		residual = (float*) malloc(N * sizeof(float));
		for (i = 0; i < N; i++) {
			residual[i]  = vec[i] - theta;
		}
		estScale = MSSEWeighted(residual, weights,
								N, MSSE_LAMBDA,
								(int)(N*likelyRatio),
								minimumResidual);
		if(optIters>1) {
			theta = 0;
			wAVG = 0;
			for(i=0; i<N; i++) {
				if(fabs(residual[i]) <= MSSE_LAMBDA*estScale) {
					theta += vec[i];
					wAVG += weights[i];
				}
			}
			if(wAVG>0) {
				theta = theta / wAVG;
			}
			else {
				theta = 0;
				estScale = 0;
			}

			for (i = 0; i < N; i++) {
				residual[i]  = vec[i] - theta;
			}
			estScale = MSSEWeighted(residual, weights,
									N, MSSE_LAMBDA,
									(int)(N*likelyRatio),
									minimumResidual);
			theta = 0;
			wAVG = 0;
			for(i=0; i<N; i++) {
				if(fabs(residual[i]) <= MSSE_LAMBDA*estScale) {
					theta += vec[i];
					wAVG += weights[i];
				}
			}
			if(wAVG > 0) {
				theta = theta / wAVG;
			}
			else {
				estScale = 0;
				theta = 0;
			}
			//for (i = 0; i < N; i++) {
			//	residual[i]  = vec[i] - theta;
			//}
		}

		free(residual);
    }
    else {
        theta = 0;
        wAVG = 0;
        for(i=0; i<N; i++) {
            theta += weights[i]*vec[i];
            wAVG += weights[i];
        }
        if(wAVG>0) {
			theta = theta / wAVG;

			estScale = 0;
	        for(i=0; i<N; i++)
	            estScale += weights[i]*(vec[i]-theta)*weights[i]*(vec[i]-theta);
	        estScale = sqrt(estScale/wAVG);
        }
        else {
			theta = 0;
			estScale = 0;
        }
    }

    modelParams[0] = theta;
    modelParams[1] = estScale;

	free(weights);
    free(errorVec);
}

void fitValue2Skewed(float* inVec,
			         float* inWeights,
			         float* modelParams,
			         float theta,
			         unsigned int inN,
					 float likelyRatio,
			         float certainRatio,
                     float MSSE_LAMBDA,
			         unsigned char optIters,
                     float minimumResidual,
			         unsigned int downSampledSize) {

    float wAVG, estScale, dsCnt;
    unsigned int i, iter, ds_N;
    float sampleRatio = likelyRatio - certainRatio;
    unsigned int step;
    float *vec;
    vec = (float*) malloc(inN * sizeof(float));
    float *weights;
    weights = (float*) malloc(inN * sizeof(float));
    unsigned int N = 0;
    for (i = 0; i < inN; i++) {
    	if(inWeights[i]>0){
			vec[N] = inVec[i];
			weights[N] = inWeights[i];
			N++;
    	}
    }

    if ((downSampledSize>N) || (downSampledSize<5)) {
    	downSampledSize = N;
    }

    struct sortStruct* errorVec;
    errorVec = (struct sortStruct*) malloc(downSampledSize * sizeof(struct sortStruct));

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
    if(N<5) {
        certainRatio = 0;
        MSSE_LAMBDA = 0;
    }

    float botkPerc_step = certainRatio/optIters;

    if(optIters>0) {
    	if(theta == modelParams[0])
    		theta = NEGATIVE_MAX;

        for(iter=0; iter<optIters; iter++) {
        	ds_N = 0;
        	dsCnt = 0;
			i = 0;
        	while (i < N) {
                errorVec[ds_N].vecData  = weights[i]*fabs(vec[i] - theta);
                errorVec[ds_N].indxs = i;
                ds_N++;
                dsCnt += (float)N/(float)downSampledSize;
                i = (unsigned int) dsCnt;
            }
            quickSort(errorVec,0,ds_N-1);

            theta = 0;
            wAVG = 0;
            if(certainRatio>0)
            	certainRatio -= botkPerc_step;
            if(certainRatio<0)
            	certainRatio = 0;
            step = (unsigned int)((likelyRatio - certainRatio) / sampleRatio);
            for(i=(int)(ds_N*certainRatio); i<(int)(ds_N*likelyRatio); i+=step) {
                theta += vec[errorVec[i].indxs];
                wAVG += weights[errorVec[i].indxs];
            }
            if(wAVG>0)
            	theta = theta / wAVG;
            else
            	theta = 0;
        }

		float* residual;
		residual = (float*) malloc(N * sizeof(float));
		for (i = 0; i < N; i++) {
			residual[i]  = vec[i] - theta;
		}
		estScale = MSSEWeighted(residual, weights,
								N, MSSE_LAMBDA,
								(int)(N*likelyRatio),
								minimumResidual);
		if(optIters>1) {
			theta = 0;
			wAVG = 0;
			for(i=0; i<N; i++) {
				if(fabs(residual[i]) <= MSSE_LAMBDA*estScale) {
					theta += vec[i];
					wAVG += weights[i];
				}
			}
			if(wAVG>0) {
				theta = theta / wAVG;
				for (i = 0; i < N; i++) {
					residual[i]  = vec[i] - theta;
				}
				estScale = MSSEWeighted(residual, weights,
										N, MSSE_LAMBDA,
										(int)(N*likelyRatio),
										minimumResidual);
			}
			else {
				estScale = 0;
			}
			theta = 0;
			wAVG = 0;
			for(i=0; i<N; i++) {
				if(fabs(residual[i]) <= MSSE_LAMBDA*estScale) {
					theta += vec[i];
					wAVG += weights[i];
				}
			}
			if(wAVG>0) {
				theta = theta / wAVG;
				//for (i = 0; i < N; i++) {
				//	residual[i]  = vec[i] - theta;
				//}
			}
			else {
				theta = 0;
				estScale = 0;
			}
		}

		free(residual);
    }
    else {
        theta = 0;
        wAVG = 0;
        for(i=0; i<N; i++) {
            theta += weights[i]*vec[i];
            wAVG += weights[i];
        }
        if(wAVG>0) {
			theta = theta / wAVG;

			estScale = 0;
	        for(i=0; i<N; i++)
	            estScale += weights[i]*(vec[i]-theta)*weights[i]*(vec[i]-theta);
	        estScale = sqrt(estScale/wAVG);
        }
        else {
			theta = 0;
			estScale = 0;
        }
    }

    modelParams[0] = theta;
    modelParams[1] = estScale;

	free(weights);
    free(errorVec);
}

void medianOfFits(float *vec, float *weights, 
                  float *modelParams, float theta, unsigned int N,
                  float likelyRatio_min, float likelyRatio_max,
				  unsigned int numSamples, float sampleRatio,
                  float MSSE_LAMBDA, unsigned char optIters,
				  float minimumResidual,
				  unsigned int downSampledSize) {
    float* rSTSDs;
    float mP[2];
    float likelyRatio;
    unsigned int i, medArg;

    if (numSamples<1)
        numSamples = 1;

    rSTSDs = (float*) malloc(numSamples * sizeof(float));
    
    struct sortStruct* rMeans;
    rMeans =
    	(struct sortStruct*) malloc(numSamples * sizeof(struct sortStruct));

    likelyRatio = likelyRatio_min;
    for(i=1; i<=numSamples; i++) {
    	fitValue2Skewed(vec,
						weights,
						mP,
						theta,
						N,
						likelyRatio,
						sampleRatio*likelyRatio_min,
						MSSE_LAMBDA,
						optIters,
						minimumResidual,
						downSampledSize);

        rMeans[i-1].vecData = mP[0];
        rMeans[i-1].indxs = i;

        rSTSDs[i-1] = mP[1];
        likelyRatio += (likelyRatio_max - likelyRatio_min)/numSamples;
    }
    quickSort(rMeans, 0, numSamples-1);
    medArg = rMeans[(int)(numSamples/2)].indxs;

    modelParams[0] = rMeans[medArg].vecData;
    modelParams[1] = rSTSDs[medArg];
    
    free(rMeans);
    free(rSTSDs);
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

void lineAlgebraicModelEval(float* x, float* y_fit,
		float* mP, unsigned int N) {
    unsigned int i;
    for (i=0;i<N;i++)
        y_fit[i]=mP[0]*x[i]+mP[1];
}

void RobustAlgebraicLineFitting(float* x,
								float* y,
								float* mP,
                                unsigned int N,
								float likelyRatio,
								float certainRatio,
								float MSSE_LAMBDA) {
    float model[2];
    unsigned int i, iter, cnt;
    unsigned int sampleSize;
    float *residual;
    float* sample_x;
    float* sample_y;
    struct sortStruct* errorVec;
    errorVec = (struct sortStruct*) malloc(N * sizeof(struct sortStruct));

    sampleSize = (unsigned int)(N*likelyRatio)- (unsigned int)(N*certainRatio);

    sample_x = (float*) malloc(sampleSize * sizeof(float));
    sample_y = (float*) malloc(sampleSize * sizeof(float));

    model[0]=0;
    model[1]=0;
    for(iter=0; iter<12; iter++) {
        
        for (i = 0; i < N; i++) {
            //could have called the function
            errorVec[i].vecData  = fabs(y[i] - (model[0]*x[i] + model[1]));
            errorVec[i].indxs = i;
        }
        quickSort(errorVec,0,N-1);
        
        cnt = 0;
        for(i=(int)(N*certainRatio); i<(int)(N*likelyRatio); i++) {
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
        residual[i]  = y[i] - (model[0]*x[i] + model[1]);
    	// + ((float) rand() / (RAND_MAX))/4; //Noise stabilizes MSSE
    mP[2] = MSSE(residual, N, MSSE_LAMBDA, (int)(N*likelyRatio), 0);

    free(sample_x);
    free(sample_y);
    free(residual);
    free(errorVec);
}

void RobustAlgebraicLineFittingTensor(float* inTensorX, float* inTensorY,
                                      float* modelParamsMap, unsigned int N,
                                      unsigned int X, unsigned int Y, 
                                      float likelyRatio, float certainRatio,
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
            
            RobustAlgebraicLineFitting(xVals, yVals, mP, N,
            		likelyRatio, certainRatio, MSSE_LAMBDA);
            modelParamsMap[cCnt + rCnt*Y + 0*X*Y] = mP[0];
            modelParamsMap[cCnt + rCnt*Y + 1*X*Y] = mP[1];
            modelParamsMap[cCnt + rCnt*Y + 2*X*Y] = mP[2];
        }
    }
    free(xVals);
    free(yVals);
}

void TLS_AlgebraicPlaneFitting(float* x, float* y,
		float* z, float* mP, unsigned int N) {
    unsigned int i;
    float x_mean, y_mean, z_mean, x_x_sum, y_y_sum;
	float x_y_sum, x_z_sum, y_z_sum, a,b,c, D;

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

void RobustAlgebraicPlaneFitting(float* x, float* y, float* z, 
                                 float* mP, float* mP_Init,
                                 unsigned int N, float likelyRatio,
								 float certainRatio,
                                 float MSSE_LAMBDA,
								 unsigned char stretch2CornersOpt,
                                 float minimumResidual,
								 unsigned char optIters) {
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
    
    sampleSize = (unsigned int)(N*likelyRatio)- (unsigned int)(N*certainRatio);
    sample_x = (float*) malloc(sampleSize * sizeof(float));
    sample_y = (float*) malloc(sampleSize * sizeof(float));
    sample_z = (float*) malloc(sampleSize * sizeof(float));
    sample_inds = (unsigned int*) malloc(sampleSize * sizeof(unsigned int));

    cnt = 0;
    for(i=(unsigned int)(N*certainRatio); i<(unsigned int)(N*likelyRatio); i++)
        sample_inds[cnt++] = i;
    
    model[0] = mP_Init[0];
    model[1] = mP_Init[1];
    model[2] = mP_Init[2];
    for(iter=0; iter<optIters; iter++) {
        
        for (i = 0; i < N; i++) {
            errorVec[i].vecData  =
            		fabs(z[i] - (model[0]*x[i] + model[1]*y[i] + model[2]));
            errorVec[i].indxs = i;
        }
        quickSort(errorVec,0,N-1);
        
  
        if(stretch2CornersOpt>0) {
            for(i=0; i<(unsigned int)(N*likelyRatio); i++) {
                sortedX[i] = x[errorVec[i].indxs];
                sortedY[i] = y[errorVec[i].indxs];
            }
            isStretchingPossible = stretch2CornersFunc(sortedX, sortedY,
            		(unsigned int)(N*likelyRatio),
					stretch2CornersOpt,
					sample_inds,
					sampleSize);
            if(isStretchingPossible==0) {
                cnt = 0;
                for(i=(unsigned int)(N*certainRatio);
                		i<(unsigned int)(N*likelyRatio);
                		i++)
                    sample_inds[cnt++] = i;
            }
        }

        for(i=0; i<sampleSize; i++) {
            sample_x[i] = x[errorVec[sample_inds[i]].indxs];
            sample_y[i] = y[errorVec[sample_inds[i]].indxs];
            sample_z[i] = z[errorVec[sample_inds[i]].indxs];
        }
        TLS_AlgebraicPlaneFitting(sample_x, sample_y,
        		sample_z, model, sampleSize);
    }

    mP[0] = model[0];
    mP[1] = model[1];
    mP[2] = model[2];
    
    for (i = 0; i < N; i++)
        residual[i]  = z[i] - (model[0]*x[i] + model[1]*y[i] + model[2]);
    
    mP[3] = MSSE(residual, N, MSSE_LAMBDA, (int)(N*likelyRatio), minimumResidual);

    free(residual);
    free(errorVec);
    free(sortedX);
    free(sortedY);
    free(sample_inds);
    free(sample_x);
    free(sample_y);
    free(sample_z);
}

void fitValueTensor(float* inTensor, float* inWeights, float* modelParamsMap,
					unsigned int N, unsigned int X, unsigned int Y,
					float likelyRatio, float certainRatio, float MSSE_LAMBDA,
					unsigned char optIters, float minimumResidual,
					unsigned int downSampledSize) {

    float* vec;
    float* weights;

    float mP[2];
    unsigned int rCnt, cCnt, i, L;

    vec = (float*) malloc(N * sizeof(float));
    weights = (float*) malloc(N * sizeof(float));

    for(rCnt=0; rCnt<X; rCnt++) {
        for(cCnt=0; cCnt<Y; cCnt++) {
            L = 0;
            for(i=0; i<N; i++) {
				vec[L]=inTensor[cCnt + rCnt*Y + i*X*Y];
				weights[L++]=inWeights[cCnt + rCnt*Y + i*X*Y];
            }
            fitValue2Skewed(vec, weights, mP, 0, L, likelyRatio, certainRatio,
            						MSSE_LAMBDA, optIters, minimumResidual,
									downSampledSize);
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
                float likelyRatio, float certainRatio,
                float MSSE_LAMBDA, unsigned char stretch2CornersOpt,
                unsigned char numModelParams, unsigned char optIters,
				float minimumResidual) {

    float* x;
    x = (float*) malloc(winX*winY * sizeof(float));
    float* y;
    y = (float*) malloc(winX*winY * sizeof(float));
    float* z;
    z = (float*) malloc(winX*winY * sizeof(float));
    float* w;
    w = (float*) malloc(winX*winY * sizeof(float));
    unsigned int rCnt, cCnt, numElem, pRowStart, pRowEnd, pClmStart, pClmEnd;
    float mP[4];
    float mP_MOne[2];
    pRowStart = 0;
    pRowEnd = winX;
    while(pRowStart<X) {
        
        pClmStart = 0;
        pClmEnd = winY;
        while(pClmStart<Y) {

            if(numModelParams==1) {
                numElem = 0;
                for(rCnt = pRowStart; rCnt < pRowEnd; rCnt++) {
                    for(cCnt = pClmStart; cCnt < pClmEnd; cCnt++) {
                        if(inMask[cCnt + rCnt*Y]) {
                            z[numElem] = inImage[cCnt + rCnt*Y];
                            w[numElem] = inMask[cCnt + rCnt*Y];
                            numElem++;

                        }
					}
				}
                mP_MOne[0] = 0;
				mP_MOne[1] = 0;
				fitValue2Skewed(z, w, mP_MOne, 0, numElem,
										likelyRatio, certainRatio, 
										MSSE_LAMBDA, optIters,
										minimumResidual,
										numElem);
				for(rCnt=pRowStart; rCnt<pRowEnd; rCnt++) {
					for(cCnt=pClmStart; cCnt<pClmEnd; cCnt++) {
						modelParamsMap[cCnt + rCnt*Y + 0*X*Y] = mP_MOne[0];
						modelParamsMap[cCnt + rCnt*Y + 1*X*Y] = mP_MOne[1];
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
                if((int) (certainRatio*numElem)>12) {
                    mP[0]=0; mP[1]=0; mP[2]=0;
                    
                    RobustAlgebraicPlaneFitting(x, y, z, mP, mP,
                                                numElem, likelyRatio,
                                                certainRatio, MSSE_LAMBDA,
												stretch2CornersOpt,
												minimumResidual, 12);
                    
                    for(rCnt=pRowStart; rCnt<pRowEnd; rCnt++) {
                        for(cCnt=pClmStart; cCnt<pClmEnd; cCnt++) {
                            modelParamsMap[cCnt + rCnt*Y + 0*X*Y] = mP[0]*rCnt
                            							  + mP[1]*cCnt + mP[2];
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
    free(w);
}

void RSGImage_by_Image_Tensor(float* inImage_Tensor,
		unsigned char* inMask_Tensor,
                        float* model_mean, float* model_std,
                        unsigned int winX, unsigned int winY,
                        unsigned int N, unsigned int X, unsigned int Y, 
                        float likelyRatio, float certainRatio,
                        float MSSE_LAMBDA, unsigned char stretch2CornersOpt,
                        unsigned char numModelParams, unsigned char optIters,
						float minimumResidual) {

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
                inImage[cCnt + rCnt*Y] =
                		inImage_Tensor[cCnt + rCnt*Y + frmCnt*X*Y];
                inMask[cCnt + rCnt*Y]  =
                		inMask_Tensor[cCnt + rCnt*Y + frmCnt*X*Y];
                modelParamsMap[cCnt + rCnt*Y + 0*X*Y] = 0;
                modelParamsMap[cCnt + rCnt*Y + 1*X*Y] = 0;
            }
        }
        
        RSGImage(inImage, inMask, modelParamsMap,
                 winX, winY, X, Y,
                 likelyRatio, certainRatio,
                 MSSE_LAMBDA, stretch2CornersOpt,
                 numModelParams, optIters, minimumResidual);
        
        for(rCnt=0; rCnt<X; rCnt++) {
            for(cCnt=0; cCnt<Y; cCnt++) {
                model_mean[cCnt + rCnt*Y + frmCnt*X*Y] =
                		modelParamsMap[cCnt + rCnt*Y + 0*X*Y];
                model_std[cCnt + rCnt*Y + frmCnt*X*Y]  =
                		modelParamsMap[cCnt + rCnt*Y + 1*X*Y];
            }
        }
    }        
    free(modelParamsMap);
    free(inImage);
    free(inMask);
}

void _fitBackgroundRadially(float* inImage, unsigned char* inMask,
                           float* modelParamsMap, float* vecMP,
                           unsigned int minRes,
                           unsigned int maxRes, 
                           unsigned int shellWidth,
                           unsigned char includeCenter, 
                           unsigned int finiteSampleBias,
                           unsigned int X, unsigned int Y,
                           float likelyRatio, float certainRatio, 
                           float MSSE_LAMBDA, 
                           unsigned char optIters,
						   float minimumResidual) {

    int r, maxR, shellCnt, pixX, pixY, i, pixInd, new_numElem;
    int shell_low, shell_high, numElem;
    float theta, sCnt, jump, X_Cent, Y_Cent;
    float mP[2];
    float* inVec;
    float* inWeights;
    float shellWidth_FSB, currentShell, area;
    float squareSide;
    
    if(finiteSampleBias<3)
    	finiteSampleBias = 3;

    if(minRes<3)
    	minRes = 3;

    X_Cent = (int)(ceil(X/2));
    Y_Cent = (int)(ceil(Y/2));

    squareSide = X_Cent;
    if(squareSide > Y_Cent)		squareSide = Y_Cent;

    maxR = (int)(ceil(sqrt(X*X/4 + Y*Y/4)));
    float *resShells;
    resShells = (float*) malloc(maxR * sizeof(float));
    for(i=0; i<maxR; i++)	resShells[i]=0;

    shellCnt = 0;
    if(includeCenter) {
        resShells[shellCnt] = 1;	//no radius 0
        shellCnt++;
    }
    resShells[shellCnt] = minRes;
    while(resShells[shellCnt] < maxRes) {
    	shellCnt++;
    	currentShell = resShells[shellCnt-1];
        shellWidth_FSB = 0;
        area = 0;
        while( (area<8) &&
        	   ( currentShell+shellWidth_FSB < maxRes ) ) {
			if(currentShell+shellWidth_FSB > squareSide) {
				area += 4*(M_PI/2 - 2*(acos(squareSide/(currentShell+
						shellWidth_FSB))))*(currentShell+shellWidth_FSB);
			}
			else {
				area += 2*M_PI*(currentShell + shellWidth_FSB);
			}
			shellWidth_FSB += 1;
        }

        if(shellWidth_FSB<shellWidth)	shellWidth_FSB = shellWidth;
        resShells[shellCnt] = currentShell + shellWidth_FSB;
    }

    if(resShells[shellCnt]>maxRes) {
        resShells[shellCnt] = maxRes;
    }

    for(i=0; i<shellCnt; i++) {
        shell_low = resShells[i];
        shell_high = resShells[i+1];


        //Fill in the vector of resolution shell
        inVec = (float*) malloc((int)ceil( 2*M_PI*(shell_high-shell_low+1)*
        		                    (shell_high+shell_low)/2 ) * sizeof(float));
        inWeights = (float*) malloc((int)ceil( 2*M_PI*(shell_high-shell_low+1)*
                (shell_high+shell_low)/2 ) * sizeof(float));
        numElem = 0;
        for(r=shell_low; r<shell_high; r++) {
            for(theta = 0; theta < 2*M_PI; theta += 1.0/r) {
                pixX = (int)((float)r*cos(theta) + X_Cent);
                pixY = (int)((float)r*sin(theta) + Y_Cent);
                pixInd = pixY + pixX*Y;
                if( (pixY>=0) && (pixY<Y) && (pixX>=0) &&
                		(pixX<X) && (inMask[pixInd]) ) {
                    inVec[numElem] = inImage[pixInd];
                    inWeights[numElem] = inMask[pixInd];
                    numElem++;
                }
            }
        }
        //Downsample it if it is too big
        if(numElem > finiteSampleBias) {
            new_numElem = 0;
            jump = (float)numElem/((float)finiteSampleBias);
            for(sCnt=0; sCnt<numElem; sCnt += jump) {
                inVec[new_numElem] = inVec[(int)sCnt];
                inWeights[new_numElem++] = inWeights[(int)sCnt];
            }
            numElem = new_numElem;
        }
        //Get the model parameters

		fitValue2Skewed(inVec, inWeights, mP, 0, numElem,
				 likelyRatio, certainRatio, MSSE_LAMBDA,
				 optIters, minimumResidual, numElem);

		free(inVec);
        free(inWeights);


        //Fill the output with model parameters
        //We will set four pixels around each radius as this shell
        //This way no pixel will be forgotten
        for(r=shell_low; r<shell_high; r++) {
			vecMP[r + 0*maxR] = mP[0];
			vecMP[r + 1*maxR] = mP[1];
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
                                float likelyRatio,
								float certainRatio,
                                float MSSE_LAMBDA,
                                unsigned char optIters,
						        float minimumResidual) {

	int r, maxR, shellCnt, pixX, pixY, i, pixInd, new_numElem;
    int shell_low, shell_high, numElem, fCnt;
    float theta, sCnt, jump, X_Cent, Y_Cent;
    float mP[2];
    float* inVec;
    float* inWeights;
    float shellWidth_FSB, currentShell, area;
    float squareSide;

    if(finiteSampleBias<3)
    	finiteSampleBias = 3;

    if(minRes<3)
    	minRes = 3;

    X_Cent = (int)(ceil(X/2));
    Y_Cent = (int)(ceil(Y/2));

    squareSide = X_Cent;
    if(squareSide > Y_Cent)		squareSide = Y_Cent;

    maxR = (int)(ceil(sqrt(X*X/4 + Y*Y/4)));
    float *resShells;
    resShells = (float*) malloc(maxR * sizeof(float));
    for(i=0; i<maxR; i++)	resShells[i]=0;

    shellCnt = 0;
    if(includeCenter) {
        resShells[shellCnt] = 1;	//no radius 0
        shellCnt++;
    }
    resShells[shellCnt] = minRes;
    while(resShells[shellCnt] < maxRes) {
    	shellCnt++;
    	currentShell = resShells[shellCnt-1];
        shellWidth_FSB = 0;
        area = 0;
        while( (area<8) &&
        	   ( currentShell+shellWidth_FSB < maxRes ) ) {
			if(currentShell+shellWidth_FSB > squareSide) {
				area += 4*(M_PI/2 - 2*(acos(squareSide/(currentShell+
						shellWidth_FSB))))*(currentShell+shellWidth_FSB);
			}
			else {
				area += 2*M_PI*(currentShell + shellWidth_FSB);
			}
			shellWidth_FSB += 1;
        }

        if(shellWidth_FSB<shellWidth)	shellWidth_FSB = shellWidth;
        resShells[shellCnt] = currentShell + shellWidth_FSB;
    }

    if(resShells[shellCnt]>maxRes) {
        resShells[shellCnt] = maxRes;
    }

    for(i=0; i<shellCnt; i++) {
        shell_low = resShells[i];
        shell_high = resShells[i+1];

        //Fill in the vector of resolution shell
        inVec = (float*) malloc((int)ceil( N*2*M_PI*(shell_high-shell_low+1)
        		                   *(shell_high+shell_low)/2 ) * sizeof(float));
        inWeights = (float*) malloc((int)ceil( N*2*M_PI
        		* (shell_high-shell_low+1)*(shell_high+shell_low)/2 )
				* sizeof(float));
        numElem = 0;
        for(r=shell_low; r<shell_high; r++) {
            for(theta = 0; theta < 2*M_PI; theta += 1.0/r) {
                pixX = (int)((float)r*cos(theta) + X_Cent);
                pixY = (int)((float)r*sin(theta) + Y_Cent);
                pixInd = pixY + pixX*Y;
                if( (pixY>=0) && (pixY<Y) && (pixX>=0) &&
                		(pixX<X) && (inMask[pixInd]) ) {
                    for(fCnt = 0; fCnt < N; fCnt++) {
						inVec[numElem] = inTensor[pixInd + fCnt*X*Y];
						inWeights[numElem] = inMask[pixInd + fCnt*X*Y];
						numElem++;
                    }
                }
            }
        }
        //Downsample it if it is too big
        if(numElem > finiteSampleBias) {
            new_numElem = 0;
            jump = (float)numElem/((float)finiteSampleBias);
            for(sCnt=0; sCnt<numElem; sCnt += jump) {
                inVec[new_numElem] = inVec[(int)sCnt];
                inWeights[new_numElem++] = inWeights[(int)sCnt];
            }
            numElem = new_numElem;
        }
        //Get the model parameters

		fitValue2Skewed(inVec, inWeights, mP, 0, numElem,
				 likelyRatio, certainRatio, MSSE_LAMBDA,
				 optIters, minimumResidual, numElem);

		free(inVec);
        free(inWeights);

        //Fill the output with model parameters
        //We will set four pixels around each radius as this shell
        //This way no pixel will be forgotten
        for(r=shell_low; r<shell_high; r++) {
			vecMP[r + 0*maxR] = mP[0];
			vecMP[r + 1*maxR] = mP[1];
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
                           float likelyRatio, float certainRatio,
                           float MSSE_LAMBDA,
                           unsigned char optIters,
						   float minimumResidual) {

    int r, maxR, shellCnt, pixX, pixY, i, pixInd, new_numElem;
    int shell_low, shell_high, numElem;
    float theta, sCnt, jump;
    float mP[2];
    float* inVec;
    float* inWeights;
    float currentShell;
    int includeFarEnd = 0;
    int X_righSide, Y_righSide, maxR_LU, maxR_LD, maxR_RU, maxR_RD;

    if(shellWidth<1)
    	shellWidth = 1;

    if(stride == 0)
    	stride = shellWidth;

    if(finiteSampleBias<3)
    	finiteSampleBias = 3;

    if(minRes<3)
    	minRes = 3;


    X_righSide = X - X_Cent;
    Y_righSide = Y - Y_Cent;

    maxR_LU = (int)(ceil(sqrt(X_Cent*X_Cent + Y_Cent*Y_Cent)));
    maxR_LD = (int)(ceil(sqrt(X_Cent*X_Cent + Y_righSide*Y_righSide)));
    maxR_RU = (int)(ceil(sqrt(X_righSide*X_righSide + Y_Cent*Y_Cent)));
    maxR_RD = (int)(ceil(sqrt(X_righSide*X_righSide + Y_righSide*Y_righSide)));

    maxR = maxR_LU;
    if(maxR < maxR_LD)    maxR = maxR_LD;
    if(maxR < maxR_RU)    maxR = maxR_RU;
    if(maxR < maxR_RD)    maxR = maxR_RD;

    if(minRes >= maxR)
    	return;

    if(maxRes>maxR)
    	includeFarEnd = 1;

    if(maxRes>maxR - 10)
    	maxRes = maxR - 10;

    if(maxRes<minRes)
    	maxRes = minRes + 1;

    float *resShells;
    resShells = (float*) malloc(maxR * sizeof(float));
    for(i=0; i<maxR; i++)	resShells[i]=0;

    shellCnt = 0;
    if(includeCenter) {
        resShells[shellCnt] = 1;	//radius 1
        shellCnt++;
    }
    resShells[shellCnt] = minRes;
    while(resShells[shellCnt] < maxRes) {
    	shellCnt++;
    	currentShell = resShells[shellCnt-1];
        if(currentShell + stride >= maxRes) {
        	if(includeFarEnd) {
        		resShells[shellCnt] = maxR;
        	}
        	else {
        		resShells[shellCnt] = maxRes;
        	}
        }
        else {
        	resShells[shellCnt] = currentShell + stride;
        }
    }

    for(i=0; i<shellCnt; i++) {
        shell_low = resShells[i];
        shell_high = shell_low + shellWidth;

        //Fill in the vector of resolution shell
        inVec = (float*) malloc((int)ceil( 2*M_PI*(shell_high-shell_low+1)*
        		                    (shell_high+shell_low)/2 ) * sizeof(float));
        inWeights = (float*) malloc((int)ceil( 2*M_PI*(shell_high-shell_low+1)*
                (shell_high+shell_low)/2 ) * sizeof(float));
        numElem = 0;
        for(r=shell_low; r<shell_high; r++) {
            for(theta = 0; theta < 2*M_PI; theta += 1.0/r) {
                pixX = (int)((float)r*cos(theta) + X_Cent);
                pixY = (int)((float)r*sin(theta) + Y_Cent);
                pixInd = pixY + pixX*Y;
                if( (pixY>=0) && (pixY<Y) && (pixX>=0) &&
                		(pixX<X) && (inMask[pixInd]) ) {
                    inVec[numElem] = inImage[pixInd];
                    inWeights[numElem] = inMask[pixInd];
                    numElem++;
                }
            }
        }
        //Downsample it if it is too big
        if(numElem > finiteSampleBias) {
            new_numElem = 0;
            jump = (float)numElem/((float)finiteSampleBias);
            for(sCnt=0; sCnt<numElem; sCnt += jump) {
                inVec[new_numElem] = inVec[(int)sCnt];
                inWeights[new_numElem++] = inWeights[(int)sCnt];
            }
            numElem = new_numElem;
        }
        //Get the model parameters

		fitValue2Skewed(inVec, inWeights, mP, 0, numElem,
				 likelyRatio, certainRatio, MSSE_LAMBDA,
				 optIters, minimumResidual, numElem);

		free(inVec);
        free(inWeights);

        //Fill the output with model parameters
        //We will set four pixels around each radius as this shell
        //This way no pixel will be forgotten
        for(r=shell_low; r<resShells[i+1]; r++) {
			vecMP[r + 0*maxR] = mP[0];
			vecMP[r + 1*maxR] = mP[1];
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
