// gcc -fPIC -shared -o RobustGausFitLib.c RobustGausFitLib.so
#include <math.h>
#include <stdlib.h>


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

float MSSE(float *error, unsigned int vecLen, float MSSE_LAMBDA, unsigned int k) {
	unsigned int i;
	float estScale, cumulative_sum, cumulative_sum_perv;
	struct sortStruct* sortedSqError;
	estScale = 30000.0;

	if((k < 12) || (vecLen<k))
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
	cumulative_sum_perv = cumulative_sum;
	for (i = k; i < vecLen; i++) {
		if ( MSSE_LAMBDA*MSSE_LAMBDA * cumulative_sum < (i-1)*sortedSqError[i].vecData )		// in (i-1), the 1 is the dimension of model
			break;
		cumulative_sum_perv = cumulative_sum;
		cumulative_sum += sortedSqError[i].vecData ;
	}
	estScale = fabs(sqrt(cumulative_sum_perv / ((float)i - 1)));
	return estScale;
}

void RobustSingleGaussianVec(float *vec, float *modelParams, unsigned int N,
		float topKthPerc, float bottomKthPerc, float MSSE_LAMBDA) {

	float model, avg, tmp, estScale; // ,stdDev;
	unsigned int i, iter;

	float *residual;

	struct sortStruct* errorVec;
	errorVec = (struct sortStruct*) malloc(N * sizeof(struct sortStruct));
	
	model=0;
	for(iter=0; iter<10; iter++) {
		for (i = 0; i < N; i++) {
			errorVec[i].vecData  = fabs(vec[i] - model);
			errorVec[i].indxs = i;
		}
		quickSort(errorVec,0,N-1);
		model = 0;
		tmp = 0;
		for(i=(int)(N*bottomKthPerc); i<(int)(N*topKthPerc); i++) {
			model += vec[errorVec[i].indxs];
			tmp++;
		}
		model = model / tmp;
	}

	avg = 0;
	for(i=0; i<(int)(N*topKthPerc); i++)
		avg += vec[errorVec[i].indxs];
	avg = avg / (int)(N*topKthPerc);

	if((int)(N*topKthPerc)>12) {	
		residual = (float*) malloc(N * sizeof(float));
		for (i = 0; i < N; i++)
			residual[i]  = vec[i] - avg;// + ((double) rand() / (RAND_MAX))/4;	//Noise stabilizes MSSE
		estScale = MSSE(residual, N, MSSE_LAMBDA, (int)(N*topKthPerc));
		free(residual);
	}
	else {	//finite sample bias of MSSE is 12[RezaJMIJV'06]
		estScale = 0;
		tmp = 0;
		for(i=0; i<(int)(N*topKthPerc); i++) {
			tmp = vec[errorVec[i].indxs] - avg;
			estScale += tmp*tmp;
		}
		estScale = sqrt(estScale / (int)(N*topKthPerc));
	}
	modelParams[0] = avg;
	modelParams[1] = estScale;

	free(errorVec);
}

void TLS_AlgebraicLineFitting(float* x, float* y, float* mP, unsigned int N) {
	unsigned int i;
	double xsum,x2sum,ysum,xysum; 
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
    mP[0]=(N*xysum-xsum*ysum)/(N*x2sum-xsum*xsum);
    mP[1]=(x2sum*ysum-xsum*xysum)/(x2sum*N-xsum*xsum);
}

void lineAlgebraicModelEval(float* x, float* y_fit, float* mP, unsigned int N) {
    unsigned int i;
	for (i=0;i<N;i++)
        y_fit[i]=mP[0]*x[i]+mP[1];
}

void RobustAlgebraicLineFitting(float* x, float* y, float* mP,
							unsigned int N, float topKthPerc, float bottomKthPerc, float MSSE_LAMBDA) {
	float model[2];
	unsigned int i, iter, cnt;
	unsigned int sampleSize;
	float *residual;
	float* sample_x;
	float* sample_y;
	struct sortStruct* errorVec;
	errorVec = (struct sortStruct*) malloc(N * sizeof(struct sortStruct));

	sampleSize = (unsigned int)(N*topKthPerc)- (unsigned int)(N*bottomKthPerc);

	model[0]=0;
	model[1]=0;
	for(iter=0; iter<12; iter++) {
		
		for (i = 0; i < N; i++) {
			errorVec[i].vecData  = fabs(y[i] - (model[0]*x[i] + model[1]));	//could have called the function
			errorVec[i].indxs = i;
		}
		quickSort(errorVec,0,N-1);
		
		sample_x = (float*) malloc(sampleSize * sizeof(float));
		sample_y = (float*) malloc(sampleSize * sizeof(float));
		cnt = 0;
		for(i=(int)(N*bottomKthPerc); i<(int)(N*topKthPerc); i++) {
			sample_x[cnt] = x[errorVec[i].indxs];
			sample_y[cnt] = y[errorVec[i].indxs];
			cnt++;
		}
		TLS_AlgebraicLineFitting(sample_x, sample_y, model, sampleSize);
		free(sample_x);
		free(sample_y);
	}

	mP[0] = model[0];
	mP[1] = model[1];
	residual = (float*) malloc(N * sizeof(float));
	for (i = 0; i < N; i++)
		residual[i]  = y[i] - (model[0]*x[i] + model[1]) + ((double) rand() / (RAND_MAX))/4;	//Noise stabilizes MSSE
	mP[2] = MSSE(residual, N, MSSE_LAMBDA, (int)(N*topKthPerc));
	free(residual);
	free(errorVec);
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
	a = (y_y_sum * x_z_sum - x_y_sum * y_z_sum)/D;
    b = (x_x_sum * y_z_sum - x_y_sum * x_z_sum)/D;
    c = z_mean - a*x_mean - b*y_mean;
	mP[0] = a;
	mP[1] = b;
	mP[2] = c;
}

void stretch2CornersFunc(struct sortStruct errorVec[], float* x, float* y, 
						unsigned int N, unsigned char D) {
    if((D<2) || (N/100< D) || (N<200))
		return;
						
	unsigned int i;
	float x_min, y_min, x_max, y_max;
	float winXCnt, winYCnt;
	unsigned int winCnt, winStart, newSortInd;
	unsigned int* winPtsCnt;
	
	struct sortStruct* errorVecMod;
	errorVecMod = (struct sortStruct*) malloc(N * sizeof(struct sortStruct));
	winPtsCnt = (unsigned int*) malloc(D*D * sizeof(unsigned int));

	for(i=0; i<D*D; i++)
		winPtsCnt[i]=0;
	x_min = x[0];
	x_max = x[0];
	y_min = y[0];
	y_max = y[0];	
	for (i = 0; i < N; i++) {
		if(x_min>=x[i])
			x_min = x[i];
		if(x_max<=x[i])
			x_max = x[i];
		if(y_min>=y[i])
			y_min = y[i];
		if(y_max<=y[i])
			y_max = y[i];
		
		errorVecMod[i].vecData = errorVec[i].vecData;
		errorVecMod[i].indxs = errorVec[i].indxs;
	}
	
	for (i = 0; i < N; i++) {
		winXCnt = (int) floor(D*(x[i] - x_min)/(x_max-x_min+1));
		winYCnt = (int) floor(D*(y[i] - y_min)/(y_max-y_min+1));
		winCnt = winYCnt + winXCnt*D;
		winStart = (int) floor(winCnt*N/(D*D));
		newSortInd = winPtsCnt[winCnt] + winStart;
		winPtsCnt[winCnt]++;
		errorVec[newSortInd].vecData  = errorVecMod[i].vecData;
		errorVec[newSortInd].indxs = errorVecMod[i].indxs;
	}	
	free(errorVecMod);
	free(winPtsCnt);
}

void RobustAlgebraicPlaneFitting(float* x, float* y, float* z, float* mP,
							unsigned int N, float topKthPerc, float bottomKthPerc, 
							float MSSE_LAMBDA, unsigned char stretch2CornersOpt) {
	float model[3];
	unsigned int i, iter, cnt;
	unsigned int sampleSize;
	float* residual;
	float* sample_x;
	float* sample_y;
	float* sample_z;
	struct sortStruct* errorVec;

	errorVec = (struct sortStruct*) malloc(N * sizeof(struct sortStruct));
	residual = (float*) malloc(N * sizeof(float));
	
	sampleSize = (unsigned int)(N*topKthPerc)- (unsigned int)(N*bottomKthPerc);
	sample_x = (float*) malloc(sampleSize * sizeof(float));
	sample_y = (float*) malloc(sampleSize * sizeof(float));
	sample_z = (float*) malloc(sampleSize * sizeof(float));

	model[0]=0;
	model[1]=0;
	model[2]=0;
	for(iter=0; iter<12; iter++) {
		
		for (i = 0; i < N; i++) {
			errorVec[i].vecData  = fabs(z[i] - (model[0]*x[i] + model[1]*y[i] + model[2]));	
			errorVec[i].indxs = i;
		}
		quickSort(errorVec,0,N-1);
		
		stretch2CornersFunc(errorVec, x, y, N, stretch2CornersOpt);
		
		cnt = 0;
		for(i=(int)(N*bottomKthPerc); i<(int)(N*topKthPerc); i++) {
			sample_x[cnt] = x[errorVec[i].indxs];
			sample_y[cnt] = y[errorVec[i].indxs];
			sample_z[cnt] = z[errorVec[i].indxs];
			cnt++;
		}
		TLS_AlgebraicPlaneFitting(sample_x, sample_y, sample_z, model, sampleSize);
	}

	mP[0] = model[0];
	mP[1] = model[1];
	mP[2] = model[2];
	
	for (i = 0; i < N; i++)
		residual[i]  = z[i] - (model[0]*x[i] + model[1]*y[i] + model[2]);
	
	mP[3] = MSSE(residual, N, MSSE_LAMBDA, (int)(N*topKthPerc));

	free(residual);
	free(errorVec);
	free(sample_x);
	free(sample_y);
	free(sample_z);
}

void RobustSingleGaussianTensor(float *inTensor, float *modelParamsMap, unsigned int N,
		unsigned int X, unsigned int Y, float topKthPerc, float bottomKthPerc, float MSSE_LAMBDA) {

	float* vec;
	vec = (float*) malloc(N * sizeof(float));
	float mP[2];
	unsigned int rCnt, cCnt, i;

	for(rCnt=0; rCnt<X; rCnt++) {
		for(cCnt=0; cCnt<Y; cCnt++) {
			
			for(i=0; i<N; i++)
				vec[i]=inTensor[cCnt + rCnt*Y + i*X*Y];
			
			RobustSingleGaussianVec(vec, mP, N, topKthPerc, bottomKthPerc, MSSE_LAMBDA);
			modelParamsMap[cCnt + rCnt*Y + 0*X*Y] = mP[0];
			modelParamsMap[cCnt + rCnt*Y + 1*X*Y] = mP[1];
		}
	}
}

void RSGImage(float* inImage, unsigned char* inMask, float *modelParamsMap,
				unsigned int winX, unsigned int winY,
				unsigned int X, unsigned int Y, 
				float topKthPerc, float bottomKthPerc, 
				float MSSE_LAMBDA, unsigned char stretch2CornersOpt) {

	float* x;
	x = (float*) malloc(winX*winY * sizeof(float));
	float* y;
	y = (float*) malloc(winX*winY * sizeof(float));
	float* z;
	z = (float*) malloc(winX*winY * sizeof(float));
	unsigned int rCnt, cCnt, numElem, pRowStart, pRowEnd, pClmStart, pClmEnd;
	float mP[4];
	pRowStart = 0;
	pRowEnd = winX;
	while(pRowStart<X) {
		
		pClmStart = 0;
		pClmEnd = winY;
		while(pClmStart<Y) {
			
			numElem = 0;
			for(rCnt = pRowStart; rCnt < pRowEnd; rCnt++)
				for(cCnt = pClmStart; cCnt < pClmEnd; cCnt++)
					if(inMask[cCnt + rCnt*Y]) {
						x[numElem] = rCnt;
						y[numElem] = cCnt;
						z[numElem] = inImage[cCnt + rCnt*Y];
						numElem++;
					}
			if((int) (bottomKthPerc*numElem)>12) {
				RobustAlgebraicPlaneFitting(x, y, z, mP, numElem, topKthPerc, 
											bottomKthPerc, MSSE_LAMBDA, stretch2CornersOpt);

				for(rCnt=pRowStart; rCnt<pRowEnd; rCnt++) {
					for(cCnt=pClmStart; cCnt<pClmEnd; cCnt++) {
						modelParamsMap[cCnt + rCnt*Y + 0*X*Y] = mP[0]*rCnt + mP[1]*cCnt + mP[2];
						modelParamsMap[cCnt + rCnt*Y + 1*X*Y] = mP[3];
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

void RSGImagesInTensor(float *inTensor, unsigned char* inMask, 
					float *modelParamsMap, unsigned int N,
					unsigned int X, unsigned int Y, float topKthPerc,
					float bottomKthPerc, float MSSE_LAMBDA, 
					unsigned char stretch2CornersOpt) {

	unsigned int rCnt, cCnt, i, numElem;
	float modelParams[4];

	float* x;
	x = (float*) malloc(X*Y * sizeof(float));
	float* y;
	y = (float*) malloc(X*Y * sizeof(float));
	float* z;
	z = (float*) malloc(X*Y * sizeof(float));

	for(i=0; i<N; i++) {
		
		numElem = 0;
		for(rCnt=0; rCnt<X; rCnt++)
			for(cCnt=0; cCnt<Y; cCnt++)
				if(inMask[cCnt + rCnt*Y]) {
					x[numElem] = rCnt;
					y[numElem] = cCnt;
					z[numElem] = inTensor[cCnt + rCnt*Y + i*X*Y];
					numElem++;
				}
		if((int) (bottomKthPerc*numElem)>12) {
			RobustAlgebraicPlaneFitting(x, y, z, modelParams, 
										numElem, topKthPerc, bottomKthPerc, MSSE_LAMBDA, stretch2CornersOpt);
			modelParamsMap[0*N + i] = modelParams[0];
			modelParamsMap[1*N + i] = modelParams[1];
			modelParamsMap[2*N + i] = modelParams[2];
			modelParamsMap[3*N + i] = modelParams[3];
		}
	}

	free(x);
	free(y);
	free(z);
}
