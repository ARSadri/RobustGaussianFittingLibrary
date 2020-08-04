%#################################################################################################
%# This file is part of RobustGaussianFittingLibrary, a free library WITHOUT ANY WARRANTY        # 
%# Copyright: 2019-2020 Deutsches Elektronen-Synchrotron                                         # 
%#################################################################################################
%uncomment for MEX compile and then comment again for MATLAB use
mex RGFLib_mex_fitValue2Skewed.c, return

numTests = 1000;
Rmode_rec=zeros(1,numTests);
for test = 1: numTests
	N = 60;
	inlierPerc = 0.75;
	topKthPerc = 0.5;
	bottomKthPerc = 0.4;
	MSSE_LAMBDA = 3.0;	%std away from mean is still a guassians.
	optIters = 12;
	
	Gaus_mean = 0;
	Gaus_std = 1;
	uniform_spread = 3;
	uniform_bias = 1.5;
	initModel = 0;
	
	inliers = Gaus_mean + Gaus_std*randn(1, floor(N*inlierPerc));
	inliers_W = rand(1, length(inliers));
	outliers = uniform_spread*(rand(1,floor(N*(1-inlierPerc)))-0.5)+uniform_bias;
	outliers_W = rand(1, length(outliers));
	inVec = [inliers outliers];
	inWeights = [inliers_W outliers_W];
	N = numel(inVec);

	Rmode = RGFLib_mex_fitValue2Skewed(inVec, inWeights, initModel, N, topKthPerc, bottomKthPerc, MSSE_LAMBDA, optIters);
	Rmode_rec(test) = Rmode;
end
disp(median(Rmode_rec))
