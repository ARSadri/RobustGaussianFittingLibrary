%uncomment for MEX compile and then comment again for MATLAB use
%mex RGFLib_fitValue_mex.c, return
clear
close all
numTests = 1000;
Rmean_rec=zeros(1,numTests);
for test = 1: numTests
	N = 10;
	inlierPerc = 0.75;
	topKthPerc = 0.5;
	bottomKthPerc = 0.4;
	MSSE_LAMBDA = 3.0;	%std away from mean is still a guassians.
	optIters = 12;

	Gaus_mean = 0;
	Gaus_std = 1;
	uniform_spread = 100;
	uniform_bias = 0;
	initModel = 0;

	inliers = Gaus_mean + Gaus_std*randn(1, floor(N*inlierPerc));
	inliers_W = rand(1, length(inliers));
	outliers = uniform_spread*(rand(1,floor(N*(1-inlierPerc)))-0.5)+uniform_bias;
	%outliers(abs(outliers-Gaus_mean)<Gaus_std*MSSE_LAMBDA)=[];
	outliers_W = 10*rand(1, length(outliers));
	inVec = [inliers outliers];
	inWeights = [inliers_W outliers_W];
	N = numel(inVec);
	if N<3
		inVec = [inVec 1230];
		inWeights = [inWeights 1];
	end
	N = numel(inVec);
	%maybe you would like to help the optimization a bit if you think median is an inlier??? no?
	%initModel = median(inVec)

	Rmean = RGFLib_fitValue_mex(inVec, inWeights, initModel, N, topKthPerc, bottomKthPerc, MSSE_LAMBDA, optIters);
	Rmean_rec(test) = Rmean;
end
disp(median(Rmean_rec))
return
hist(inVec, 20)
hold on
plot([Rmode, Rmode], [0, floor(N/10)], 'green')
plot([Rmean, Rmean], [0, floor(N/10)], 'red')

