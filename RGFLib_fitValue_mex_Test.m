%uncomment for MEX compile and then comment again for MATLAB use
%mex -g RGFLib_fitValue_mex.c, return

N = 1000
topKthPerc = 0.5
bottomKthPerc = 0.45
MSSE_LAMBDA = 3.0	%std away from mean is still a guassians.
optIters = 12

Gaus_mean = 100
Gaus_std = 2
uniform_spread = 10
uniform_bias = 0
initModel = -10

inliers = Gaus_mean + Gaus_std*randn(1, N*0.6);
inliers_W = rand(1, length(inliers));
outliers = uniform_spread*(rand(1,N*(1-0.6))-0.5)+uniform_bias;
outliers(abs(outliers-Gaus_mean)<Gaus_std*MSSE_LAMBDA)=[];
outliers_W = 10*rand(1, length(outliers));
inVec = [inliers outliers];
N = numel(inVec)

%maybe you would like to help the optimization a bit if you think median is an inlier??? no?
%initModel = median(inVec)

inWeights = 1+0*[inliers_W outliers_W];

[Rmean, RStd] = RGFLib_fitValue_mex(inVec, inWeights, initModel, N, topKthPerc, bottomKthPerc, MSSE_LAMBDA, optIters)
