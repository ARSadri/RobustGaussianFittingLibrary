clear

%uncomment for MEX compile and then comment again for MATLAB use
%mex -g RGFLib_fitValue_mex.c, return


N = 1000
topKthPerc = 0.5
bottomKthPerc = 0.45
MSSE_LAMBDA = 3.0
optIters = 12

Gaus_mean = 43232
Gaus_std = 1912
uniform_spread = 1232312
uniform_bias = 75432
initModel = 87332111

inliers = Gaus_mean + Gaus_std*randn(1, N*topKthPerc);
outliers = uniform_spread*(rand(1,N*(1-topKthPerc))-0.5)+uniform_bias;
outliers(abs(outliers-Gaus_mean)<Gaus_std*MSSE_LAMBDA)=[];
inVec = [inliers outliers];
N = numel(inVec)

%maybe you would like to help the optimization a bit if you think median is an inlier??? no?
%initModel = median(inVec)

[Rmean, Rstd] = RGFLib_fitValue_mex(inVec, initModel, N, topKthPerc, bottomKthPerc, MSSE_LAMBDA, optIters)
