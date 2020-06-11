clear
mex -g RGFLib_fitValue_mex.c

inVec = [randn(1, 1000) 10*rand(1,300)]

[Rmean, Rstd] = RGFLib_fitValue_mex(inVec, median(inVec), numel(inVec), 0.5, 0.4, 3.0, 12)
