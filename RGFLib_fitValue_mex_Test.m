clc
clear
close all
mex -g RGFLib_fitValue_mex.c

inVec = randn(1, 1000);
inVec = [inVec 10*(rand(1,300)+0.3)]

RGFLib_fitValue_mex(inVec, inVec.mean(), numel(inVec), 0.5, 0.4, 3.0, 12)