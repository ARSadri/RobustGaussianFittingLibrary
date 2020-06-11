clc
clear
close all
mex -g RGFLib_fitValue_mex.c

inVec = randn(1, 1000);
inVec = [inVec 10*(rand(1,300)+0.3)]

[Rmean, Rstd] = RGFLib_fitValue_mex(inVec, median(inVec), numel(inVec), 0.5, 0.4, 3.0, 100)

return
[bins, edges] = hist(inVec, 40)

[~, indMax] = min(abs(edges-Rmean))

plot(edges, bins)
hold on
plot(edges, bins(indMax)*exp(((edges-Rmean).^2)./(2*Rstd.^2)),'r')