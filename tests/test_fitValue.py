#!/usr/bin/env python

"""Tests for `RobustGaussianFittingLibrary` package."""

import pytest

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import RobustGaussianFittingLibrary as rgflib

def test_fitValue():
    LWidth = 3
    font = {'weight' : 'bold',
            'size'   : 14}
    params = {'legend.fontsize': 'x-large',
              'axes.labelsize' : 'x-large',
              'axes.titlesize' : 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    plt.rc('font', **font)
    plt.rcParams.update(params)

    print('testing test_fitValue')
    
    n_inlliers = 70
    n_outliers = 30
    inSigma = 0.5
    RNN0 = inSigma*np.random.randn(n_inlliers)
    RNN1 = 10*(np.random.rand(n_outliers)-0.25)
    testData = np.concatenate((RNN0, RNN1)).flatten()
    data_indices = np.arange(testData.shape[0], dtype='int32')
    n_pts = data_indices.shape[0]
    
    np.random.shuffle(data_indices)
    testData = testData[data_indices]
    label = np.ones(n_pts, dtype='int8')
    label[np.fabs(testData) <= 3*inSigma] = 0
    mP_R = rgflib.fitValue(testData, 
                         fit2Skewed = True,
                         likelyRatio = 0.5, 
                         certainRatio=0.35, 
                         MSSE_LAMBDA=3.0)
    print(mP_R)
    mP_nR = rgflib.fitValue(testData, 
                         fit2Skewed = False,
                         likelyRatio = 1, 
                         certainRatio=0, 
                         MSSE_LAMBDA=3.0)
    print(mP_nR)

    plt.figure(figsize = (10,8))

    plt.scatter(np.arange(n_pts)[label==0], testData[label==0], color='blue', label='inliers', marker='o')
    plt.scatter(np.arange(n_pts)[label==1], testData[label==1], color='red', label='outliers', marker='*')

    plt.plot(np.array([0, testData.shape[0]]), np.array([0, 0]), linewidth = 4, label = 'True model', color = 'blue')
    plt.plot(np.array([0, testData.shape[0]]), np.array([mP_R[0], mP_R[0]]), linewidth = 3, label = 'Robust', color = 'red')
    plt.plot(np.array([0, testData.shape[0]]), np.array([mP_nR[0], mP_nR[0]]), linewidth = 3, label = 'Non-robust', color = 'green')

    plt.xlabel('index')
    plt.ylabel('Value')
    plt.grid()
    plt.legend(loc='upper left')
    plt.show()
    
    ############################### Histogram ##############################3

    inVec = testData
    mP = mP_R
    SNR_ACCEPT = 3
    
    tmpL  = (inVec[  (inVec<=mP[0]-SNR_ACCEPT*mP[1]) & 
                   (inVec>=mP[0]-4*SNR_ACCEPT*mP[1])  ]).copy()
    tmpM  = (inVec[(inVec>mP[0]-SNR_ACCEPT*mP[1]) & 
                   (inVec<mP[0]+SNR_ACCEPT*mP[1])]).copy()
    tmpH  = (inVec[  (inVec>=mP[0]+SNR_ACCEPT*mP[1]) & 
                   (inVec<=mP[0]+4*SNR_ACCEPT*mP[1]) ]).copy()
    _xlimMin = tmpM.min()
    _xlimMax = tmpM.max()

    plt.figure(figsize = (10,8))
    plt.rc('font', **font)
    plt.rcParams.update(params)

    if (tmpL.any()):
        hist,bin_edges = np.histogram(tmpL, tmpL.shape[0])
        plt.bar(bin_edges[:-1], hist, 
                width = tmpM.std()/SNR_ACCEPT, color='royalblue',alpha=0.5)
        _xlimMin = tmpL.min()
    hist,bin_edges = np.histogram(tmpM, 20)
    tmpMmax = hist.max()
    plt.bar(bin_edges[:-1], hist, 
            width = tmpM.std()/SNR_ACCEPT, color='blue',alpha=0.5)
    if (tmpH.any()):
        hist,bin_edges = np.histogram(tmpH, tmpH.shape[0])
        plt.bar(bin_edges[:-1], hist, 
                width = tmpM.std()/SNR_ACCEPT, color='red',alpha=0.5)
        _xlimMax = tmpH.max()
    
    n_x_steps = 1000
    x = np.linspace(testData.min(), testData.max(), n_x_steps)
    y_true = tmpMmax * np.exp(-x**2/2) 
    
    robust_cost = np.zeros(n_x_steps)
    non_robust_cost = np.zeros(n_x_steps)
    for cnt, mP0 in enumerate(x):
        _err = np.abs(testData - mP0)
        sorted_err = np.sort(_err)
        robust_cost[cnt] = sorted_err[int(0.5*n_pts)]
        non_robust_cost[cnt] = ((_err**2).mean())**0.5
    
    
    plt.plot(x, y_true, linewidth = 4, label = 'True density', color = 'blue')
    plt.plot(x, robust_cost, linewidth = 3, label = 'Robust cost function', color = 'red')
    plt.plot(x, non_robust_cost, linewidth = 3, label = 'Non-robust cost function', color = 'green')
    
    plt.plot(np.array([0, 0]),
             np.array([0, tmpMmax]), linewidth = 4, color = 'blue')
    plt.plot(np.array([mP_R[0], mP_R[0]]),
             np.array([0, tmpMmax]), linewidth = 3, color = 'red')
    plt.plot(np.array([mP_nR[0], mP_nR[0]]),
             np.array([0, tmpMmax]), linewidth = 3, color = 'green')
    
    plt.xlim(_xlimMin, _xlimMax)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Model parameter')
    plt.ylabel('Histogram')
    plt.xticks()
    plt.yticks()
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()
    
if __name__ == '__main__':
    test_fitValue()
