#!/usr/bin/env python

"""Tests for `RobustGaussianFittingLibrary` package."""

import pytest

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import RobustGaussianFittingLibrary as rgflib

@pytest.fixture
def response():
    """Sample pytest fixture.
    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')

def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
    
def test_locating_colorbar():
    fig = plt.figure()
    
    ax = fig.add_subplot(1, 2, 1)
    im = ax.imshow(np.random.rand(100, 100))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title('There is a uniform here')

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(np.random.randn(100, 100))
    ax.set_title('There is a Gaussian here')
    plt.show()

def test_RobustAlgebraicLineFitting():
    LWidth = 3
    font = {'weight' : 'bold',
            'size'   : 14}
    params = {'legend.fontsize': 'x-large',
              'axes.labelsize' : 'x-large',
              'axes.titlesize' : 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    
    plt.figure(figsize=(8, 8))
    
    plt.rc('font', **font)
    plt.rcParams.update(params)

    print('testing RobustAlgebraicLineFitting')
    inSigma = 0.15
    slope = 1
    intercept = 0
    print('slope: ', slope)
    print('intercept: ', intercept)
    print('Scale: ', inSigma)
    n_in = 45
    inX = 2*(np.random.rand(n_in)-0.5)
    inY = slope*inX + intercept + inSigma*np.random.randn(n_in)
    n_out = 5
    outX = 2*(np.random.rand(n_out)-0.5)
    outY = 3*(np.random.rand(n_out))
    X = np.concatenate((inX, outX))
    Y = np.concatenate((inY, outY))
    
    label = np.ones(X.shape[0], dtype='int8')
    _errors = Y - (slope*X + intercept)
    label[np.fabs(_errors) <= 3*inSigma] = 0
    label[-n_out:] = 1
    

    Xax = np.linspace(X.min(), X.max(), 100)
    print(f'X.min(): {X.min()}')
    print(f'X.max(): {X.max()}')
    print(X.shape)

    mP_T = rgflib.fitLine(X[label==0], Y[label==0], 1, 0)
    print(mP_T)
    Yax_U = mP_T[0]*Xax + mP_T[1] + 3*mP_T[2]
    Yax_M = mP_T[0]*Xax + mP_T[1]
    Yax_L = mP_T[0]*Xax + mP_T[1] - 3*mP_T[2]
    plt.plot(Xax, Yax_M, linewidth = 4, label = 'True model', color = 'blue')

    mP_notR = rgflib.fitLine(X, Y, 1, 0)
    print(mP_notR)
    Yax_U = mP_notR[0]*Xax + mP_notR[1] + 3*mP_notR[2]
    Yax_M = mP_notR[0]*Xax + mP_notR[1]
    Yax_L = mP_notR[0]*Xax + mP_notR[1] - 3*mP_notR[2]
    plt.plot(Xax, Yax_M, linewidth = 3, label = 'Non-robust fitting', color = 'green')

    mP_R = rgflib.fitLine(X, Y, 0.5, 0.3)
    print(mP_R)
    YaxR_U = mP_R[0]*Xax + mP_R[1] + 3*mP_R[2]
    YaxR_M = mP_R[0]*Xax + mP_R[1]
    YaxR_L = mP_R[0]*Xax + mP_R[1] - 3*mP_R[2]
    plt.plot(Xax, YaxR_M, linewidth = 3, label = 'Robust', color = 'red')

    plt.scatter(X[label==0], Y[label==0], color='blue', label='inliers', marker='o')
    plt.scatter(X[label==1], Y[label==1]/2, color='red', label='outliers', marker='o')
    
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.xlabel('X')
    plt.ylabel('Y').set_rotation(0)
    plt.ylabel('Y')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()
    
    rgflib.misc.naiveHistTwoColors(_errors, np.array([0, mP_R[2]]), figsize = (8,8))
    
if __name__ == '__main__':
    test_RobustAlgebraicLineFitting()