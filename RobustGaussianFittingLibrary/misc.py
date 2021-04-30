"""
------------------------------------------------------
This file is part of RobustGaussianFittingLibrary,
a free library WITHOUT ANY WARRANTY
Copyright: 2017-2020 LaTrobe University Melbourne,
           2019-2020 Deutsches Elektronen-Synchrotron
------------------------------------------------------
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from .cWrapper import RGFCLib
from multiprocessing import Process, Queue, cpu_count
from scipy.spatial.transform import Rotation as R

class textProgBar:
    """
    While I respect all the codes and methods on web that use \r for this task,
    there are cases such as simple ssh terminals that do not support \r.
    In such cases, if something is written at the end of the line it cannot be deleted.
    The following code provides a good looking simple title for the progress bar
    and shows the limits and is very simple to use.
    define the object with length of the for loop, call its go funciton 
    every time in the loop and delete the object afterwards.
    """
    def __init__(self, length, numTicks = 70 , title = 'progress'):
        self.startTime = time.time()
        self.ck = 0
        self.prog = 0
        self.length = length
        if(numTicks < len(title) + 2 ):
            self.numTicks = len(title)+2
        else:
            self.numTicks = numTicks
        print(' ', end='')
        for _ in range(self.numTicks):
            print('_', end='')
        print(' ', flush = True)

        print('/', end='')
        for idx in range(self.numTicks - len(title)):
            if(idx==int((self.numTicks - len(title))/2)):
                print(title, end='')
            else:
                print(' ', end='')
        print(' \\')
        print(' ', end='')
    def go(self, ck=1):
        self.ck += ck
        cProg = int(self.numTicks*self.ck/self.length/3)    #3: because 3 charachters are used
        while (self.prog < cProg):
            self.prog += 1
            remTimeS = self.startTime + \
                       (time.time() - self.startTime)/(self.ck/self.length) - time.time()
            time_correct = 2-2*(self.ck/self.length)
            #remTimeS *= time_correct
            if(remTimeS>=5940):
                progStr = "%02d" % int(np.ceil(remTimeS/3600))
                print(progStr, end='')
                print('h', end='', flush = True)
            elif(remTimeS>=99):
                progStr = "%02d" % int(np.ceil(remTimeS/60))
                print(progStr, end='')
                print('m', end='', flush = True)
            else:
                progStr = "%02d" % int(np.ceil(remTimeS))
                print(progStr, end='')
                print('s', end='', flush = True)
    
    def __del__(self):
        print('\n ', end='')
        for _ in range(self.numTicks):
            print('~', end='')
        print(' ', flush = True)

def PDF2Uniform(inVec, inMask=None, numBins=10, 
                nUniPoints=None, lowPercentile = 0, 
                highPercentile=100, showProgress = False):
    """
    This function takes an array of numbers and returns indices of those who
    form a uniform density over numBins bins, between lowPercentile and highPercentile
    values in the array, and not masked by 0. The output vector will have indices with the size nUniPoints.
    It is possible to only provide the vector with no parameter. I will calculate the maximum
    number of elements it can, as long as it still provides a uniform density.
    """
    n_pts = inVec.shape[0]
    if((highPercentile - lowPercentile)*n_pts < nUniPoints):
        lowPercentile = 0
        highPercentile=100
    indPerBin = np.digitize(inVec, np.linspace(np.percentile(inVec, lowPercentile),
                                          np.percentile(inVec, highPercentile), 
                                          numBins) )
    binValue, counts = np.unique(indPerBin, return_counts = True)
    counts = counts[counts>0]
    if(nUniPoints is None):
        nUniPoints = int(numBins*np.median(counts))

    outIndicator = binValue.min()-1
    indPerBin[inVec < np.percentile(inVec, lowPercentile)] = outIndicator
    indPerBin[inVec > np.percentile(inVec, highPercentile)] = outIndicator
    if(inMask is not None):
        indPerBin[inMask==0] = outIndicator
    nUniPoints = np.minimum(nUniPoints, (indPerBin != outIndicator).sum())
    uniInds = np.zeros(nUniPoints, dtype='uint32')
    ptCnt = 0
    if(showProgress):
        pBar = textProgBar(nUniPoints)
    while(ptCnt < nUniPoints):
        for bin in binValue:
            if(ptCnt >= nUniPoints):
                break
            lclInds = np.where(indPerBin==bin)[0]
            if(lclInds.shape[0]>0):
                uniInds[ptCnt] = np.random.choice(lclInds,1)
                indPerBin[uniInds[ptCnt]] = outIndicator
                ptCnt += 1
                if(showProgress):
                    pBar.go()            
    if(showProgress):
        del pBar
    return(uniInds)

def removeIslands(inMask, minSize = 1):
    """small islands of 0s in the middle of 1s will turn into 1s
    Given a mask in the input where the good pixels are marked by zero and bad areas are 1,
    the output will not have islands of zeros with size less or equal to minSize,
    sorrounded by 1s. Notice that if the island has diagonal routes to other islands, it will
    still be an isolated island. If it has a vertical or horizontal route to other good areas,
    it is not an isolated island.
    Input arguments
    ~~~~~~~~~~~~~~~
        inMask : 2D array, numbers are either 0: good and 1: bad.
    Output
    ~~~~~~    
        same size as input, some of the pixels that were 0s in the input are now 1s if they were on lonly islands of good pixels surrounded by bad 1s.
    """
    outMask = np.zeros(inMask.shape, dtype='uint8')
    RGFCLib.islandRemoval(1 - inMask.astype('uint8'),
                          outMask,
                          inMask.shape[0],
                          inMask.shape[1],
                          minSize)
    return(outMask + inMask)      
    
def naiveHist(vec, mP):
    plt.figure(figsize=[10,8])
    hist,bin_edges = np.histogram(vec, 100)
    plt.bar(bin_edges[:-1], hist, width = mP[1], color='#0504aa',alpha=0.7)
    x = np.linspace(vec.min(), vec.max(), 1000)
    y = hist.max() * np.exp(-(x-mP[0])*(x-mP[0])/(2*mP[1]*mP[1]))
    plt.plot(x,y, 'r')
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value',fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.title('Normal Distribution Histogram',fontsize=15)
    plt.show()

def naiveHist_multi_mP(vec, mP):
    plt.figure(figsize=[10,8])
    hist,bin_edges = np.histogram(vec, 100)
    plt.bar(bin_edges[:-1], hist, width = 3, color='#0504aa',alpha=0.7)
    x = np.linspace(vec.min(), vec.max(), 1000)
    for modelCnt in range(mP.shape[1]):
        yMax = hist[np.fabs(bin_edges[:-1]-mP[0, modelCnt]) < 3 * mP[1, modelCnt]].max()
        y = yMax * np.exp(-(x-mP[0, modelCnt])*(x-mP[0, modelCnt])/(2*mP[1, modelCnt]*mP[1, modelCnt]))
        plt.plot(x,y, 'r')
    plt.show()

def naiveHistTwoColors(inVec, mP, SNR_ACCEPT=3.0):
    LWidth = 3
    font = {
            'weight' : 'bold',
            'size'   : 8}
    params = {'legend.fontsize': 'x-large',
             'axes.labelsize': 'x-large',
             'axes.titlesize':'x-large',
             'xtick.labelsize':'x-large',
             'ytick.labelsize':'x-large'}


    tmpL  = (inVec[  (inVec<=mP[0]-SNR_ACCEPT*mP[1]) & (inVec>=mP[0]-4*SNR_ACCEPT*mP[1])  ]).copy()
    tmpM  = (inVec[(inVec>mP[0]-SNR_ACCEPT*mP[1]) & (inVec<mP[0]+SNR_ACCEPT*mP[1])]).copy()
    tmpH  = (inVec[  (inVec>=mP[0]+SNR_ACCEPT*mP[1]) & (inVec<=mP[0]+4*SNR_ACCEPT*mP[1]) ]).copy()
    _xlimMin = tmpM.min()
    _xlimMax = tmpM.max()

    plt.figure()
    plt.rc('font', **font)
    plt.rcParams.update(params)

    if (tmpL.any()):
        hist,bin_edges = np.histogram(tmpL, tmpL.shape[0])
        plt.bar(bin_edges[:-1], hist, width = tmpM.std()/SNR_ACCEPT, color='royalblue',alpha=0.5)
        _xlimMin = tmpL.min()
    hist,bin_edges = np.histogram(tmpM, 20)
    tmpMmax = hist.max()
    plt.bar(bin_edges[:-1], hist, width = tmpM.std()/SNR_ACCEPT, color='blue',alpha=0.5)
    if (tmpH.any()):
        hist,bin_edges = np.histogram(tmpH, tmpH.shape[0])
        plt.bar(bin_edges[:-1], hist, width = tmpM.std()/SNR_ACCEPT, color='royalblue',alpha=0.5)
        _xlimMax = tmpH.max()
    x = np.linspace(mP[0]-SNR_ACCEPT*mP[1], mP[0]+SNR_ACCEPT*mP[1], 1000)
    y = tmpMmax * np.exp(-(x-mP[0])*(x-mP[0])/(2*mP[1]*mP[1])) 
    
    plt.plot(np.array([mP[0] - SNR_ACCEPT*mP[1], mP[0] - SNR_ACCEPT*mP[1]]) ,
             np.array([0, tmpMmax]), linewidth = LWidth, color = 'm')
    plt.plot(np.array([mP[0] - 0*SNR_ACCEPT*mP[1], mP[0] - 0*SNR_ACCEPT*mP[1]]) ,
             np.array([0, tmpMmax]), linewidth = LWidth, color = 'g')
    plt.plot(np.array([mP[0] + SNR_ACCEPT*mP[1], mP[0] + SNR_ACCEPT*mP[1]]) ,
             np.array([0, tmpMmax]), linewidth = LWidth, color = 'r')
    plt.plot(x,y, 'orange', linewidth = LWidth)

    
    plt.xlim(_xlimMin, _xlimMax)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Fitting error of line to points',fontsize=15)
    plt.ylabel('Histogram',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

def sGHist(inVec, mP, SNR_ACCEPT=3.0):
    tmpL  = (inVec[  (inVec<=mP[0]-SNR_ACCEPT*mP[1]) & (inVec>=mP[0]-4*SNR_ACCEPT*mP[1])  ]).copy()
    tmpM  = (inVec[(inVec>mP[0]-SNR_ACCEPT*mP[1]) & (inVec<mP[0]+SNR_ACCEPT*mP[1])]).copy()
    tmpH  = (inVec[  (inVec>=mP[0]+SNR_ACCEPT*mP[1]) & (inVec<=mP[0]+4*SNR_ACCEPT*mP[1]) ]).copy()
    _xlimMin = tmpM.min()
    _xlimMax = tmpM.max()
    plt.figure()
    if (tmpL.any()):
        hist,bin_edges = np.histogram(tmpL, tmpL.shape[0])
        plt.bar(bin_edges[:-1], hist, width = 0.1*tmpL.std()/SNR_ACCEPT, color='b',alpha=0.5)
        _xlimMin = tmpL.min()
    hist,bin_edges = np.histogram(tmpM, 40)
    tmpMmax = hist.max()
    plt.bar(bin_edges[:-1], hist, width = 0.5*tmpM.std()/SNR_ACCEPT, color='g',alpha=0.5)
    if (tmpH.any()):
        hist,bin_edges = np.histogram(tmpH, tmpH.shape[0])
        plt.bar(bin_edges[:-1], hist, width = 0.1*tmpH.std()/SNR_ACCEPT, color='r',alpha=0.5)
        _xlimMax = tmpH.max()
    x = np.linspace(mP[0]-SNR_ACCEPT*mP[1], mP[0]+SNR_ACCEPT*mP[1], 1000)
    y = tmpMmax * np.exp(-(x-mP[0])*(x-mP[0])/(2*mP[1]*mP[1])) 
    plt.plot(x,y, 'm')
    plt.xlim(_xlimMin, _xlimMax)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value',fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.title('Normal Distribution Histogram',fontsize=15)
    plt.show()

def sGHist_multi_mP(inVec, mP, SNR=3.0):
    numModels = mP.shape[1]
    flag = np.zeros(inVec.size)
    plt.figure()
    for mCnt in range(numModels):
        flag[(inVec>=mP[0, mCnt]-SNR*mP[1, mCnt]) & (inVec<=mP[0, mCnt]+SNR*mP[1, mCnt])] = mCnt + 1
        modelVec = inVec[flag == mCnt + 1].copy()
        hist,bin_edges = np.histogram(modelVec, 40)
        tmpMmax = hist.max()
        plt.bar(bin_edges[:-1], hist, width = mP[1, mCnt]/SNR,alpha=0.5)
        x = np.linspace(mP[0, mCnt]-SNR*mP[1, mCnt], mP[0, mCnt]+SNR*mP[1, mCnt], 1000)
        y = tmpMmax * np.exp(-((x-mP[0, mCnt])**2)/(2*mP[1, mCnt]**2)) 
        plt.plot(x,y)
    
    modelVec = inVec[flag == 0]
    #hist,bin_edges = np.histogram(modelVec, modelVec.shape[0])
    #plt.bar(bin_edges[:-1], hist, width =, color='g',alpha=0.5)
    plt.bar(modelVec, np.ones(modelVec.size), color='g',alpha=0.5)
    plt.show()

def scatter3(mat, inFigure = None, returnFigure = False):
    """ given a matrix input of size 3 x N, it scatters it in 3D
    """
    if(inFigure is None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        figureCnt = 0
    else:
        fig, ax, figureCnt = inFigure
        figureCnt += 1
    ax.scatter(mat[0], mat[1], mat[2], label = 'Plot ' + str(figureCnt))
    if(returnFigure):
        return((fig, ax, figureCnt))
    else:
        if(figureCnt>0):
            plt.legend()
        plt.show()
    
def getTriangularVertices(n,
                          rotationAngles = [0, 0, 0],
                          phi_start = 0,
                          phi_end = np.pi,
                          plotIt = False):
    """ Triangular approximation of a sphere
    Two angular sweeps are necessary. One is theta that goes around a circle
        and the other is phi that moves the circle from 0 to 180 degree back
        and forth.
    input argument
    ~~~~~~~~~~~~~~
    n : number of vertices on the sphere
    rotationAngles: input must be a numpy array of three values
         in radian (0~2pi) and the output will rotate according 
         to these angles around axes xyz.
        default:[0, 0, 0]
    phi_start: 
        default: 0
    phi_end: 
        default: pi
    output
    ~~~~~~
    numpy array of shape 3 x n, each coloumn vector is
        x,y,z location of the vertex on a sphere
        all norms are one.
    """
    goldenRatio = (1 + 5**0.5)/2
    i = np.arange(0, n)
    theta = (2*np.pi / goldenRatio) * i 
    phi = np.arccos(np.linspace(np.cos(phi_end), np.cos(phi_start), n))
    triVecs = np.array([np.cos(theta) * np.sin(phi), 
                        np.sin(theta) * np.sin(phi), 
                        np.cos(phi)])
    rotMat = R.from_euler('xyz', rotationAngles)
    triVecs = rotMat.as_matrix() @ triVecs

    if(plotIt):
        scatter3(triVecs)    

    return(triVecs)

class multiprocessor:
    """ multiprocessor makes the use of multiprocessing in Python easy
    You need a function that can interpret the input arguments. Then you
    need to make the inputs argument to be an indexable list of tuples.
    These N tuples each, will be the input given to a processor, which
    is the function you provide. The function provides an output and it
    will be given as a list of the ourputs with the same order as the input.
    You need to use the list as the list of outputs associated with the list of
    the inputs.
    """    
    def __init__(self, 
        targetFunction,
        inputs,
        outputIsNumpy = False,
        max_cpu = None,
        showProgress = False):
        """
        input arguments
        ~~~~~~~~~~~~~~~
            targetFunction: Target function
            inputs: an indexable list of tuples. Each tuple will be sent to
                the targetFunction, such as by syntax:
                    targetFunction(inputs[0])
                Indeed we actually do this syntax before anything. 
            outputIsNumpy: If outputIsNumpy is True, then a numpy array will
                be allocated and filled in with outputs of the queue. This 
                only works if the output of the targetFunction is numpy array.
                notice that it cannot be a number, if it isa number you
                must set the return of the targetFunction to np.array([output])
                default: False
            max_cpu: max number of allowed cpu
                default: None
            showProgress: using textProgBar, it shows the progress of 
                multiprocessing of your task.
                default: False
        """
        try:
            targetFunction(inputs[0])
        except:
            print('Running the following syntax raised an exception:')
            print('targetFunction(inputs[0])')
            print('I cannot call your function with input[0]')
            print('You need to make inputs mutable and indexable by axis 0')
            exit()

        self.targetFunction = targetFunction
        self.inputs = inputs
        self.outputIsNumpy = outputIsNumpy
        self.showProgress = showProgress
        self.max_cpu = max_cpu
    
    def _multiprocFunc(self, aQ, _procID, inputArgs):
        result = self.targetFunction(inputArgs)
        aQ.put(list([_procID, result]))

    def __call__(self):
        #for example allData is size N by D
        try:
            N = len(self.inputs)
        except:
            N = self.inputs.shape[0]
        
        myCPUCount = cpu_count()-1
        if(self.max_cpu is not None):
            myCPUCount = np.minimum(self.max_cpu, myCPUCount)
        aQ = Queue()
        numProc = N
        procID = 0
        numProcessed = 0
        numBusyCores = 0
        if(not self.outputIsNumpy):
            allResults = []
        Q_procID = np.zeros(N, dtype='uint')
        while(numProcessed<numProc):
            if (not aQ.empty()):
                aQElement = aQ.get()
                Q_procID[numProcessed] = aQElement[0]
                if(not self.outputIsNumpy):
                    allResults.append(aQElement[1])
                else:
                    if(numProcessed == 0):
                        allResults = np.zeros(
                            shape = ((N,) + aQElement[1].shape), 
                            dtype = aQElement[1].dtype)
                    else:
                        allResults[aQElement[0]] = aQElement[1] 
                numProcessed += 1
                numBusyCores -= 1
                if(self.showProgress):
                    if(numProcessed == 1):
                        pBar = textProgBar(numProc-1, title = 'starting ' \
                            + str(numProc) + ' processes with ' \
                            + str(myCPUCount) + ' CPUs')
                    if(numProcessed > 1):
                        pBar.go()
    
            if((procID<numProc) & (numBusyCores < myCPUCount)):
                Process(target = self._multiprocFunc, args = (aQ, procID, 
                                           self.inputs[procID])).start()
                procID += 1
                numBusyCores += 1
        if(self.showProgress):
            del pBar
        
        if(not self.outputIsNumpy):
            sortInds = np.argsort(Q_procID)
            _output = []
            for cnt in sortInds:
                _output.append(allResults[cnt])
            return (_output)
        else:
            return (allResults)


