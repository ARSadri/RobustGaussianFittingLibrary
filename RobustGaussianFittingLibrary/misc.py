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

def scatter3(mat, inFigure = None, returnFigure = False, label = None):
    """ given a matrix input of size 3 x N, it scatters it in 3D
    """
    LWidth = 3
    font = {
            'weight' : 'bold',
            'size'   : 8}
    params = {'legend.fontsize': 'x-large',
             'axes.labelsize': 'x-large',
             'axes.titlesize':'x-large',
             'xtick.labelsize':'x-large',
             'ytick.labelsize':'x-large'}
    plt.rc('font', **font)
    plt.rcParams.update(params)

    if(inFigure is None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        figureCnt = 0
    else:
        fig, ax, figureCnt = inFigure
        figureCnt += 1
    if(label is None):
        label = 'Plot ' + str(figureCnt)
    ax.scatter(mat[0], mat[1], mat[2], label = label)
    if(returnFigure):
        return((fig, ax, figureCnt))
    else:
        if(figureCnt>0):
            plt.legend()
        plt.show()

class plotGaussianGradient:
    def __init__(self, xlabel, ylabel, num_bars = 100, 
                 title = None, xmin = None, xmax = None, 
                 ymin = None, ymax = None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.num_bars = num_bars
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        LWidth = 1
        font = {
                'weight' : 'bold',
                'size'   : 14}
        plt.rc('font', **font)
        params = {'legend.fontsize': 'x-large',
                 'axes.labelsize': 'x-large',
                 'axes.titlesize':'x-large',
                 'xtick.labelsize':'x-large',
                 'ytick.labelsize':'x-large'}
        plt.rcParams.update(params)
        plt.figure(figsize=(8, 6), dpi=50)
        self.ax1 = plt.subplot(111)

    def addPlot(self, x, mu, std, gradient_color, label, snr = 3.0):

        for idx in range(self.num_bars-1):
            y1 = ((self.num_bars-idx)*mu + idx*(mu + snr*std))/self.num_bars
            y2 = y1 + snr*std/self.num_bars
            
            prob = np.exp(-(snr*idx/self.num_bars)**2/2)
            plt.fill_between(
                x, y1, y2, color = gradient_color, alpha = prob  )

            y1 = ((self.num_bars-idx)*mu + idx*(mu - snr*std))/self.num_bars
            y2 = y1 - snr*std/self.num_bars
            
            plt.fill_between(
                x, y1, y2, color = gradient_color, alpha = prob  )
        plt.plot(x, mu, linewidth = 3, color = gradient_color, label = label)
        
    def show(self, show_legend = True):
        if(self.xmin is not None) & (self.xmax is not None):
            plt.xlim([self.xmin, self.xmax])
        if(self.ymin is not None) & (self.ymax is not None):
            plt.ylim([self.ymin, self.ymax])
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        if(self.title is not None):
            plt.title(self.title)
        if(show_legend):
            plt.legend()
        plt.tight_layout()
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
    need to make the inputs argument to be an indexable object by inputes[0]
    to inputs[N].
    These N tuples each, will be the input given to a processor, which
    is the function you provide. The function provides an output and it
    will be given as a list of the ourputs with the same order as the input.
    You need to use the list as the list of outputs associated with the list of
    the inputs.
    parameters: they will not be indexed but will be sent to the function
        without change.
        
    A code snippet is brought here for the case of non numpy input output
    
    def multiprocessor_targetFunc(tuple_of_indexables, tuple_of_nonindexables):
        op_type = tuple_of_nonindexables
        data, mask = tuple_of_indexables
        if(op_type=='median'):
            to_return = np.median(data[mask==1])
        return(np.array([to_return]))
        
    def test_multiprocessor():
        N = 1000
        D = 100000
        Data = (10+100*np.random.randn(N,D)).astype('int')
        Mask = (2*np.random.rand(N,D)).astype('int')
        Param = 'median'
        
        tuple_of_indexables = (Data, Mask)
        tuple_of_nonindexables = (Param) 
        medians = RobustGaussianFittingLibrary.misc.multiprocessor(
            multiprocessor_targetFunc, 
            tuple_of_indexables, tuple_of_nonindexables).start()
        
    """    
    def __init__(self, 
        targetFunction,
        inputs, 
        parameters = None,
        max_cpu = None,
        batchSize = None,
        showProgress = False):
        """
        input arguments
        ~~~~~~~~~~~~~~~
            targetFunction: Target function
            inputs: an indexable list of tuples. Each tuple will be sent to
                the targetFunction, such as by syntax:
                    targetFunction(inputs[0])
                Indeed we actually do this syntax before anything. 
            parameters: they will not be indexed but will be sent to the function
                without change. default = None
            max_cpu: max number of allowed cpu
                default: None
            batchSize: how many data points are sent to each CPU at a time
                default: n_CPU/n_points/2
            showProgress: using textProgBar, it shows the progress of 
                multiprocessing of your task.
                default: False
        """
        if(type(inputs).__module__ == np.__name__):
            self.inputIsNumpy = True
            self.n_pts = inputs.shape[0]     
            if(showProgress):
                print('Input is a numpy ndArray with ', 
                      self.n_pts, ' data points')
        else:
            self.inputIsNumpy = False
            try:
                self.n_individualInputs = len(inputs)
                self.n_pts = inputs[0].shape[0]
                if(showProgress):
                    print('Input is not a numpy ndArray, yet with ', 
                          self.n_pts, ' data points')
            except:
                print('I can not get the length of the input list')
                print('we are supporting list or tuple of indexable objects')
                exit()

        if(not self.inputIsNumpy):
            inputs_first_tst = []
            for memberCnt in range(self.n_individualInputs):
                inputs_first_tst.append(inputs[memberCnt][0])
        else:
            inputs_first_tst = inputs[0]
            
        try:
            if(parameters is not None):
                funcOutput = targetFunction(inputs_first_tst, parameters)
            else:
                funcOutput = targetFunction(inputs_first_tst)
        except:
            print('I cannot call your function')
            print('Running the following syntax raised an exception:')
            if(parameters is not None):
                print('targetFunction(inputs[0], parameters)')
            else:
                print('targetFunction(inputs[0])')
            print('You need to make inputs mutable and indexable by axis 0')
            exit()
        if(showProgress):
            print('I could call your given function with first data point')

        self.outputIsNumpy = False
        if(type(funcOutput).__module__ == np.__name__):
            self.outputIsNumpy = True
            self.output_shape = funcOutput.shape
            self.output_dtype = funcOutput.dtype
            self.allResults = np.zeros(
                shape = ((self.n_pts,) + self.output_shape), 
                dtype = self.output_dtype)
            if(showProgress):
                print('output is a numpy ndArray')
                print('output_shape, ', self.output_shape)
                print('shape to prepare: ', ((self.n_pts,) + self.output_shape))
                print('allResults shape, ', self.allResults.shape)
        else:
            self.n_individualOutputs = len(funcOutput)
            
            self.allResults = []
            self.Q_procID = np.array([])
            if(showProgress):
                print('output is a list with ', 
                      self.n_individualOutputs, ' members.')
                output_types = []
            
        self.targetFunction = targetFunction
        self.inputs = inputs
        self.parameters = parameters
        self.showProgress = showProgress
        if(max_cpu is not None):
            self.max_cpu = max_cpu
        else:
            self.max_cpu = cpu_count()
        self.default_batchSize = int(np.ceil(self.n_pts/self.max_cpu/2))
        if(batchSize is not None):
            self.default_batchSize = batchSize
        if(showProgress):
            print('RGFLib multiprocessor initialized with:') 
            print('max_cpu: ', self.max_cpu)
            print('n_pts: ', self.n_pts)
            print('default_batchSize: ', self.default_batchSize)
        
    def _multiprocFunc(self, theQ, procID_range):
        if(self.outputIsNumpy):
            allResults = np.zeros(
                shape = ((procID_range.shape[0],) + self.output_shape), 
                dtype = self.output_dtype)
        else:
            allResults = []
        for idx, procCnt in enumerate(procID_range):
            if(not self.inputIsNumpy):
                inputs_single = []
                for memberCnt in range(self.n_individualInputs):
                    inputs_single.append(self.inputs[memberCnt][procCnt])
            else:
                inputs_single = self.inputs[procCnt]
            
            if(self.parameters is not None):
                results = self.targetFunction(inputs_single, self.parameters)
            else:
                results = self.targetFunction(inputs_single)
            if(self.outputIsNumpy):
                allResults[idx] = results
            else:
                allResults.append(results)
                
        theQ.put(list([procID_range, allResults]))
        
    def start(self):        
        aQ = Queue()
        numProc = self.n_pts
        procID = 0
        numProcessed = 0
        numBusyCores = 0
        firstProcess = True
        while(numProcessed<numProc):
            if (not aQ.empty()):
                aQElement = aQ.get()
                _batchSize = aQElement[0].shape[0]
                if(self.outputIsNumpy):
                    self.allResults[aQElement[0]] = aQElement[1]
                else:
                    self.Q_procID = np.concatenate((self.Q_procID, aQElement[0]))
                    self.allResults += aQElement[1]
                numProcessed += _batchSize
                numBusyCores -= 1
                if(self.showProgress):
                    if(firstProcess):
                        pBar = textProgBar(numProc-1, title = 'starting ' \
                            + str(numProc) + ' processes with ' \
                            + str(self.max_cpu) + ' CPUs')
                        firstProcess = False
                    else:
                        pBar.go(_batchSize)
    
            if((procID<numProc) & (numBusyCores < self.max_cpu)):
                batchSize = np.minimum(self.default_batchSize, numProc - procID)
                procID_arange = np.arange(procID, procID + batchSize, dtype = 'int')
                Process(target = self._multiprocFunc, 
                        args = (aQ, procID_arange)).start()
                procID += batchSize
                numBusyCores += 1
        if(self.showProgress):
            del pBar
        
        if(self.outputIsNumpy):
            return (self.allResults)
        else:
            sortArgs = np.argsort(self.Q_procID)
            ret_list = [self.allResults[i] for i in sortArgs]
            endResultList = []
            for memberCnt in range(self.n_individualOutputs):
                _currentList = []
                for ptCnt in range(self.n_pts):
                    _currentList.append(ret_list[ptCnt][memberCnt])
                endResultList.append(_currentList)
            for memberCnt in range(self.n_individualOutputs):
                if(type(endResultList[memberCnt][0]).__module__ == np.__name__):
                    endResultList[memberCnt] = \
                        np.vstack(endResultList[memberCnt])
            return (endResultList)

