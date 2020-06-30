# Robust Gaussian Fitting Library #
A C Library for Robust Gaussian Fitting using geometric models in presence of outliers. This library is based on two main algorithms, FLKOS for finding the average of Gaussians, and MSSE for finding the scale.

In robust segmentation, the main assumtion is that the Gaussian we are looking for has the majority of data points. If it doesn't, it turns the problem into a clustering problem.

## Compilation into shared library
Run the following command to generate a shared .so library:
```
make
```
**Note**: if you are using windows, you can used mingwin and it has a make in its bin folder with a different name. Copy it and rename it to make.
## Usage from Python
A Python wrapper is also provided. Tha wrapper will be looking for the .so shared library file. The wrapper is in the file cWrapper.py and is used by other python files.

### importable libraries ###
* __RGFLib__: Basic functions can be found here for 1D and 2D data. Also for Tensors.
	* MSSE : Given set of residuals, it finds the scale of a gaussian
	* fitValue : Given a vector, it finds average and standard deviation of the gaussian.
	* fitValue2Skewed : Given a vector (and weights are accepted too), it finds the mode by (Median of inliers) and reports it along with a scale which is the distance of the mode from the edges of the Gaussian (by 3 STDs) divided by 3.
	* fitValueTensor : Given a tensor of size n_F, n_R, n_C, it finds the gaussian mean and std for each pixel in n_R and n_C.
	* fitLine : Given vectors X and Y, it finds three parameters describing a line by slope, intercept and scale of noise.
	* fitLineTensor : Given a tensor, it fits a line for each pixel
	* fitPlane : Given an image, returns four parameters of for algebraic fitting of a plane
	* fitBackground : Given an image, returns the mean and std of background at each pixel.
	* fitBackgroundTensor : Given a tensor of images n_F x n_R x n_C, returns the background mean and std for each pixel for each frame in n_F.

* __useMultiproc__: In this set, the tensor oporations are run by python Multiprocessing.
	* fitValueTensor_MultiProc : Does fitValueTensor using multiprocessing
	* fitLineTensor_MultiProc : Does fitLineTensor using multiprocessing
	* fitBackgroundTensor_multiproc : Does fitBackgroundTensor using multiprocessing

### Examples in Python ###
Many test funcitons are availble in the tests.py script. in the script, llok for the main function and choose one of them to run or Simply type in:
```
make test
```
## Usage from MATLAB ##
Currently, only the fitValue funciton is supported by a mex C code for MATLAB. However, you can request for more, or implement it yourself accordingly. Look at the ..._Test.m file
