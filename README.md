# Robust Gaussian Fitting Library #
A C Library for Robust Gaussian Fitting using geometric models in presence of outliers. This library is based on two main algorithms, FLKOS for finding the average of Gaussians, and MSSE for finding the scale.

In robust segmentation, the main assumtion is that the Gaussian we are looking for has the majority of data points. If it doesn't, it turns the problem into a clustering problem.

# Compilation into shared library #
Run the following command to generate a shared .so library:
```
make
```
**Note**: using the first line of the C file also compiles the library.
# Usage from Python #
A Python wrapper is also provided. Tha wrapper will be looking for the .so shared library file.

## importable libraries ##
* __RobustGausFitLibPy__: In this file, you can find the basic funciton of robust gaussian fitting which include:
	* MSSEPy : Given set of residuals, it find the scale of a gaussian
	* RobustSingleGaussianVecPy : Given a vector, it finds average and standard deviation of the gaussian.
	* RobustSingleGaussianTensorPy : Given a tensor of size N_frames, N_rows, N_clms, it finds the gaussian mean and std for each pixel in N_rows and N_clms.
	* RobustAlgebraicLineFittingPy : Given vectors X and Y, it finds three parameters describing a line by slope, intercept and scale of noise.
	* RobustAlgebraicLineFittingTensorPy : Given a tensor, it fits a line for each pixel
	* RobustAlgebraicPlaneFittingPy : Given an image, returns four parameters of a fit planethe
	* RMGImagePy : Given an image returns the mean and std of background at each pixel.
	* RSGImage_by_Image_TensorPy : Given a tensor of images N_f x N_r x N_c, returns the background mean and std for each pixel for each frame in N_f.

* __RobustGausFitLibMultiprocPy__: In this file, there are two important funcitons:
	* RobustSingleGaussiansTensorPy_MultiProc : Does RobustSingleGaussianTensorPy using multiprocessing over segmented Tensor
	* RSGImage_by_Image_TensorPy_multiproc : Does RSGImage_by_Image_TensorPy using multiprocessing over segmented Tensor.

## Examples in Python ##
Many test funcitons are availble in the test script. in the script choose one of them to run in the main function and Simply type in:
```
make test
```
