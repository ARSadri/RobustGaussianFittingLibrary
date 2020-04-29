# RGFLib: Robust Gaussian Fitting Library
A C Library for Robust Gaussian Fitting using geometric models in presence of outliers. This library is based on two main algorithms, FLKOS for finding the average of Gaussians, and MSSE for finding the scale.

In robust segmentation, the main assumtion is that the Gaussian we are looking for has the majority of data points. If it doesn't, it turns the problem into a clustering problem.

# Compilation into shared library
Run the following command to generate a shared .so library:
```
make
```
**Note**: using the first line of the C file also compiles the library.
# Usage from Python
A Python wrapper is also provided. Tha wrapper will be looking for the .so shared library file.

# Examples in Python 
Many test funcitons are availble in the test script. in the script choose one of them to run in the main function and Simply type in:
```
make test
```
