#!/usr/bin/env python

"""
------------------------------------------------------
This file is part of RobustGaussianFittingLibrary,
a free library WITHOUT ANY WARRANTY
Copyright: 2017-2020 LaTrobe University Melbourne,
           2019-2021 Deutsches Elektronen-Synchrotron
           2021-2023 Monash University
Authors: Alireza Sadri, Marjan Hadian Jazi
------------------------------------------------------
"""

from setuptools import setup
from setuptools import Extension

_version = '0.2.2'

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ['numpy', 'scipy', 'matplotlib', 'lognflow']

test_requirements = ['pytest>=3', ]

setup(
    author="Alireza Sadri",
    author_email='arsadri@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Science/Research',      # Define that your audience are developers
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhthon versions that you want to support
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
    install_requires=requirements,
    license='MIT',
    description = 'A library for robust Gaussian fitting '\
                + 'of geometric models in presence of outliers. ',   # Give a short description about your library
    long_description_content_type = 'text/markdown',
    long_description=readme,
    include_package_data=True,
    keywords=['rgflib', 
              'outlier', 
              'outlier detection', 
              'outlier removal', 
              'anamoly detection', 
              'curve fitting', 
              'line fitting', 
              'plane fitting', 
              'fit a Gaussian', 
              'Gaussian fitting'],   # Keywords that define your package best
    name='RobustGaussianFittingLibrary',
    packages = ['RobustGaussianFittingLibrary'],
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/arsadri/RobustGaussianFittingLibrary',
    version=_version,
    zip_safe=False,
    ext_modules=[Extension(name = 'RGFLib', 
                           sources = ['RobustGaussianFittingLibrary/RGFLib.c'],
                           language = 'c',
                           extra_compile_args = ['-shared'])],
)
