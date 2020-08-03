from distutils.core import setup, Extension
setup(
  name = 'RobustGaussianFittingLibrary',         # How you named your package folder (MyLib)
  packages = ['RobustGaussianFittingLibrary'],   # Chose the same as "name"
  version = '0.1.2',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A library for robust Gaussian fitting using geometric models in presence of outliers. ',   # Give a short description about your library
  author = 'Alireza Sadri',                   # Type in your name
  author_email = 'ARSadri@domain.com',      # Type in your E-Mail
  url = 'https://github.com/ARSadri/RobustGaussianFittingLibrary',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/ARSadri/RobustGaussianFittingLibrary/archive/v0.1.2.tar.gz',    # I explain this later on
  keywords = ['outlier', 'outlier detection', 'outlier removal', 'anamoly detection', 'curve fitting', 'line fitting', 'plane fitting', 'fit a Gaussian', 'Gaussian fitting'],   # Keywords that define your package best
  install_requires=[ 
          'numpy',
          'matplotlib',
          'scipy'
      ],
  #cmdclass={'install': 'make all'},
  ext_modules=[Extension(name = 'RGFLib', 
                         sources = ['RGFLib.c'],
                         language = 'c',
                         extra_compile_args = ['-shared'])],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Science/Research',      # Define that your audience are developers
    'Topic :: Scientific/Engineering :: Mathematics',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
