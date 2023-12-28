#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("readme.md", "r") as fh:
    long_description = fh.read()

__version__ = "0.4.1.1"

setup(name='eeglib',
      version = __version__,
      description='A library with some tools and functions for EEG signal analysis',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Luis Cabañero Gómez',
      author_email='Luis.Cabanero@uclm.es',
      url="https://github.com/Xiul109/eeglib",
      py_modules = ['eeglib'],
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',

          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',

          'License :: OSI Approved :: MIT License',

          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering',
      ],
      keywords='lib EEG signal analysis',

      packages=find_packages(exclude=["docs"]),

      install_requires = ['numpy','scipy',
                          'scikit-learn',
                          'numba',
                          'pandas',
                          'pyedflib',
                          'fastdtw',
                          'progressbar2'],

      test_require = ['colorednoise'],
      
      extras_require = {
          'doc': ["sphinx", "numpydoc"]
          }
)
