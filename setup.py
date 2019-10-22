#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

__version__ = "0.3.3"

setup(name='eeglib',
      version = __version__,
      description='A library with some tools and functions for EEG signal analysis',
      long_description=open('readme.md').read(),
      author='Luis Cabañero Gómez',
      author_email='luiscabanerogomezxcr@hotmail.com',
#      url='',
      py_modules = ['eeglib'],
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',

          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',

          'License :: OSI Approved :: MIT',

          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering::EEG',
      ],
      keywords='lib EEG signal analysis',

      packages=find_packages(exclude=["docs"]),

      install_requires = ['numpy','scipy',
                          'sklearn',
                          'numba',
                          'pandas',
                          'pyedflib',
                          'fastdtw'],
      
      test_require = ['colorednoise']
)
