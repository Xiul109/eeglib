# eeglib

The module eeglib is a library for python 3.4+ that aims to be a set of tools to analyse EEG signals. It defines some data structures aswell as some processings in order to make easier to work with EEG data.

This library was initialy a Final Degree Project and you can find the documentation of the development in the next link:

[Documentation (Spanish)](https://ruidera.uclm.es/xmlui/handle/10578/15441)

## Main features
* Different types of processings
    * FFT
    * Averaged amplitude for each band
    * Synchronization Likelihood
    * Petrosian and Higuchi Fractal Dimensions
    * Hjorth Parameters
* Load data from CSV files
* Works with windows
* Flexible to use

## Installation

In order to install this package you just have to open a terminal in this directory and write:

`$ pip install .`

## Dependencies

* numpy
* numba

# Getting started

## Basic example

The next example shows a basic usage of the library. In it is shown how to load a file and apply a processing (Petrosian Fractal Dimension) to the data in windows of all the data.

```python
from eeglib.helpers import CSVHelper

helper= CSVHelper("fake_EEG_signal.csv")

for eeg in helper:
    print(eeg.PFD())
```

This will show this:

```python
[ 1.03089233  1.03229887  1.03181488  1.03123267  1.03069761]
```
This returns an array of the same size of the channels of the data (5) and each position of the array correspond with each channel.

## Using windows

The previous example applies the PFD to all the data in the file, but you may want to segment the data in different windows and that can be done in the next way:

```python
from eeglib.helpers import CSVHelper

helper= CSVHelper("fake_EEG_signal.csv",windowSize=256)

for eeg in helper:
    print(eeg.PFD())
```

This will show this:

```python
[ 1.03922468  1.03897773  1.03971798  1.03674636  1.03873059]
[ 1.03848326  1.04168343  1.04094783  1.04168343  1.03699509]
[ 1.03996434  1.04045647  1.03996434  1.03774006  1.03947143]
[ 1.03749194  1.04045647  1.03897773  1.0402105   1.03873059]
```

Now the function has been called 4 times, this is because of the data has a lenght of 1024 samples and the window selected has a size of 256, so the windows contained in the data are 1024/256=4.

## Using iterations

Now you may want to move the windows in another ways, like the ones that are shown in the next image:
![windows](/Examples/slidingWindow.png)

So, if i want to make the windows overlap between them you can do it this way:

```python
from eeglib.helpers import CSVHelper

helper= CSVHelper("fake_EEG_signal.csv",windowSize=256)

for eeg in helper[::128]:
    print(eeg.PFD())
```
