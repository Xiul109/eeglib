# eeglib

The module *eeglib* is a library for Python that provides tools to analyse electroencephalography (EEG) signals. This library is mainly a feature extraction tool that includes lots of frequently used algorithms in EEG processing with using a sliding window approach. *eeglib* provides a friendly interface that allows data scientists who work with EEG signals to extract lots of features with just a few lines.

## Main features
* Different types of processings
    * FFT
    * Band Power
    * Synchronization Likelihood
    * Petrosian and Higuchi Fractal Dimensions
    * Hjorth Parameters
    * Detrended Fluctuation Analysis
    * Sample Entropy
    * Lempel-Ziv Complexity
    * Cross Correlation Coeficient
* Load data from
    * CSV files
    * EDF files
    * numpy arrays
* Feature extraction oriented
* Sliding window oriented
* Flexible and easy

## Installation

Installation using pip:

`$ pip install eeglib`

## Dependencies

* numpy
* numba
* scipy
* sklearn
* pandas
* pyedflib
* fastdtw

# Getting started

Bellow there is a Quickstart Guide to eeglib. If you are interested in the API, you can find it [here](https://eeglib.readthedocs.io/en/latest/index.html).

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

## Using iterators

Now you may want to move the windows in another ways, like the ones that are shown in the next image:
![windows](/Examples/slidingWindow.png)

So, if you want to make the windows overlap between them you can do it this way:

```python
helper= CSVHelper("fake_EEG_signal.csv",windowSize=256)

for eeg in helper[::128]:
    print(eeg.PFD())
```

## Preprocessing

Maybe you want to preprocess the signals stored in the window before extracting features from them. Currently this library allows the next Preprocessings:
* Bandpass filtering
* Z-Scores normalization
* Independent Component Analysis

These preprocessings can be applied at the load of the data by the Helpers:
```python
helper = CSVHelper("fake_EEG_signal.csv",
        lowpass=30, highpass=1, normalize=True, ICA=True)
```

## Using wrappers

A Wrapper is an object that envelops a helper and simplifies the proccess of computing features that can be later be used, for example, in machine learning algorithms. The next example shows an example of how wrappers can be used:

```python
from eeglib import wrapper, helpers

helper = helpers.CSVHelper("fake_EEG_signal.csv", windowSize=128)

wrap = wrapper.Wrapper(helper)

wrap.addFeature.HFD()
wrap.addFeature.getFourierTransform()
wrap.addFeature.synchronizationLikelihood()

features=wrap.getAllFeatures()
```
So, the scheme to follow with wrappers is the next:
1. Create the Helper object.
2. Create the wrapper object.
3. Select the desired features to compute. They can be parameterized by adding the parameters just behind the name.
4. Call the method "getAllFeatures()" in order to compute every feature from every window at once or iterate over the Wrapper object for obtaining the features of each window. They are returned as a pandas.DataFrame or a pandas.Series.

# Citing
If eeglib has been useful in your research, please, consider citing the next article.

[eeglib: computational analysis of cognitive performance during the use of video games](https://link.springer.com/article/10.1007%2Fs12652-019-01592-9)


# Documents related
This library was initialy a Final Degree Project and you can find the documentation of the development in the next link:

[Final Degree Project Documentation (Spanish)](https://ruidera.uclm.es/xmlui/handle/10578/15441)

Later it was extented as part of a Master's thesis that can be found in the next link:

[Master's thesis (Spanish)](https://ruidera.uclm.es/xmlui/handle/10578/19062)


## Scientific papers

There are also some papers related to this library that can be seen bellow:

### Open Access

* [Computational EEG Analysis Techniques When Playing Video Games: A Systematic Review](https://www.mdpi.com/2504-3900/2/19/483)
* [Analysis of Cognitive Load Using EEG when Interacting with Mobile Devices](https://www.mdpi.com/2504-3900/31/1/70)

### Not open access

* [Characterisation of mobile-device tasks by their associated cognitive load through EEG data processing](https://www.sciencedirect.com/science/article/abs/pii/S0167739X20305112)
* [eeglib: computational analysis of cognitive performance during the use of video games](https://link.springer.com/article/10.1007%2Fs12652-019-01592-9)
