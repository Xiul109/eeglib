#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"This module define the basic data structure that are used in this library"

from itertools import permutations, combinations

import numpy as np
import scipy as sp

import fastdtw

from eeglib.features import (bandPower, hjorthActivity, hjorthMobility,
                             hjorthComplexity, sampEn, LZC, DFA, HFD, PFD,
                             synchronizationLikelihood)
from eeglib.preprocessing import bandPassFilter
from eeglib.auxFunctions import listType

# Default bands ranges
defaultBands = {"delta": (1, 4), "theta": (4, 7),
                "alpha": (8, 12), "beta": (12, 30)}


class SampleWindow:
    """
    This class is a data structure that stores the signal data and works like a
    sliding window.

    Attributes
    ----------
    windowSize: int
        The maximun samples the window will store.
    channelNumber: int
        The number of channels of samples the window will handle.
    window: list of lists
        It stores the data.
    """

    def __init__(self, windowSize, channelNumber,names=None):
        """
        Creates a SampleWindow wich stores the value of windowSize as the
        number of samples and the value of channelNumber as the number of
        channels.

        Parameters
        ----------
        windowSize: int
            The size of the sliding window.

        channelNumber: int
            The number of channels recording simultaneously.
        names: list of strings, optional
            The optional names that can be used to refer to each channel.
        """
        self.windowSize = windowSize
        self.channelNumber = channelNumber

        self.window = np.zeros((channelNumber,windowSize))
        if names:
            self._names=names
            self._windowDict={n:l for (n,l) in zip(names,self.window)}
        else:
            self._names=None

    def set(self, samples, columnMode=False):
        """
        Sets multiple samples at a time. The sample number must be the same as
        the window size.

        Parameters
        ----------
        samples: array_like of 2 dimensions
            Samples of data with a size equals to window.

        columnMode: boolean, optional
            By default it is assummed that the shape of the data given is
            nSamples X nchannels. If the given data is the inverse,
            columnMode should be True.
        """
        if hasattr(samples, "__getitem__") \
            and hasattr(samples[0], "__getitem__"):
            if((len(samples) == self.windowSize and
                len(samples[0]) == self.channelNumber and not columnMode)
                or (len(samples) == self.channelNumber and
                len(samples[0]) == self.windowSize and columnMode)):
                if not columnMode:
                    self.window = np.transpose(np.array(samples))
                else:
                    self.window = np.array(samples)
                if self._names:
                    for i in range(self.channelNumber):
                        self._windowDict[self._names[i]]=self.window[i]
            else:
                raise ValueError(("The number of samples must be equal to the "
                                 + "window size and each sample length must be"
                                 +" a equals to channelNumber ( {0} ) if not "
                                 +"in columnMode and viceversa if in "+
                                 "columnMode.").format(self.channelNumber))
        else:
            raise ValueError("Samples must be a subscriptable object wich "+
                             "contains subcriptable objects.")

    def getIndicesList(self, i):
        """
        Returns a list of numeric indices from a combined type of indices.
        """
        if i is None:
            return [*range(0, self.channelNumber)]
        if isinstance(i, slice):
            return self.getIndicesList(None)[i]
        if np.issubdtype(type(i), np.integer):
            return [i]
        if isinstance(i, str):
            return [self._names.index(i)]
        if isinstance(i, list):
            return sum((self.getIndicesList(j) for j in i), start=[])
        raise ValueError("This method only accepts int, str, list or" +
                             "slice types and not %s"%type(i))

    def getChannel(self, i=None):
        """
        Returns an array containing the data of the the selected channel/s.

        Parameters
        ----------
        i: Variable type, optional
            * int:  the index of the channel.
            * str:  the name of the channel.
            * list of strings and integers:  a list of channels that will be
              returned as a 2D ndarray.
            * slice:  a slice selecting the range of channels that wll be
              returned as a 2D ndarray.
            * None: all the data returned as a 2D ndarray

        Returns
        -------
        numpy.ndarray
            Can be a one or a two dimension matrix, depending of the
            parameters.
        """
        if i is None:
            return self.window

        if isinstance(i, str):
            if self._names:
                return self._windowDict[i]
            raise ValueError("There aren't names asociated to the channels.")

        if isinstance(i, list):
            selectedChannels=np.zeros((len(i),self.windowSize),dtype=np.float)
            for c, e in enumerate(i):
                if type(e) in [int, str]:
                    selectedChannels[c]=self.getChannel(e)
                else:
                    raise ValueError("The list can only contain int and/or " +
                                     "strings.")
            return selectedChannels

        if type(i) in [int,slice]:
            return self.window[i]

        raise ValueError("This method only accepts int, str, list or " +
                             "slice types and not %s"%type(i))

    def getPairsIndicesList(self, i, allPermutations=False):
        """
        Returns a list of tuples of numeric indices from a combined type of
        indices.
        """
        permutate = permutations if allPermutations else combinations
        if i is None:
            return permutate([*range(0, self.channelNumber)], 2)

        if isinstance(i, slice):
            return self.getPairsIndicesList(self.getIndicesList(None)[i])

        if isinstance(i, tuple) and len(i) == 2:
            return [tuple(self.getIndicesList(list(i)))]

        if isinstance(i, list):
            if listType(i) is tuple:
                return [self.getPairsIndicesList(c)[0] for c in i]
            return permutate(self.getIndicesList(i), 2)

        raise ValueError("This method only accepts int, str, list or " +
                             "slice types and not %s"%type(i))

    def getPairOfChannels(self, channels = None, allPermutations=False):
        """
        Applies a function that uses two signals by selecting the channels to
        use. It will apply the function to different channels depending on the
        parameters. Note: a single channel can be selected by using an int or
        a string if a name for the channel was specified.

        Parameters
        ----------
        channels: Variable type, optional
            * tuple of lenght 2 containing channel indexes: applies the
              function to the two channels specified by the tuple.
            * list of tuples(same restrictions than the above): applies the
              function to every tuple in the list.
            * list of index: creates combinations of channels specifies in the
              list depending of the allPermutations parameter.
            * None: Are channels are used in the same way than in the list
              above. This is the default value.

        allPermutations: bool, optional
            Only used when channels is a list of index or None. If True all
            permutations of the channels in every order are used; if False only
            combinations of different channels are used. Example: with the list
            [0, 2, 4] and allPermutations = True, the channels used will be
            (0,2), (0,4), (2,0), (2,4), (4,0), (4,2); meanwhile, with
            allPermutations = False, the channels used will be: (0,2), (0,4),
            (2,4). Default: False.

        Returns
        -------
        If a tuple is passed the it returns the result of applying the function
        to the channels specified in the tuple. If another valid value is
        passed, the method returns a dictionary, being the key the two channels
        used and the value the result of applying the function to those
        channels.
        """
        permutate = permutations if allPermutations else combinations

        if channels is None:
            channels = [*range(self.channelNumber)]
        elif isinstance(channels, slice):
            channels = [*range(self.channelNumber)[channels]]

        if isinstance(channels, tuple):
            if len(channels) != 2:
                raise ValueError("If you specify a tuple, it must have " +
                                 "exactly two(2) elements")
            return (self.getChannel(channels[0]), self.getChannel(channels[1]))

        if isinstance(channels, list):
            t = listType(channels)
            if t is tuple:
                acs = np.array(channels)
                if len(acs.shape) == 2 and acs.shape[1] == 2:
                    return [(self.getChannel(c[0]), self.getChannel(c[1]))
                            for c in channels]
                raise ValueError("The tuples of the list must have " +
                                 "exactly two(2) elements")
            try:
                return permutate(self.getChannel(channels), 2)
            except ValueError as err:
                raise ValueError("The list must contain either tuples " +
                                 "or integers and strings, but no "+
                                 "combinations of them.") from err

        raise ValueError("This method only accepts list (either of int "+
                         "and str or tuples) or tuple types and not %s" %
                         type(channels))

    def __getitem__(self, n):
        return self.window[:,n]

    # Returns a string representation of the inner list
    def __str__(self):
        return str(self.window)


class EEG:
    """
    This class apply sginal analysis functions over the data stored in its
    attribute window.

    Attributes
    ----------
    windowSize: int
        The maximun samples the window will store.
    sampleRate: int
        The number of samples per second
    channelNumber: int
        The number of channels of samples the window will handle.
    window: SampleWindow
        It stores the data.
    """

    def __init__(self, windowSize, sampleRate, channelNumber, names=None):
        """
        Parameters
        ----------
        windowSize: int
            The maximun samples the window will store.
        sampleRate: int
            The number of samples per second
        channelNumber: int
            The number of channels of samples the window will handle.
        names: list of strings, optional
            The optional names that can be used to refer to each channel.
        """
        self.windowSize = windowSize
        self.sampleRate = sampleRate
        self.channelNumber = channelNumber
        self.window = SampleWindow(windowSize, channelNumber,names=names)
        self.outputMode = "array"

    # Function to handle the windowFunction parameter
    def _handleWindowFunction(self, windowFunction):
        if windowFunction is None:
            return 1

        if type(windowFunction) in (str, tuple):
            return sp.signal.get_window(windowFunction, self.windowSize)

        if isinstance(windowFunction, np.ndarray):
            if len(windowFunction) == self.windowSize:
                return windowFunction
            raise ValueError("The size of windowFunction is not the same " +
                             "as windowSize.")

        raise ValueError("Not a valid type for windowFunction.")


    def set(self, samples, columnMode=False):
        """
        Sets multiple samples at a time into the window. The sample number must
        be the same as the window size.

        Parameters
        ----------
        samples: array_like of 2 dimensions
            Samples of data with a size equals to window.

        columnMode: boolean, optional
            By default it is assummed that the shape of the data given is
            nSamples X nchannels. If the given data is the inverse,
            columnMode should be True.
        """
        self.window.set(samples, columnMode)

    def getChannel(self, i=None):
        """
        Returns the raw data stored at the given index of the windows.

        Parameters
        ----------
        i: Variable type, optional
            * int:  the index of the channel.
            * str:  the name of the channel.
            * list of strings and integers:  a list of channels that will be
              returned as a 2D ndarray.
            * slice:  a slice selecting the range of channels that wll be
              returned as a 2D ndarray.
            * None: all the data returned as a 2D ndarray

        Returns
        -------
        list
            The list of values of a specific channel.
        """
        return self.window.getChannel(i)

    def _applyFunctionTo(self, function, i=None):
        """
        Returns the raw data stored at the given index of the windows.

        Parameters
        ----------
        i: Variable type, optional
            * int:  the index of the channel.
            * str:  the name of the channel.
            * list of strings and integers:  a list of channels that will be
              returned as a 2D ndarray.
            * slice:  a slice selecting the range of channels that wll be
              returned as a 2D ndarray.
            * None: all the data returned as a 2D ndarray

        Returns
        -------
        list
            The list of values of a specific channel.
        """
        data=self.window.getChannel(i)
        if len(np.shape(data))==1:
            return function(data)

        if self.outputMode=="dict":
            indices = self.window.getIndicesList(i)
            if self.window._names:
                keys = np.array(self.window._names)[indices]
            else:
                keys=indices
            return {key:function(d) for key, d in zip(keys, data)}

        return np.array([function(d) for d in data])

    def _applyFunctionTo2C(self, function, channels = None,
                            allPermutations=False):
        """
        Applies a function that uses two signals by selecting the channels to
        use. It will apply the function to different channels depending on the
        parameters. Note: a single channel can be selected by using an int or
        a string if a name for the channel was specified.

        Parameters
        ----------
        function: function
            A function that receives two streams of data and will be applied to
            the selected channels.

        channels: Variable type, optional
            * tuple of lenght 2 containing channel indexes: applies the
              function to the two channels specified by the tuple.
            * list of tuples(same restrictions than the above): applies the
              function to every tuple in the list.
            * list of index: creates combinations of channels specifies in the
              list depending of the allPermutations parameter.
            * None: Are channels are used in the same way than in the list
              above. This is the default value.

        allPermutations: bool, optional
            Only used when channels is a list of index or None. If True all
            permutations of the channels in every order are used; if False only
            combinations of different channels are used. Example: with the list
            [0, 2, 4] and allPermutations = True, the channels used will be
            (0,2), (0,4), (2,0), (2,4), (4,0), (4,2); meanwhile, with
            allPermutations = False, the channels used will be: (0,2), (0,4),
            (2,4). Default: False.

        Returns
        -------
        If a tuple is passed the it returns the result of applying the function
        to the channels specified in the tuple. If another valid value is
        passed, the method returns a dictionary, being the key the two channels
        used and the value the result of applying the function to those
        channels.
        """
        data = self.window.getPairOfChannels(channels, allPermutations)

        if isinstance(data, tuple):
            return function(*data)

        if self.outputMode == "dict":
            indices = self.window.getPairsIndicesList(channels,allPermutations)
            if self.window._names:
                keys = [(self.window._names[i[0]], self.window._names[i[1]])
                        for i in indices]
            else:
                keys=indices
            return {key:function(*value) for key, value in zip(keys, data)}

        return [function(*value) for value in data]

    def DFT(self, i=None, windowFunction=None, output="complex",
            onlyPositiveFrequencies = False):
        """
        Returns the Discrete Fourier Transform of the data at a given index of
        the window.

        Parameters
        ----------
        i: Variable type, optional
            * int:  the index of the channel.
            * str:  the name of the channel.
            * list of strings and integers:  a list of channels that will be
              used.
            * slice: a slice selecting the range of channels that will be
              used.
            * None: all the channels will be used.

        windowFunction: str, tuple or numpy.ndarray, optional
            This can be a string with the name of the function, a tuple with a
            str with the name of the funcion in the first position and the
            parameters of the funcion in the nexts or a numpy array with a size
            equals to the window size. In the first case an array with the
            size of windowSize will be created. The created array will be
            multiplied by the data in the window before FFT is used.

        output: str, optional
            * "complex": default output of the FFT, x+yi
            * "magnitude": computes the magnitude of the FFT, sqrt(x^2+y^2)
            * "phase": computes the phase of the FFT, atan2(yi,x)

        Returns
        -------
        numpy.ndarray
            An array with the result of the Fourier Transform. If more than one
            channel was selected the array will be of 2 Dimensions.
        """
        windowFunction=self._handleWindowFunction(windowFunction)
        f1 = lambda x:np.fft.fft(x*windowFunction)
        output = output.lower()
        if output == "magnitude":
            f = lambda x: np.abs(f1(x))
        elif output == "phase":
            f = lambda x: np.angle(f1(x))
        elif output == "complex":
            f=f1
        else:
            raise ValueError('Only "complex", "magnitude" or "phase" are ' +
                             'valid values por output parameter')

        if onlyPositiveFrequencies:
            auxF = f
            f = lambda x: auxF(x)[:self.windowSize//2+1]

        return self._applyFunctionTo(f, i)


    def PSD(self, i = None, windowFunction = "hann", nperseg = None,
            retFrequencies = False):
        """
        Returns the Power Spectral Density of the data at a given index of
        the window using the Welch method.

        Parameters
        ----------
        i: Variable type, optional
            * int:  the index of the channel.
            * str:  the name of the channel.
            * list of strings and integers:  a list of channels that will be
              used.
            * slice: a slice selecting the range of channels that will be
              used.
            * None: all the channels will be used.

        windowFunction: str or tuple or array_like, optional
            Desired window to use. If window is a string or tuple, it is passed
            to get_window to generate the window values, which are DFT-even by
            default. See get_window for a list of windows and required
            parameters. If window is array_like it will be used directly as the
            window and its length must be nperseg. Defaults to a Hann window.

        nperseg: int, optional
            Length of each segment. Defaults to None, but if window is str or
            tuple, is set to 256, and if window is array_like, is set to the
            length of the window.

        retFrequencies: bool, optional
            If True two arrays will be raturned for each channel, one with the
            frequencies and another with the spectral density of those
            frequencies.

        Returns
        -------
        numpy.ndarray
            An array with the result of the Fourier Transform. If more than one
            channel was selected the array will be of 2 Dimensions.
        """
        if nperseg is None:
            nperseg = self.windowSize//2
        f1 = lambda x: sp.signal.welch(x, self.sampleRate,
                                       window = windowFunction,
                                       nperseg = nperseg)
        if not retFrequencies:
            f = lambda x: f1(x)[1]
        else:
            f = f1
        return self._applyFunctionTo(f, i)


    def bandPower(self, i=None, bands=defaultBands, spectrumFrom = "DFT",
                  windowFunction="hann", nperseg = None, normalize=False):
        """
        Returns the power of each band at the given index.

        Parameters
        ----------
        i: Variable type, optional
            * int:  the index of the channel.
            * str:  the name of the channel.
            * list of strings and integers:  a list of channels that will be
              used.
            * slice: a slice selecting the range of channels that will be
              used.
            * None: all the channels will be used.

        bands: dict, optional
            This parameter is used to indicate the bands that are going to be
            used. It is a dict with the name of each band as key and a tuple
            with the lower and upper bounds as value.

        spectrumFrom: str, optional
            * "DFT": uses the spectrum from the DFT of the signal.
            * "PSD": uses the spectrum from the PSD of the signal.

        windowFunction: str or tuple or array_like, optional
            Desired window to use. If window is a string or tuple, it is passed
            to get_window to generate the window values, which are DFT-even by
            default. See get_window for a list of windows and required
            parameters. If window is array_like it will be used directly as the
            window and its length must be nperseg. Defaults to a Hann window.

        nperseg: int, optional
            This parameter is only relevant when powerFrom is "PSD", else it is
            ignored. Length of each segment. Defaults to None, but if window is
            str or tuple, is set to 256, and if window is array_like, is set to
            the length of the window.

        normalize: bool, optional
            If True the each band power is divided by the total power of the
            spectrum. Default False.

        Returns
        -------
        dict or list of dict
            The keys are the name of each band and the values are the mean of
            the magnitudes. If more than one channel was selected the return
            object will be a list containing the dict for each channel selected
        """
        bands={key:self.getBoundsForBand(b) for key,b in bands.items()}
        freqRes = self.sampleRate/self.windowSize
        if spectrumFrom.upper() == "DFT":
            spectrum = self.DFT(i, windowFunction, output="magnitude",
                                onlyPositiveFrequencies=True)
        elif spectrumFrom.upper() == "PSD":
            if nperseg is None:
                nperseg = self.windowSize//2
            spectrum = self.PSD(i, windowFunction, nperseg)
            factor = self.windowSize//nperseg
            bands = {k:(b[0]//factor, b[1]//factor) for k, b in bands.items()}
            freqRes*=factor
        else:
            raise ValueError('%s is not a valid value for '%spectrumFrom +
                             'powerFrom parameter. Only "DFT" or "PSD" are ' +
                             'valid values.')

        if isinstance(spectrum, dict):
            return {key:bandPower(s, bands, freqRes, normalize) for key, s
                    in spectrum.items()}

        if spectrum.ndim==1:
            return bandPower(spectrum, bands, freqRes, normalize)

        return [bandPower(s, bands, freqRes, normalize) for s in spectrum]

    def getBoundsForBand(self, bandBounds):
        """
        Returns the bounds of each band depending of the sample rate and the
        window size.

        Parameters
        ----------
        bandbounds: tuple
            A tuple containig the lower and upper bounds of a frequency band.

        Returns
        -------
        tuple
            A tuple containig the the new bounds of the given band.
        """
        return tuple(map(lambda val:
            int(val * self.windowSize / self.sampleRate), bandBounds))

    def getSignalAtBands(self, i=None, bands=defaultBands):
        """
        Rebuilds the signal from a component i but only in the specified
        frequency bands.

        Parameters
        ----------
        i: Variable type, optional
            * int:  the index of the channel.
            * str:  the name of the channel.
            * list of strings and integers:  a list of channels that will be
              used.
            * slice: a slice selecting the range of channels that will be
              used.
            * None: all the channels will be used.

        bands: dict, optional
            This parameter is used to indicate the bands that are going to be
            used. It is a dict with the name of each band as key and a tuple
            with the lower and upper bounds as value.

        Returns
        -------
        dict of numpy.ndarray (1D or 2D)
            The keys are the same keys the bands dictionary is using. The
            values are the signal filtered in every band at the given index of
            the window. If more than one channel is selected the return object
            will be a dict containing 2D arrays in which each row is a signal
            filtered at the corresponding channel.
        """

        bandsSignals = {}
        for key, band in bands.items():
            bounds = self.getBoundsForBand(band)
            bandsSignals[key]=self._applyFunctionTo(lambda x:
                bandPassFilter(x, self.sampleRate, bounds[0], bounds[1]), i)

        return bandsSignals

    def PFD(self, i=None):
        """
        Returns the Petrosian Fractal Dimension at the given index of the
        window.

        Parameters
        ----------
        i: Variable type, optional
            * int:  the index of the channel.
            * str:  the name of the channel.
            * list of strings and integers:  a list of channels that will be
              used.
            * slice: a slice selecting the range of channels that will be
              used.
            * None: all the channels will be used.

        Returns
        -------
        float or array
            The resulting value. If more than one channe was selected the
            return object will be a 1D array containing the result of the
            procesing.
        """
        return self._applyFunctionTo(PFD,i)

    def HFD(self, i=None, kMax=None):
        """
        Returns the Higuchi Fractal Dimension at the given index of the window.

        Parameters
        ----------
        i: Variable type, optional
            * int:  the index of the channel.
            * str:  the name of the channel.
            * list of strings and integers:  a list of channels that will be
              used.
            * slice: a slice selecting the range of channels that will be
              used.
            * None: all the channels will be used.
        kmax: int, optional
            By default it will be windowSize//2.

        Returns
        -------
        float or array
            The resulting value. If more than one channe was selected the
            return object will be a 1D array containing the result of the
            procesing.
        """
        return self._applyFunctionTo(lambda v:HFD(v,kMax),i)

    # Hjorth Parameters:
    def hjorthActivity(self, i=None):
        """
        Returns the Hjorth Activity at the given channel

        Parameters
        ----------
        i: int or string, optional
            Index or name of the channel.

        Returns
        -------
        float or array
            The resulting value. If more than one channe was selected the
            return object will be a 1D array containing the result of the
            procesing.
        """
        return self._applyFunctionTo(hjorthActivity,i)

    def hjorthMobility(self, i=None):
        """
        Returns the Hjorth Mobility at the given channel

        Parameters
        ----------
        i: Variable type, optional
            * int:  the index of the channel.
            * str:  the name of the channel.
            * list of strings and integers:  a list of channels that will be
              used.
            * slice: a slice selecting the range of channels that will be
              used.
            * None: all the channels will be used.

        Returns
        -------
        float or array
            The resulting value. If more than one channe was selected the
            return object will be a 1D array containing the result of the
            procesing.
        """
        return self._applyFunctionTo(hjorthMobility,i)

    def hjorthComplexity(self, i=None):
        """
        Returns the Hjorth Complexity at the given channel

        Parameters
        ----------
        i: Variable type, optional
            * int:  the index of the channel.
            * str:  the name of the channel.
            * list of strings and integers:  a list of channels that will be
              used.
            * slice: a slice selecting the range of channels that will be
              used.
            * None: all the channels will be used.

        Returns
        -------
        float or array
            The resulting value. If more than one channe was selected the
            return object will be a 1D array containing the result of the
            procesing.
        """
        return self._applyFunctionTo(hjorthComplexity,i)

    def synchronizationLikelihood(self, channels = None, allPermutations=False,
                                  m=None, l=None, w1=None, w2=None, pRef=0.05,
                                  **kargs):
        """
        Returns the Synchronization Likelihood value applied over the i1 and i2
        channels by calling :func:`~eeglib.eeg.synchronizationLikelihood`.

        Parameters
        ----------
        channels: Variable type, optional
           In order to understand how this parameter works go to the doc of
           :py:meth:`eeglib.eeg.EEG._applyFunctionTo2C`
        allPermutations: bool, optional
            In order to understand how this parameter works go to the doc of
           :py:meth:`eeglib.eeg.EEG._applyFunctionTo2C`
        m: int, optional
            Numbers of elements of the embedded vectors.
        l: int, optional
            Separation between elements of the embedded vectors.
        w1: int, optional
            Theiler correction for autocorrelation effects
        w2: int, optional
            A window that sharpens the time resolution of the Synchronization
            measure
        pRef: float, optional
            The p Ref param of the synchronizationLikelihood. Default 0.05
        epsilonIterations: int,optional
            Number of iterations used to determine the value of epsilon

        Returns
        -------
        float or dict
            If a tuple is passed the it returns the result of applying the
            function to the channels specified in the tuple. If another valid
            value is passed, the method returns a dictionary, being the key the
            two channels used and the value the result of applying the function
            to those channels.
        """
        if l is None:
            l=1
        if m is None:
            m=int(np.sqrt(self.windowSize))
        l = 1 if l == 0 else l
        if w1 is None:
            w1 = int(2 * l * (m - 1))
        if w2 is None:
            w2 = int(10 // pRef + w1)
        return self._applyFunctionTo2C(lambda c1,c2:synchronizationLikelihood(
                                             c1,c2, m, l, w1, w2, pRef,**kargs)
                                        ,channels, allPermutations)

    def engagementLevel(self):
        """
        Returns the engagament level, which is calculated with this formula:
        beta/(alpha+theta), where alpha, beta and theta are the average of the
        average band values between al the channels.

        Returns
        -------
        float
            The engagement level.
        """
        bandValues = self.bandPower()
        alphas, betas, thetas = [], [], []
        for d in bandValues:
            alphas.append(d["alpha"])
            betas.append(d["beta"])
            thetas.append(d["theta"])

        alpha, beta, theta = np.mean(alphas), np.mean(betas), np.mean(thetas)

        return beta / (alpha + theta)

    def sampEn(self, i=None, *args, **kargs):
        """
        Returns Multiscale Sample Entropy at the given channel/s.

        Parameters
        ----------
        i: Variable type, optional
            * int:  the index of the channel.
            * str:  the name of the channel.
            * list of strings and integers:  a list of channels that will be
              used.
            * slice: a slice selecting the range of channels that will be
              used.
            * None: all the channels will be used
        m: int, optional
            Size of the embedded vectors. By default 2.
        l: int, optional
            Lag beetwen elements of embedded vectors. By default 1.
        r: float, optional
            Tolerance. By default fr*std(data)
        fr: float, optional
            Fraction of std(data) used as tolerance. If r is passed, this
            parameter is ignored. By default, 0.2.

        Returns
        -------
        float or array
            The resulting value. If more than one channe was selected the
            return object will be a 1D array containing the result of the
            procesing.
        """
        return self._applyFunctionTo(lambda x: sampEn(x,*args, **kargs), i)

    def LZC(self, i=None, *args, **kargs):
        """
        Returns the Lempel-Ziv Complexity at the given channel/s.

        Parameters
        ----------
        i: Variable type, optional
            * int:  the index of the channel.
            * str:  the name of the channel.
            * list of strings and integers:  a list of channels that will be
              used.
            * slice: a slice selecting the range of channels that will be
              used.
            * None: all the channels will be used
        theshold: numeric, optional
            A number use to binarize the signal. The values of the signal above
            threshold will be converted to 1 and the rest to 0. By default, the
            median of the data.
        normalize: bool
            If True the resulting value will be between 0 and 1, being 0 the
            minimal posible complexity of a sequence that has the same lenght
            of data and 1 the maximal posible complexity. By default, False.

        Returns
        -------
        float or array
            The resulting value. If more than one channe was selected the
            return object will be a 1D array containing the result of the
            procesing.
        """
        return self._applyFunctionTo(lambda x: LZC(x, *args, **kargs), i)

    def DFA(self, i=None, *args, **kargs):
        """
        Applies Detrended Fluctuation Analysis algorithm to the given data.

        Parameters
        ----------
         i: Variable type, optional
            * int:  the index of the channel.
            * str:  the name of the channel.
            * list of strings and integers:  a list of channels that will be
              used.
            * slice: a slice selecting the range of channels that will be
              used.
            * None: all the channels will be used
        fit_degree: int, optional
            Degree of the polynomial used to model de local trends. Default: 1.
        min_window_size: int, optional
            Size of the smallest window that will be used. Default:
            signalSize//2.
        fskip: float, optional
            Fraction of the window that will be skiped in each iteration for
            each window size. Default: 1.

        Returns
        -------
        float or array
            The resulting value. If more than one channe was selected the
            return object will be a 1D array containing the result of the
            procesing.
        """
        return self._applyFunctionTo(lambda x: DFA(x, *args, **kargs), i)

    def CCC(self, channels = None, allPermutations=False):
        """
        Computes the Cross Correlation Coeficient between the data in c1 and
        the data in c2.

        Parameters
        ----------
        channels: Variable type, optional
           In order to understand how this parameter works go to the doc of
           :py:meth:`eeglib.eeg.EEG._applyFunctionTo2C`
        allPermutations: bool, optional
            In order to understand how this parameter works go to the doc of
           :py:meth:`eeglib.eeg.EEG._applyFunctionTo2C`

        Returns
        -------
        float or dict
            If a tuple is passed the it returns the result of applying the
            function to the channels specified in the tuple. If another valid
            value is passed, the method returns a dictionary, being the key the
            two channels used and the value the result of applying the function
            to those channels.
        """
        f = lambda x,y: np.corrcoef(x,y)[0,1]
        return self._applyFunctionTo2C(f, channels,
                                        allPermutations=allPermutations)

    def DTW(self, channels=None, allPermutations = False, normalize = False,
            returnOnlyDistances = True ,*args, **kargs):
        """
        Computes the Dynamic Time Warping algortihm between the data of the
        given channels. It uses the FastDTW implementation given by the library
        fastdtw.

        Parameters
        ----------
        channels: Variable type, optional
           In order to understand how this parameter works go to the doc of
           :py:meth:`eeglib.eeg.EEG._applyFunctionTo2C`
        allPermutations: bool, optional
            In order to understand how this parameter works go to the doc of
           :py:meth:`eeglib.eeg.EEG._applyFunctionTo2C`
        normalize: bool optional
             If True the result of the algorithm is divided by the window size.
             Default: True.
        returnOnlyDistances: bool, optional
            If True, the result of the function will include only the distances
            after applying the DTW algorithm. If False it will return also the
            path. Default: True.
        radius : int, optional
            size of neighborhood when expanding the path. A higher value will
            increase the accuracy of the calculation but also increase time
            and memory consumption. A radius equal to the size of x and y will
            yield an exact dynamic time warping calculation.
        dist : function or int, optional
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.

        Returns
        -------
        tuple, float, dict of floats or dict of tuples
            If a tuple is passed the it returns the result of applying the
            function to the channels specified in the tuple. If another valid
            value is passed, the method returns a dictionary, being the key the
            two channels used and the value the result of applying the function
            to those channels.
        """

        fAux=lambda c1, c2: fastdtw.fastdtw(c1, c2, *args, **kargs)

        if normalize and returnOnlyDistances:
            f= lambda c1, c2: fAux(c1,c2)[0]/self.windowSize
        elif returnOnlyDistances:
            f= lambda c1, c2: fAux(c1,c2)[0]
        elif normalize:
            def f(c1,c2):
                ret = fAux(c1,c2)
                return ret[0]/self.windowSize,ret[1]
        else:
            f=fAux

        ret = self._applyFunctionTo2C(f, channels,
                                       allPermutations=allPermutations)

        return ret
