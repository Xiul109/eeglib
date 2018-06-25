#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"This module define the data structures that are used in this library"

import numpy as np

from eeglib.features import (HFD, PFD, averageBandValues, hjorthActivity,
                             hjorthMobility, hjorthComplexity, MSE, LZC,
                             synchronizationLikelihood)
from eeglib.preprocessing import bandPassFilter

# Default bands ranges
defaultBands = {"delta": (1, 4), "theta": (4, 7),
                "alpha": (8, 12), "beta": (12, 30)}
"This variable are the default frequency bands used in this library"


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
            self.__names=names
            self.__windowDict={n:l for (n,l) in zip(names,self.window)}
        else:
            self.__names=None

    
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
                if self.__names:
                    for i in range(self.channelNumber):
                        self.__windowDict[self.__names[i]]=self.window[i]
            else:
                raise ValueError(("the number of samples must be equal to the "
                                 + "window size and each sample length must be"
                                 +" a equals to channelNumber ( {0} ) if not"
                                 +"in columnMode and viceversa if in"+
                                 "columnMode").format(self.channelNumber))
        else:
            raise ValueError("samples must be a subscriptable object wich"+
                             "contains subcriptable objects")

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
            Can be a one or a two dimension matrix, depending of the parameters.
        """
        if i is None:
            return self.window
        elif type(i) is str:
            if self.__names:
                return self.__windowDict[i]
            else:
                raise ValueError("There aren't names asociated to the channels.")
        elif type(i) is list:
            selectedChannels=np.zeros((len(i),256),dtype=np.float)
            for c,e in enumerate(i):
                if type(e) in [int,str]:
                    selectedChannels[c]=self.getChannel(e)
                else:
                    ValueError("The list can only contain int or strings.")
            return selectedChannels
        elif type(i) in [int,slice]:
            return self.window[i]
        else:
            raise ValueError("This methods only accepts int, str, list or" +
                             "slice types and not %s"%type(i))

    class SampleWindowIterator:
        """
        This class is used as a only for iterating over the data. It only has a
        __init__ method and __next__ method.
        """
        
        def __init__(self, iterWindow):
            """
            Parameters
            ----------
            iterWindow: SampleWindow
                The SampleWindow object to be iterated.
            """
            self.iterWindow = iterWindow
            self.i = 0

        def __next__(self):
            if self.i < len(self.iterWindow):
                self.i += 1
                return self.iterWindow[self.i - 1]
            else:
                raise StopIteration
    # Returns the iteration class

    def __iter__(self):
        return self.SampleWindowIterator(self)

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

    # Function to handle the windowFunction parameter
    def __handleWindowFunction(self, windowFunction):
        if windowFunction is None:
            return 1
        elif type(windowFunction) == str:
            if windowFunction is "hamming":
                return np.hamming(self.windowSize)
            else:
                raise ValueError("the option chosen is not valid")
        elif type(windowFunction) == np.ndarray:
            if len(windowFunction) == self.windowSize:
                return windowFunction
            else:
                raise ValueError(
                    "the size of windowFunction is not the same as windowSize")

        else:
            raise ValueError("not a valid type for windowFunction")

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
    
    def __applyFunctionTo(self,function,i=None):
        data=self.window.getChannel(i)
        if len(np.shape(data))==1:
            return function(data)
        else:
            return np.array([function(d) for d in data])

    def getFourierTransform(self, i=None,  windowFunction=None):
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
        
        windowFunction: String or numpy.ndarray, optional
            This can be a String with the name of the function (currently only
            supported **"hamming"**) or it can be a numpy array with a size
            equals to the window size. In the first case an array with the
            size of windowSize will be created. The created array will be
            multiplied by the data in the window before FFT is used.

        Returns
        -------
        numpy.ndarray
            An array with the result of the Fourier Transform. If more than one
            channel was selected the array will be of 2 Dimensions.
        """
        windowFunction=self.__handleWindowFunction(windowFunction)
        return self.__applyFunctionTo(lambda x:np.fft.fft(x*windowFunction), i)

    # Gets the magnitude of each complex value resulting from a Fourier
    # Transform from a component i
    def getMagnitudes(self, *args, **kargs):
        """
        Returns the magnitude of each complex value resulting from a Discrete
        Fourier Transform at a the given index of the window.

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
            
        windowFunction: String or numpy.ndarray, optional
            This can be a String with the name of the function (currently only
            supported **"hamming"**) or it can be a numpy array with a size
            equals to the window size. In the first case an array with the
            size of windowSize will be created. The created array will be
            multiplied by the data in the window before FFT is used.

        Returns
        -------
        numpy.ndarray
            An array with the magnitudes of the Fourier Transform. If more than
            one channel was selected the array will be of 2 Dimensions.
        """
        return abs(self.getFourierTransform(*args,**kargs))

    def getAverageBandValues(self, i=None, bands=defaultBands, *args, **kargs):
        """
        Returns the average magnitude of each band at the given index of the
        window

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
        
        windowFunction: String or numpy.ndarray, optional
            This can be a String with the name of the function (currently only
            supported **"hamming"**) or it can be a numpy array with a size
            equals to the window size. In the first case an array with the
            size of windowSize will be created. The created array will be
            multiplied by the data in the window before FFT is used.

        Returns
        -------
        dict or list of dict
            The keys are the name of each band and the values are the mean of
            the magnitudes. If more than one channel was selected the return
            object will be a list containing the dict for each channel selected
        """
        magnitudes = self.getMagnitudes(i,*args,**kargs)
        bands={key:self.getBoundsForBand(b) for key,b in bands.items()}
        if magnitudes.ndim==1:
            return averageBandValues(magnitudes,bands)
        else:
            averagedValues=[]
            for c in magnitudes:
                averagedValues.append(averageBandValues(c,bands))
            return averagedValues

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
            bandsSignals[key]=self.__applyFunctionTo(lambda x: 
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
        return self.__applyFunctionTo(PFD,i)
    
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
        return self.__applyFunctionTo(lambda v:HFD(v,kMax),i)

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
        float
            The resulting value
        """
        return self.__applyFunctionTo(hjorthActivity,i)

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
        return self.__applyFunctionTo(hjorthMobility,i)

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
        return self.__applyFunctionTo(hjorthComplexity,i)

    def synchronizationLikelihood(self, i1, i2, m=None, l=None,
                                  w1=None, w2=None, pRef=0.05,**kargs):
        """
        Returns the Synchronization Likelihood value applied over the i1 and i2
        channels by calling :func:`~eeglib.eeg.synchronizationLikelihood`.

        Parameters
        ----------
        i1: int or string
            Index or name of the first channel.
        i2: int or string
            Index or name of the second channel.
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
        float
            The resulting value
        """
        c1, c2 = self.getChannel(i1), self.getChannel(i2)
        if l == None:
            l=1
        if m==None:
            m=int(np.sqrt(self.windowSize))
        l = 1 if l == 0 else l
        if w1==None:
            w1 = int(2 * l * (m - 1))
        if w2 == None:
            w2 = int(10 // pRef + w1)
        return synchronizationLikelihood(c1, c2, m, l, w1, w2, pRef,**kargs)

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
        bandValues = self.getAverageBandValues()
        alphas, betas, thetas = [], [], []
        for d in bandValues:
            alphas.append(d["alpha"])
            betas.append(d["beta"])
            thetas.append(d["theta"])

        alpha, beta, theta = np.mean(alphas), np.mean(betas), np.mean(thetas)

        return beta / (alpha + theta)
    
    def MSE(self, i, *args, **kargs):
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
        float
            The resulting value    
        """
        return self.__applyFunctionTo(lambda x: MSE(x,*args, **kargs), i)
    
    def LZC(self, i, *args, **kargs):
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
            minimal posible complexity of a sequence that has the same lenght of 
            data and 1 the maximal posible complexity. By default, False.
        """
        return self.__applyFunctionTo(lambda x: LZC(x, *args, **kargs), i)