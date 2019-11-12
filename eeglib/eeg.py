#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"This module define the data structures that are used in this library"

import numpy as np

from itertools import permutations, combinations

from eeglib.features import (averageBandValues, hjorthActivity, hjorthMobility,
                             hjorthComplexity, MSE, LZC, DFA, HFD, PFD, CCC,
                             synchronizationLikelihood)
from eeglib.preprocessing import bandPassFilter
from eeglib.auxFunctions import listType

import fastdtw

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
            Can be a one or a two dimension matrix, depending of the
            parameters.
        """
        if i is None:
            return self.window
        elif type(i) is str:
            if self._names:
                return self._windowDict[i]
            else:
                raise ValueError("There aren't names asociated to the \
                                 channels.")
        elif type(i) is list:
            selectedChannels=np.zeros((len(i),self.windowSize),dtype=np.float)
            for c, e in enumerate(i):
                if type(e) in [int, str]:
                    selectedChannels[c]=self.getChannel(e)
                else:
                    raise ValueError("The list can only contain int and/or \
                                     strings.")
            return selectedChannels
        elif type(i) in [int,slice]:
            return self.window[i]
        else:
            raise ValueError("This method only accepts int, str, list or" +
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
    def _handleWindowFunction(self, windowFunction):
        if windowFunction is None:
            return 1
        elif type(windowFunction) == str:
            if windowFunction == "hamming":
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
    
    def _applyFunctionTo(self, function, i=None):
        data=self.window.getChannel(i)
        if len(np.shape(data))==1:
            return function(data)
        else:
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
        permutate = permutations if allPermutations else combinations
        
        if channels == None:
            channels = list(range(self.channelNumber))
        
        if type(channels) is tuple:
            if len(channels) != 2:
                raise ValueError("If you specify a tuple, it must have \
                                 exactly two(2) elements")
            else:
                return function(*self.getChannel(list(channels)))
        elif type(channels) is list:
            t = listType(channels)
            if t is tuple:
                acs = np.array(channels)
                if len(acs.shape) == 2 and acs.shape[1] == 2:
                    return {c:function(*self.getChannel(list(c))) for c in 
                            channels}
                else:
                    raise ValueError("The tuples of the list must have \
                                     exactly two(2) elements")
            else:
                return {c:function(*self.getChannel(list(c))) for c in
                        permutate(channels, 2)}
        else:
            raise ValueError("This method only accepts list (either of int\
                              and str or tuples) or tuple types and not %s" %
                             type(channels))
                    

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
        windowFunction=self._handleWindowFunction(windowFunction)
        return self._applyFunctionTo(lambda x:np.fft.fft(x*windowFunction), i)

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
        if l == None:
            l=1
        if m==None:
            m=int(np.sqrt(self.windowSize))
        l = 1 if l == 0 else l
        if w1==None:
            w1 = int(2 * l * (m - 1))
        if w2 == None:
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
        bandValues = self.getAverageBandValues()
        alphas, betas, thetas = [], [], []
        for d in bandValues:
            alphas.append(d["alpha"])
            betas.append(d["beta"])
            thetas.append(d["theta"])

        alpha, beta, theta = np.mean(alphas), np.mean(betas), np.mean(thetas)

        return beta / (alpha + theta)
    
    def MSE(self, i=None, *args, **kargs):
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
        return self._applyFunctionTo(lambda x: MSE(x,*args, **kargs), i)
    
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
#        c1, c2 = self.getChannel(i1), self.getChannel(i2)
        return self._applyFunctionTo2C(CCC, channels, 
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