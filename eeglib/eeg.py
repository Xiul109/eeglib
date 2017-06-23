#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"This module define the data structures aswell the functions for EEG analysis"
import numpy as np

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
    electrodeNumber: int
        The number of channels of samples the window will handle.
    window: list of lists
        It stores the data.
    means: list
        It stores the means of each channels in order to use it later for
        normalization.
    """

    def __init__(self, windowSize, electrodeNumber):
        """
        Creates a SampleWindow wich stores the value of windowSize as the
        number of samples and the value of electrodeNumber as the number of
        channels.

        Parameters
        ----------
        windowSize: int
            The size of the sliding window.

        electrodeNumber: int
            The number of electrodes recording simultaneously.
        """
        self.windowSize = windowSize
        self.electrodeNumber = electrodeNumber

        self.window = [[0 for i in range(windowSize)]
                       for i in range(electrodeNumber)]
        self.means = [0 for j in range(electrodeNumber)]

    def add(self, sample):
        """
        Adds a sample to the begining of the window and pops the last one and
        recalculates the means.

        Parameters
        ----------
        sample: array_like
            One sample of data
        """
        if hasattr(sample, "__getitem__") and len(sample) == self.electrodeNumber:
            for i in range(self.electrodeNumber):
                poped = self.window[i].pop()
                self.window[i].insert(0, sample[i])
                self.means[i] += (sample[i] - poped) / self.windowSize
        else:
            raise ValueError(
                "sample must be a subscriptable object with a length equals to electrodeNumber (" + str(self.electrodeNumber) + ")")

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
            nSamples X nElectrodes. If the given data is the inverse,
            columnMode should be True.
        """
        if hasattr(samples, "__getitem__") and hasattr(samples[0], "__getitem__"):
            if((len(samples) == self.windowSize and
               len(samples[0]) == self.electrodeNumber and not columnMode) or
               (len(samples) == self.electrodeNumber and
               len(samples[0]) == self.windowSize and columnMode)):
                for i in range(self.electrodeNumber):
                    if not columnMode:
                        self.window[i] = getComponent(i, samples)
                    else:
                        self.window[i] = list(samples[i])
                    self.means[i] = np.mean(self.window[i])
            else:
                raise ValueError("the number of samples must be equal to the window size and each sample length must be a equals to electrodeNumber (" + str(
                    self.electrodeNumber) + ") if not in columnMode and viceversa if in columnMode")
        else:
            raise ValueError(
                "samples must be a subscriptable object wich contains subcriptable objects")

    def getComponentAt(self, i):
        """
        Returns the list of values at a specific electrode

        Parameters
        ----------
        i: int
            Index of the electrode

        Returns
        -------
        list
            The list of values of a specific electrode
        """
        return self.window[i]

    def getNormalizedComponentAt(self, i):
        """
        Returns the list of normalized values at specific electrode. The
        normalization is done simply by substracting the mean of the values to
        each value.

        Parameters
        ----------
        i: int
            Index of the electrode.

        Returns
        -------
        list
            The list of normalized values of a specific electrode.
        """
        return list(map(lambda x: x - self.means[i], self.getComponentAt(i)))

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
        return [win[n] for win in self.window]

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
    electrodeNumber: int
        The number of channels of samples the window will handle.
    window: SampleWindow
        It stores the data.
    """
    def __init__(self, windowSize, sampleRate, electrodeNumber, windowFunction = None):
        """
        Parameters
        ----------
        windowSize: int
            The maximun samples the window will store.
        sampleRate: int
            The number of samples per second
        electrodeNumber: int
            The number of channels of samples the window will handle.
        windowFunction: String, numpy.ndarray, optional
            This can be a String with the name of the function (currently only
            supported **"hamming"**) or it can be a numpy array with a size
            equals to the window size. ThIn the first case an array with the
            size of windowSize will be created. The created array will be
            multiplied by the data in the window.
        """
        self.windowSize = windowSize
        self.sampleRate = sampleRate
        self.electrodeNumber = electrodeNumber
        self.window = SampleWindow(windowSize, electrodeNumber)
        self.__handleWindowFunction(windowFunction, windowSize)

    # Function to handle the windowFunction parameter
    def __handleWindowFunction(self, windowFunction, windowSize):
        if windowFunction is None:
            self.windowFunction = np.ones(windowSize)
        elif type(windowFunction) == str:
            if windowFunction is "hamming":
                self.windowFunction = np.hamming(windowSize)
            else:
                raise ValueError("the option chosen is not valid")
        elif type(windowFunction) == np.ndarray:
            if len(windowFunction) == windowSize:
                self.windowFunction = windowFunction
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
            nSamples X nElectrodes. If the given data is the inverse,
            columnMode should be True.
        """
        self.window.set(samples, columnMode)

    def add(self, sample):
        """
        Adds a sample to the begining of the window and pops the last one and
        recalculates the means.

        Parameters
        ----------
        sample: array_like
            One sample of data
        """
        self.window.add(sample)

    def getRawDataAt(self, i):
        """
        Returns the raw data stored at the given index of the windows.

        Parameters
        ----------
        i: int
            Index of the electrode.

        Returns
        -------
        list
            The list of values of a specific electrode.
        """
        return self.window.getComponentAt(i)

    # Gets the normalized data values of the component i
    def getNormalizedDataAt(self, i):
        """
        Returns the data stored after being normalized at the given index of the
        windows.

        Parameters
        ----------
        i: int
            Index of the electrode.

        Returns
        -------
        list
            The list of normalized values of a specific electrode.
        """
        return self.window.getNormalizedComponentAt(i)

    def getFourierTransformAt(self, i):
        """
        Returns the Discrete Fourier Transform of the data at a given index of
        the window.

        Parameters
        ----------
        i: int
            Index of the electrode.

        Returns
        -------
        numpy.ndarray
            An array with the result of the Fourier Transform.
        """
        return np.fft.fft(self.windowFunction * self.window.getNormalizedComponentAt(i))

    # Gets the magnitude of each complex value resulting from a Fourier
    # Transform from a component i
    def getMagnitudesAt(self, i):
        """
        Returns the magnitude of each complex value resulting from a Discrete
        Fourier Transform at a the given index of the window.

        Parameters
        ----------
        i: int
            Index of the electrode.

        Returns
        -------
        numpy.ndarray
            An array with the magnitudes of the Fourier Transform.
        """
        return abs(self.getFourierTransformAt(i))

    def getAverageBandValuesAt(self, i, bands=defaultBands):
        """
        Returns the average magnitude of each band at the given index of the
        window

        Parameters
        ----------
        i: int
            Index of the electrode.

        bands: dict, optional
            This parameter is used to indicate the bands that are going to be
            used. It is a dict with the name of each band as key and a tuple
            with the lower and upper bounds as value.

        Returns
        -------
        dict
            The keys are the name of each band and the values are the mean of
            the magnitudes.
        """
        magnitudes = self.getMagnitudesAt(i)
        bandsValues = {}
        for key in bands:
            bounds = self.getBoundsForBand(bands[key])
            bandsValues[key] = np.mean(
                magnitudes[bounds[0]:bounds[1]] / self.windowSize)

        return bandsValues

    def getAverageBandValues(self, bands=defaultBands):
        """
        Returns the average magnitude of each band of every electrode.

        Parameters
        ----------
        bands: dict, optional
            This parameter is used to indicate the bands that are going to be
            used. It is a dict with the name of each band as key and a tuple
            with the lower and upper bounds as value.

        Returns
        -------
        list of dict
            Each dict of the list corresponds to an electrode and in each dict
            the keys are the name of each band and the values are the mean of
            the magnitudes.
        """
        return [self.getAverageBandValuesAt(i) for i in range(self.electrodeNumber)]

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
        return tuple(map(lambda val: int(val * self.windowSize / self.sampleRate), bandBounds))

    def getBandsSignalsAt(self, i, bands=defaultBands):
        """
        Rebuilds the signal from a component i but only in the specified
        frequency bands.

        Parameters
        ----------
        i: int
            Index of the electrode.

        bands: dict, optional
            This parameter is used to indicate the bands that are going to be
            used. It is a dict with the name of each band as key and a tuple
            with the lower and upper bounds as value.

        Returns
        -------
        dict of numpy.ndarray
            The keys are the same keys the bands dictionary is using. The values
            are the signal descomposed in every band at the given index of the
            window.
        """
        fft = self.getFourierTransformAt(i)
        bandsSignals = {}
        for key in bands:
            bounds = self.getBoundsForBand(bands[key])
            bandsSignals[key] = rebuildSignalFromDFT(fft, bounds)

        return bandsSignals

    def getBandsSignals(self, bands=defaultBands):
        """
        Rebuilds the signal from all componets but only in the specified
        frequency bands.

        Parameters
        ----------
        bands: dict, optional
            This parameter is used to indicate the bands that are going to be
            used. It is a dict with the name of each band as key and a tuple
            with the lower and upper bounds as value.

        Returns
        -------
        list numpy.ndarray
            Each element of the list contains the signal descomposed in every
            band at the corresponding index of the window. The keys are the same
            keys the bands dictionary is using.
        """
        return [self.getBandSignalsAt(i) for i in range(self.electrodeNumber)]

    def getPFDAt(self, i):
        """
        Returns the Petrosian Fractal Dimension at the given index of the
        window.

        Parameters
        ----------
        i: int
            Index of the electrode

        Returns
        -------
        float
            The resulting value
        """
        derivative = np.gradient(self.getRawDataAt(i))
        return np.log(self.windowSize) / (np.log(self.windowSize) + np.log(self.windowSize / (self.windowSize + 0.4 * countSignChanges(derivative))))

    def getHFDAt(self, i, kMax=None):
        """
        Returns the Higuchi Fractal Dimension at the given index of the window.

        Parameters
        ----------
        i: int
            Index of the electrode

        kmax: int, optional
            By default it will be windowSize//2.

        Returns
        -------
        float
            The resulting value
        """
        X = self.getRawDataAt(i)
        L, x = [], []
        N = len(X)
        kMax = N // 2 if kMax is None else kMax
        for k in range(1, kMax + 1):
            Lk = []
            for m in range(0, k):
                Lmk = 0
                for i in range(1, (N - m) // k):
                    Lmk += abs(X[m + i * k] - X[m + i * k - k])
                Lmk = Lmk * (N - 1) / (((N - m) // k) * k)
                Lk.append(Lmk)
            L.append(np.log(np.mean(Lk)))
            x.append([np.log(1 / k), 1])

        (p, r1, r2, s) = np.linalg.lstsq(x, L)
        return p[0]

    # Hjorth Parameters:
    def hjorthActivityAt(self, i):
        """
        Returns the Hjorth Activity at the given electrode

        Parameters
        ----------
        i: int
            Index of the electrode

        Returns
        -------
        float
            The resulting value
        """
        return hjorthActivity(self.getRawDataAt(i))

    def hjorthMobilityAt(self, i):
        """
        Returns the Hjorth Mobility at the given electrode

        Parameters
        ----------
        i: int
            Index of the electrode

        Returns
        -------
        float
            The resulting value
        """
        return hjorthMobility(self.getRawDataAt(i))

    def hjorthComplexityAt(self, i):
        """
        Returns the Hjorth Complexity at the given electrode

        Parameters
        ----------
        i: int
            Index of the electrode

        Returns
        -------
        float
            The resulting value
        """
        return hjorthComplexity(self.getRawDataAt(i))

    def synchronizationLikelihood(self, i1, i2, bandBounds=None, pRef=0.05):
        """
        Returns the Synchronization Likelihood value applied over the i1 and i2
        electrodes by calling :func:`~eeglib.eeg.synchronizationLikelihood`.

        Parameters
        ----------
        i1: int
            Index of the first electrode
        i2: int
            Index of the second electrode
        bandBounds: tuple or list, optional
            Lower and upper bounds in wich the signal will be rebuilded. If no
            bounds especified the algorithm is applied over the raw data.
        pRef: float, optional
            The p Ref param of the synchronizationLikelihood. Default 0.05

        Returns
        -------
        float
            The resulting value
        """
        if bandBounds is None:
            l = 1
            m = 16
            c1, c2 = self.getRawDataAt(i1), self.getRawDataAt(i2)
        else:
            bounds = self.getBoundsForBand(bandBounds)
            l = self.sampleRate // (3 * bounds[1])
            m = int(3 * bounds[1] / bounds[0])
            c1, c2 = rebuildSignalFromDFT(self.getFourierTransformAt(
                i1), bounds), rebuildSignalFromDFT(self.getFourierTransformAt(i2), bounds)
        l = 1 if l == 0 else l
        w1 = int(2 * l * (m - 1))
        w2 = int(10 // pRef + w1)
        return synchronizationLikelihood(c1, c2, m, l, w1, w2, pRef)

    def engagementLevel(self):
        """
        Returns the engagament level, which is calculated with this formula:
        beta/(alpha+theta), where alpha, beta and theta are the average of the
        average band values between al the electrodes.

        Returns
        -------
        float
            The engagement level.
        """
        bandValues=self.getAverageBandValues()
        alphas, betas, thetas=[],[],[]
        for d in bandValues:
            alphas.append(d["alpha"])
            betas.append(d["beta"])
            alpha.append(d["theta"])

        alpha, beta, theta= np.mean(alphas), np.mean(betas), np.mean(thetas)

        return beta/(alpha+theta)


def rebuildSignalFromDFT(dft, bounds=None):
    """
    Return a rebuilded signal given the Discrete Fourier Transform having the
    option of giving the frequency bounds.

    Parameters
    ----------
    dft: numpy.ndarray
        A Discrete Fourier Transform
    bounds: tuple or list, optional
        The values in index out of the specified bounds are set to 0 and later
        the IDFT is calculated.

    Returns
    -------
    numpy.ndarray
        The Inverse Discrete Fourier Transform of the input dft
    """
    if bounds is None:
        return np.fft.ifft(dft)

    auxArray = np.zeros(len(dft), dtype=complex)
    for i in range(bounds[0], bounds[1]):
        auxArray[i] = dft[i]
    return np.fft.ifft(auxArray)


def synchronizationLikelihood(c1, c2, m, l, w1, w2, pRef=0.05):
    """
    Returns the Synchronization Likelihood between c1 and c2. This is a
    modified version of the algorithm.

    Parameters
    ----------
    c1: array_like
        First signal
    c2: array_like
        second signal
    m: int
        Numbers of elements of the embedded vectors.
    l: int
        Separation between elements of the embedded vectors.
    w1: int
        Theiler correction for autocorrelation effects
    w2: int
        A window that sharpens the time resolution of the Synchronization
        measure

    Returns
    -------
    float
        A value between 0 and 1. 0 means that the signal are not synchronized
        at all and 1 means that they are totally synchronized.
    """
    X1 = __getEmbeddedVectors(c1, m, l)
    E1 = [__getEpsilon(X1, i, pRef) for i in range(len(X1))]
    X2 = __getEmbeddedVectors(c2, m, l)
    E2 = [__getEpsilon(X2, i, pRef) for i in range(len(X2))]

    size = len(X1)

    SL = 0
    SLMax = 0
    for i in range(size):
        Sij = 0
        SijMax = 0
        for j in range(size):
            if w1 < abs(j - i) < w2:
                if np.linalg.norm(X1[i] - X1[j]) < E1[i]:
                    if np.linalg.norm(X2[i] - X2[j]) < E2[i]:
                        Sij += 1
                    SijMax += 1
        SL += Sij
        SLMax += SijMax
    return SL / SLMax



# Auxiliar functions for Synchronization Likeihood
def __getHij(X, i, e):
    summ = 0
    for j in range(len(X)):
        if np.linalg.norm(X[i] - X[j]) < e:
            summ += 1
    return summ


def __getProbabilityP(X, i, e):
    return __getHij(X, i, e) / len(X)


def __getEmbeddedVectors(x, m, l):
    X = []
    size = len(x)
    for i in range(size - (m - 1) * l):
        X.append(np.array(x[i:i + m * l:l]))

    return X


def __getEpsilon(X, i, pRef, iterations=20):
    eInf = 0
    eSup = None
    e = 1
    p = 1
    minP = 1 / len(X)
    for _ in range(iterations):
        p = __getProbabilityP(X, i, e)
        if pRef < minP == p:
            break
        elif e < 0.0001:
            break
        elif p < pRef:
            eInf = e
        elif p > pRef:
            eSup = e
        else:
            break
        e = e * 2 if eSup is None else (eInf + eSup) / 2
    return e


# # Detrented Fluctuation Analysis
# def dfa():
#     pass

def getComponent(i, data):
    """
    Returns a list with the i component of each list contained in the list
    given. This method assumes that data is a 2-dimensional array_like.

    Parameters
    ----------
    i: int
        The index from which the new list will be obtained
    data: 2D array_like
        The data from which the new list will be obtained

    Returns
    -------
    list
    """
    return list(map(lambda row: row[i], data))


def countSignChanges(data):
    """
    Returns the number of sign changes of a 1D array

    Parameters
    ----------
    data: array_like
        The data from which the sign changes will be counted

    Returns
    -------
    int
        Number of sign changes in the data
    """
    signChanges = 0
    for i in range(1, len(data)):
        if data[i] * data[i - 1] < 0:
            signChanges += 1
    return signChanges


# HjorthParameters
def hjorthActivity(data):
    """
    Returns the Hjorth Activity of the given data

    Parameters
    ----------
    data: array_like

    Returns
    -------
    float
        The resulting value
    """
    return np.var(data)


def hjorthMobility(data):
    """
    Returns the Hjorth Mobility of the given data

    Parameters
    ----------
    data: array_like

    Returns
    -------
    float
        The resulting value
    """
    return np.sqrt(np.var(np.gradient(data)) / np.var(data))


def hjorthComplexity(data):
    """
    Returns the Hjorth Complexity of the given data

    Parameters
    ----------
    data: array_like

    Returns
    -------
    float
        The resulting value
    """
    return hjorthMobility(np.gradient(data)) / hjorthMobility(data)
