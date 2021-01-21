#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains helper classes that are useful to iterating over a EEG
data stream. Currently there is support only for CSV files.
"""
import datetime

import csv
from pyedflib import EdfReader

import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import FastICA

from eeglib.eeg import EEG
from eeglib.preprocessing import bandPassFilter


class Helper():
    """
    This is an abstract class that defines the way every helper works.
    """

    def __init__(self, data, sampleRate=None, windowSize=None, names=None,
                 highpass=None, lowpass=None, normalize=False, ICA=False,
                 selectedSignals=None):
        """
        Parameters
        ----------
        data: 2D matrix
            The signals data in the shape (nChannels, nSamples).
        sampleRate: numeric, optional
            The frequency at which the data was recorded. By default its value
            is the lenght of the data.
        windowSize: int, optional
            The size of the window in which the calculations will be done. By
            default its value is the lenght of one second of the data.
        names: list of strings
            A list containing the names of each channel in the same positions
            than data channels.
        highpass: numeric, optional
            The signal will be filtered above this value.
        lowpass: numeric, optional
            The signal will be filtered bellow this value.
        normalize: boolean, optional
            If True, the data will be normalizing using z-scores. Default =
            False.
        ICA: boolean, optional
            If True, Independent Component Analysis will be applied to the data
            . It is applied always after normalization if normalize = True.
            Default: False.
        selectedSignals: list of strings or ints
            If the data file has names asociated to each columns, those columns
            can be selected through the name or the index of the column. If the
            data file hasn't names in the columns, they can be selected just by
            the index.
        """

        self.data = data

        #Names check
        if names:
            self.names = names
        else:
            self.names = [str(i) for i in range(data.shape[0])]

        #Check selected signals
        if selectedSignals:
            self.selectSignals(selectedSignals)

        #SampleRate check
        if not sampleRate:
            self.sampleRate=len(self.data[0])
        else:
            self.sampleRate=sampleRate

        #windowSize check
        if not windowSize:
            self.windowSize = self.sampleRate
        else:
            self.windowSize = windowSize

        #Attributes inicialization
        self.nChannels = len(self.data)
        self.nSamples = len(self.data[0])
        self.startPoint = 0
        self.endPoint = self.nSamples
        self.step=None
        self.iterator = None
        self.duration = self.nSamples/self.sampleRate

        #Preparing inner EEG ovject
        self.prepareEEG(self.windowSize)

        #Lowpass and highpass check
        if lowpass or highpass:
            for i,channel in enumerate(self.data):
                self.data[i]=bandPassFilter(channel, self.sampleRate, highpass,
                                            lowpass)

        #ICA check
        if ICA:
            ica=FastICA()
            self.data=ica.fit_transform(self.data.transpose()).transpose()
            self.names = [str(i) for i in range(self.nChannels)]

        #normilze check
        if normalize:
            self.data=zscore(self.data,axis=1)

        #Setting an initial value for the EEG window
        self.moveEEGWindow(self.startPoint)

    def __iter__(self):
        return self._getIterator()

    def __len__(self):
        return self.nSamples

    def __getitem__(self, i):
        """
        Creates and return a ready iterator thet using the slice given as
        parameter.

        Parameters
        ----------
        i: slice
            The fields of the slice are used to create the iterator.

        Returns
        ----------
        Iterator
        """
        if not isinstance(i, slice):
            raise ValueError("Only slices can be used.")
        return self.prepareIterator(i.step, i.start, i.stop)

    def _getIterator(self):
        if not self.iterator:
            self.iterator = Iterator(self, self.step,
                                     self.startPoint,
                                     self.endPoint)

        return self.iterator

    def _handleIndex(self, index, step=False):
        #Is integer?
        if np.issubdtype(type(index), np.integer):
            if index<0 and not step:
                index = self.nSamples+index
        #Is string?
        elif isinstance(index, str):
            neg = index[0]=="-"

            if neg:
                index = index.replace("-","")

            index = index.split(":")
            nComponents = len(index)

            if 0 < nComponents < 4:
                totalSecs = 0
                for i in range(nComponents):
                    totalSecs+=float(index[-(i+1)])*60**i
                index = int(totalSecs*self.sampleRate)

                if neg:
                    index = self.nSamples-index

            else:
                raise ValueError("Wrong value for the index")
        #Is timedelta?
        elif isinstance(index, datetime.timedelta):
            index = int(self.sampleRate*index.total_seconds())

        else:
            raise ValueError("An index can only be a int, a str or a " +
                             "datetime.timedelta.")

        return index

    def prepareIterator(self, step=None, startPoint=0, endPoint=None):
        """
        Prepares the iterator of the helper.

        Parameters
        ----------
        step: int,optional
            Number of samples to be skipped in each iteration.
        startPoint: int, optional
            The index of first sample from where the iteration will start. By
            default 0.
        endPoint: int, optional
            The index of the last sample + 1 until where the iteration will go.
            By default the size of the data.
        """
        if startPoint:
            self.startPoint=self._handleIndex(startPoint)
        if endPoint:
            self.endPoint = self._handleIndex(endPoint)
        if step:
            self.step = self._handleIndex(step, step=True)

        self.iterator = Iterator(self,self.step,self.startPoint,self.endPoint)

        return self


    def prepareEEG(self, windowSize):
        """
        Prepares and creates the EEG object that the iteration will use with
        the same parameters that an EEG objects is initialized. Also it returns
        the inner eeg object.

        Parameters
        ----------
        windowSize: int
            The maximun samples the window will store.

        Returns
        -------
        EEG
        """
        self.windowSize = windowSize
        self.eeg = EEG(windowSize, self.sampleRate, self.nChannels,
                       names=self.names)
        self.windowPosition = None
        if not self.step:
            self.step = windowSize
        return self.eeg

    def moveEEGWindow(self, position):
        """
        Moves the window to start at position. Also it returns the inner eeg
        object.

        Parameters
        ----------
        position: int, str or datetime.timedelta
            * int: position of the sample
            * str with format hh:mm:ss.ss: temporal position
            * timedelta: temporal position

        Returns
        -------
        EEG
        """
        position =self._handleIndex(position)
        if position+self.eeg.windowSize>self.nSamples:
            raise ValueError("The start point is too near to the end.")

        self.eeg.set(self.data[:, position:position + self.eeg.windowSize]
                     ,columnMode=True)
        self.windowPosition = position
        return self.eeg


    def getNames(self, indexes=None):
        """
        Returns the names of the specified indexes of channels.

        Parameters
        ----------
        indexes: Iterable of int, optional.
            The indexes of the channels desired. If None it will return all the
            channels' names. Default: None.
        """
        if indexes:
            names = [self.names[i] for i in indexes]
        else:
            names = self.names

        return names

    def selectSignals(self, selectedSignals):
        """
        Parameters
        ----------
        selectedSignals : iterable of str and/or int
            The indexes of the desired selected signals. It must be an iterable
            type containing either str or int.

        Returns
        -------
        None.

        """
        for i, column in enumerate(selectedSignals):
            if isinstance(selectedSignals[i], str):
                selectedSignals[i]=self.names.index(column)

        self.names=[self.names[i] for i in selectedSignals]
        self.data=np.array([self.data[i] for i in selectedSignals])
        self.nChannels = len(self.names)

        if hasattr(self, "windowSize"):
            self.prepareEEG(self.windowSize)

class Iterator():
    def __init__(self, helper,step,auxPoint, endPoint):
        self.helper=helper
        self.step=step
        self.auxPoint=auxPoint
        self.endPoint=endPoint

    # Function for iterations
    def __next__(self):
        if self.auxPoint > self.endPoint-self.helper.eeg.windowSize:
            raise StopIteration
        self.helper.moveEEGWindow(self.auxPoint)
        self.auxPoint += self.step
        return self.helper.eeg

class CSVHelper(Helper):
    """
    This class is for applying diferents operations using the EEG class over a
    csv file.
    """
    def __init__(self, path, *args, **kargs):
        """
        The rest of parameters can be seen at :meth:`Helper.__init__`

        Parameters
        ----------
        path: str
            The path to the csv file
        """
        with open(path) as file:
            reader=csv.reader(file)
            l1=reader.__next__()
            data=[[] for _ in l1]
            for row in reader:
                for i, val in enumerate(row):
                    data[i].append(float(val))
        #Checking if first row is for channels names
        try:
            l1=list(map(float,l1))
            for value,column in zip(l1, data):
                column.insert(0,value)
            names = None
        except ValueError:
            names=l1

        data=np.array(data)

        if "names" in kargs:
            super().__init__(data, *args, **kargs)
        else:
            super().__init__(data, *args, names=names, **kargs)


class EDFHelper(Helper):
    """
    This class is for applying diferents operations using the EEG class over an
    edf file.
    """
    def __init__(self, path, *args, sampleRate=None, **kargs):
        """
        The rest of parameters can be seen at :meth:`Helper.__init__`

        Parameters
        ----------
        path: str
            The path to the edf file
        """
        reader = EdfReader(path)

        ns = reader.signals_in_file

        data  = [reader.readSignal(i) for i in range(ns)]
        names = reader.getSignalLabels()
        frequencies = reader.getSampleFrequencies()

        if not sampleRate:
            sampleRate = frequencies[0]
            if not all(frequencies==sampleRate):
                raise ValueError("All channels must have the same frequency.")

        data=np.array(data)

        super().__init__(data, *args, sampleRate = sampleRate, names = names,
             **kargs)
