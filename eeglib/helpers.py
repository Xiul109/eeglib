#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains helper classes that are useful to iterating over a EEG
data stream. currently there is support only for CSV files.
"""
from abc import ABCMeta, abstractclassmethod

import csv
from eeglib.eeg import EEG


class Helper(metaclass=ABCMeta):
    """
    This is an abstract class that defines the way every helper works. Before
    performing any iteration over the data, the methods :meth:`~eeglib.helpers.Helper.prepareIterator`
    and :meth:`~eeglib.helpers.Helper.prepareEEG` should be called first.
    """
    @abstractclassmethod
    def __iter__(self): pass

    @abstractclassmethod
    def __next__(self): pass

    @abstractclassmethod
    def __len__(self): pass

    @abstractclassmethod
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
        pass

    @abstractclassmethod
    def prepareEEG(self, windowSize, sampleRate, windowFunction=None):
        """
        Prepares and creates the EEG object that the iteration will use with
        the same parameters that an EEG objects is initialized. Also it returns
        the inner eeg object.

        Parameters
        ----------
        windowSize: int
            The maximun samples the window will store.
        sampleRate: int
            The number of samples per second
        windowFunction: String, numpy.ndarray, optional
            This can be a String with the name of the function (currently only
            supported **"hamming"**) or it can be a numpy array with a size
            equals to the window size. ThIn the first case an array with the
            size of windowSize will be created. The created array will be
            multiplied by the data in the window.
        
        Returns
        -------
        EEG
        """
        pass

    @abstractclassmethod
    def moveEEGWindow(self,startPoint):
        """
        Moves the window to start at startPoint. Also it returns the inner eeg
        object.
        
        Parameters
        ----------
        startPoint: int
        
        Returns
        -------
        EEG
        """
        pass
    
    @abstractclassmethod
    def getEEG(self):
        """
        Returns the EEG object.

        Returns
        -------
        EEG
        """
        pass


class CSVHelper(Helper):
    """
    This class is for appliying diferents operations using the EEG class over a
    csv file.
    """
    def __init__(self, path, selectedColumns=None):
        """
        Parameters
        ----------
        path: str
            The path to the csv file
        selectedFields: list of strings or ints
            If the data file has names asociated to each columns, those columns
            can be selected through the name or the index of the column. If the
            data file hasn't names in the columns, they can be selected just by
            the index.
        """
        with open(path) as file:
            reader=csv.reader(file)
            l1=reader.__next__()
            self.data=[[] for _ in l1]
            for row in reader:
                for i,val in enumerate(row):
                    self.data[i].append(float(val))
            try:
                l1=list(map(lambda x: float(x),l1))
                for value,column in zip(l1,self.data):
                    column.insert(0,value)
                self.names=None
            except ValueError:
                self.names=l1
            if selectedColumns:
                for i,column in enumerate(selectedColumns):
                    if type(selectedColumns[i]) is str:
                        selectedColumns[i]=self.names.index(column)
                self.names=[self.names[i] for i in selectedColumns]
                self.data=[self.data[i] for i in selectedColumns]
                
            self.electrodeNumber = len(self.data)
            self.nSamples = len(self.data[0])
            self.startPoint = 0
            self.endPoint = self.nSamples

    # Function for iterations
    def __iter__(self):
        self.auxPoint = self.startPoint
        return self

    # Function for iterations
    def __next__(self):
        if self.auxPoint >= self.endPoint:
            raise StopIteration
        self.moveEEGWindow(self.auxPoint)
        self.auxPoint += self.step
        return self.eeg

    def __len__(self):
        return self.nSamples

    def prepareIterator(self, step=1, startPoint=0, endPoint=None):
        "Go to :meth:`eeglib.helpers.Helper.prepareIterator`"
        self.startPoint=startPoint
        if endPoint is not None:
            self.endPoint = endPoint
        if step is not None:
            self.step = float(step)

    def prepareEEG(self, windowSize, sampleRate, windowFunction=None):
        "Go to :meth:`eeglib.helpers.Helper.prepareEEG`"
        self.eeg = EEG(windowSize, sampleRate,
                       self.electrodeNumber, windowFunction=windowFunction, names=self.names)
        self.step = windowSize
        return self.eeg

    # This moves the current window to start at |startPoint|
    def moveEEGWindow(self, startPoint):
        "Go to :meth:`eeglib.helpers.Helper.moveEEGWindow`"
        startPoint=int(startPoint)
        if startPoint+self.eeg.windowSize>self.nSamples:
            raise ValueError("The start point is too near of the end.")
        else:
            self.eeg.set([col[startPoint:startPoint + self.eeg.windowSize] for
                          col in self.data],columnMode=True)
        return self.eeg

    def getEEG(self):
        "Go to :meth:`eeglib.helpers.Helper.getEEG`"
        return self.eeg
