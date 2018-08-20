#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains helper classes that are useful to iterating over a EEG
data stream. Currently there is support only for CSV files.
"""
from abc import ABCMeta
import datetime

import csv
from pyedflib import EdfReader

import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import FastICA

from eeglib.eeg import EEG
from eeglib.preprocessing import bandPassFilter


class Helper(metaclass=ABCMeta):
    """
    This is an abstract class that defines the way every helper works.
    """
    
    def __init__(self, windowSize=None, highpass=None, 
                 lowpass=None, normalize=False, ICA=False,
                 selectedSignals=None):
        """
        Parameters
        ----------
        windowSize: int, optional
            The size of the window in which the calculations will be done. By
            default its value is the lenght of the data.
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
        if selectedSignals:
            self.selectSignals(selectedSignals)
        
        self.nChannels = len(self.data)
        self.nSamples = len(self.data[0])
        self.startPoint = 0
        self.endPoint = self.nSamples
        self.step=None
        self.iterator = None
        self.duration = self.nSamples/self.sampleRate
       
        if not windowSize:
            self.windowSize = self.sampleRate
        else:
            self.windowSize = windowSize
            
        self.prepareEEG(self.windowSize)
        
        if lowpass or highpass:
            for i,channel in enumerate(self.data):
                self.data[i]=bandPassFilter(channel,self.sampleRate,highpass,
                         lowpass)
        
        if ICA:
            ica=FastICA()
            self.data=ica.fit_transform(self.data.transpose()).transpose()
            self.names = [str(i) for i in range(len(self.nChannels))]
        
        if normalize:
            self.data=zscore(self.data,axis=1)
    
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
        if type(i) is not slice:
            raise ValueError("only slices can be used.")
        return self.prepareIterator(i.step, i.start, i.stop)
    
    def _getIterator(self):
        if not self.iterator:
            self.iterator = Iterator(self, self.step,
                                     self.startPoint,
                                     self.endPoint)
            
        return self.iterator
    
    def _handleIndex(self, index, step=False):
        if type(index) is int:
            if index<0 and not step:
                index = self.nSamples+index
        
        elif type(index) is str:
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
                
        elif type(index) is datetime.timedelta:
            index = int(self.sampleRate*index.total_seconds())
            
        else:
            raise ValueError("A index can only be a int or a str.")
            
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
        if not self.step:
            raise Exception("prepareEEG method must be called before \
                            prepareIterator.")
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
        if not self.step:
            self.step = windowSize
        return self.eeg

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
        startPoint=int(startPoint)
        if startPoint+self.eeg.windowSize>self.nSamples:
            raise ValueError("The start point is too near of the end.")
        else:
            self.eeg.set(self.data[:,startPoint:startPoint+self.eeg.windowSize]
                         ,columnMode=True)
        return self.eeg
    
    def getEEG(self):
        """
        Returns the EEG object.

        Returns
        -------
        EEG
        """
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
        for i, column in enumerate(selectedSignals):
            if type(selectedSignals[i]) is str:
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
        
    def __iter__(self):
        return self

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
    def __init__(self, path, *args, sampleRate=None, **kargs):
        """
        The rest of parameters can be seen at :meth:`Helper.__init__`
        
        Parameters
        ----------
        path: str
            The path to the csv file
        sampleRate: numeric, optional
            The frequency at which the data was recorded. By default its value
            is the lenght of the data.
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
            self.names = [str(i) for i in range(len(l1))]
        except ValueError:
            self.names=l1
        
        self.data=np.array(self.data)
        
        if not sampleRate:
            self.sampleRate=len(self.data[0])
        else:
            self.sampleRate=sampleRate
        
        super().__init__(*args,**kargs)
        

class EDFHelper(Helper):
    """
    This class is for applying diferents operations using the EEG class over an
    edf file.
    """
    def __init__(self, path, *args, **kargs):
        """
        The rest of parameters can be seen at :meth:`Helper.__init__`
        
        Parameters
        ----------
        path: str
            The path to the edf file
        """
        reader = EdfReader(path)
        
        ns = reader.signals_in_file
        
        self.data  = [reader.readSignal(i) for i in range(ns)]
        self.names = reader.getSignalLabels()
        frequencies = reader.getSampleFrequencies()
        
        self.sampleRate = frequencies[0]
        if not all(frequencies==self.sampleRate):
            raise ValueError("All channels must have the same frequency.")
        self.data=np.array(self.data)
        
        super().__init__(*args,**kargs)