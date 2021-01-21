#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the Wrapper class that wraps around a Helper and computes
every feature in each window configured. It also allows specifying labels for
the whole signal or for specific segments.
"""
import pandas as pd
from eeglib.auxFunctions import flatData
from eeglib.eeg import EEG

class _FeatureAdder():
    """
    This class is used by the wrapper to add features defined in the EEG object
    to the proccesing.
    """

    _excludedFunctions = ["set", "getBoundsForBand"]

    # Init
    def __init__(self, wrapper):
        #Setting attributes
        self._wrapper = wrapper
        self._eeg = wrapper.helper.eeg

        #Iterating over EEG.__dict__ items
        for key, value in EEG.__dict__.items():
            if (key[0] != "_" and callable(value) and
                not key in self._excludedFunctions):
                # If the conditions are fullfilled the modified function is
                # added to the __dict__
                self.__dict__[key] = self._adderCreator(value)

    # Creator of adder functions
    def _adderCreator(self, f):
        """
        Creates an adder method. When an adder method is called, it adds a
        feature, that is obtained by applying f to a signal, to the wrapper.

        Parameters
        ----------
        f: function
            The function

        Returns
        -------
        function
        """
        def func(*args, name=None, hideArgs=False, **kargs):
            hadName = name
            if not name:
                name = f.__name__

            if not hideArgs and not hadName:
                name+=str(args)+str(kargs)

            self._wrapper.functions[name]=(lambda: f(self._eeg, *args,
                                                     **kargs))

        func.__name__ = f.__name__
        func.__doc__ = """
        **It adds the next feature to the wrapper.**

        %s
        """%f.__doc__

        return func

    def __call__(self, functionName, *args, name=None, hideArgs=False,
                 **kargs):
        """
        Adds a feature that will be included in the dataset.

        Parameters
        ----------
        functionName: str
            The name of the function to call.
        name: str
            A custom name for the feature that will be visible in the df.
        hideArgs: bool
            If True, the arguments are not attached to the name in the df.
            Default: False.
        *args and **kargs:
            Parameters of the function that computes the feature.

        Returns
        -------
        None
        """
        f = getattr(EEG, functionName)

        self._adderCreator(f)(*args, name=name, hideArgs=hideArgs, **kargs)

class Wrapper():
    """
    This class wraps around a helper and allows getting multiple features at
    once. The main usage of this class if for generating features to use with
    machine learning techniques.
    """
    def __init__(self, helper, flat = True, flatSeparator = "_", store=True,
                 label = None, segmentation = None):
        """
        Parameters
        ----------
        helper: :py:class:`eeglib.helpers.Helper`
            The helper object that will be used to get the data and the
            iteration settings.
        flat: Bool
            If True the result of calling getFeatures() will be a one
            dimensional sequence of data. If False, the return value will be
            a list containing the result of each feature, that can be a float,
            a dict or an array. Default: True.
        separator: str
            It is used to separate the features names when flatten.
        store: Bool
            If True, the data will be stored in a self.storedFeatures. Default:
            True.
        label: object, optional
            This value will be added as a field in the data except if it is
            None. Default: None.
        segmentation: dict of tuples, optional
            Parameter format [((begin, end),label), ]; begin is the begin of
            the segment, end is the end of the segment and label the label
            realed to that segment. Begin and end can be either an int, a str
            or a datetime.timedelta, being the same that the parameter of
            :py:meth:`eeglib.helpers.Helper.moveEEGWindow`. The label of the
            unlabelled segments will be 0. If None, no segmentation will be
            added. Default: None.
        """
        self.helper = helper
        self.iterator = iter(self.helper)

        self.functions = {}

        self.flat = flat
        self.flatSeparator = flatSeparator

        self.store = store
        self.lastStored = -1
        if store:
            self.storedFeatures = []

        self.label = label
        if segmentation:
            self.segmentation = [((self.helper._handleIndex(s0),
                                   self.helper._handleIndex(se)), label)
                                 for (s0, se), label in segmentation]
            self.segmentation = sorted(self.segmentation)
            self.segmentationIndex = 0
        else:
            self.segmentation = None

        self.addFeature = _FeatureAdder(self)

    def __iter__(self):
        self.iterator = iter(self.helper)
        return self

    def __next__(self):
        next(self.iterator)
        return self.getFeatures()

    def addFeatures(self, features):
        """
        Parameters
        ----------
        features: list(tuple(str,list, dict))
            A list containing tuples that represent the parameters needed to
            add a single feature.
        """
        for f in features:
            if isinstance(f,tuple) and len(f) == 3:
                self.addFeature(f[0],*f[1],**f[2])
            elif isinstance(f, str):
                self.addFeature(f)
            else:
                raise ValueError("Only tuples of len 3 or str are valid values")

    def addCustomFeature(self, function, channels = None, twoChannels = False,
                         name = None):
        """
        Adds a custom feature that will be included in the dataset.

        Parameters
        ----------
        function: function
            The function to apply.

        channels: Variable type, optional
            The channels over which the function will be applied.
            * int:  the index of the channel.
            * str:  the name of the channel.
            * list of strings and integers:  a list of channels.
            * slice:  a slice selecting the range of channels.
            * None: all the channels.
            The function to apply to the data to obtain the feature.

        twoChannels: bool
            If function receives two channels of data this parameter should be
            True. Default: False.
        name: str
            A custom name for the feature that will be visible in the df.

        Returns
        -------
        None
        """

        if not twoChannels:
            f = lambda: self.helper.eeg._applyFunctionTo(function, channels)
        else:
            f = lambda: self.helper.eeg._applyFunctionTo2C(function, channels)

        if not name:
            name = function.__name__

        self.functions[name] = f


    def featuresNames(self):
        """
        Returns the names of the specified features.
        """
        return self.functions.keys()


    def getFeatures(self):
        """
        Returns the features in the current window of the helper iterator.

        Returns
        -------
        pandas.Series
        """
        if self.lastStored == self.iterator.auxPoint:
            features = self.storedFeatures[-1]
        else:
            features = {name:f() for name, f in self.functions.items()}
            if self.flat:
                features = flatData(features,"", self.flatSeparator)

            if self.segmentation:
                loop = True
                features["segment_label"] = 0
                while loop and self.segmentationIndex < len(self.segmentation):
                    curSeg, curLab = self.segmentation[self.segmentationIndex]
                    curPoint = self.iterator.auxPoint - self.iterator.step
                    if curSeg[0] <= curPoint < curSeg[1]:
                        features["segment_label"] = curLab
                        loop = False
                    elif curPoint <curSeg[0]:
                        loop = False
                    else:
                        self.segmentationIndex+=1

            if self.label is not None:
                features["label"] = self.label

            if self.store:
                self.storedFeatures.append(features)
                self.lastStored = self.iterator.auxPoint

        return pd.Series(features)

    def getStoredFeatures(self):
        """
        Returns the stored features if store was set to True, else it returns
        None.
        """
        if self.store:
            return pd.DataFrame(self.storedFeatures)
        return None

    def getAllFeatures(self):
        """
        Iterates over all the windows in the helper and returns all the values.

        Returns
        -------
        pandas.DataFrame
        """
        data=[features for features in self]
        if self.store:
            data=self.storedFeatures
        return pd.DataFrame(data)

    def reset(self):
        """
        Resets the

        Returns
        -------
        None.

        """
        self.iterator = iter(self.helper)
