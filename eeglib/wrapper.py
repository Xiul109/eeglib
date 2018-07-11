import pandas as pd
from eeglib.auxFunctions import flat

class Wrapper():
    """
    This class wraps around a helper and allows getting multiple features at
    once. The main usage of this class if for generating features to use with
    machine learning techniques.
    """
    def __init__(self, helper, flat = True, store=True):
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
            a dict or an array.
        """
        self.helper = helper
        self.functions = {}
        self.flat = flat
        self.store = store
        self.currentStored = False
        if store:
            self.storedFeatures = []
            
    
    def __iter__(self):
        self.iterator = iter(self.helper)
        return self
    
    def __next__(self):
        next(self.iterator)
        self.currentStored = False
        return self.getFeatures()

    def addFeature(self, name, *args, **kargs):
        """
        Adds a feature that will be returned in every
        
        Parameters
        ----------
        name: str
            The name of the function to call.
        *args and **kargs:
            Parameters of the function that computes the feature.
        
        Returns
        -------
        None
        """
        f = getattr(self.helper.eeg, name)
        self.functions[name+str(args)+str(kargs)]=(lambda: f(*args, **kargs))
    
    
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
        
        if self.currentStored:
            return self.storedFeatures[-1]
        
        features = {name:f() for name, f in self.functions.items()}
        if self.flat:
            features = flat(features,"")
        
        if self.store:
            self.storedFeatures.append(features)
            self.currentStored = True
        
        return pd.Series(features)

        
    def getAllFeatures(self):
        """
        Iterates over all the windows in the helper and returns all the values.
        
        Returns
        -------
        pandas.DataFrame
        """
        return pd.DataFrame([features for features in self])