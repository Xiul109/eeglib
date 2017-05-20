from abc import ABCMeta,abstractclassmethod

class Helper(metaclass=ABCMeta):
	@abstractclassmethod
	def __iter__(self): pass
	@abstractclassmethod
	def __next__(self): pass
	@abstractclassmethod
	def prepareIterator(self,step=None,startPoint=0,endPoint=None): pass
	@abstractclassmethod
	def prepareEEG(self, windowSize, sampleRate,windowFunction=None):pass
	@abstractclassmethod
	def getEEG(self):pass

import csv
from EEGlib.eeg import EEG
#This class is for appliying diferents operations using the EEG class over a csv
class CSVHelper(Helper):
    #Path is the path to the csv file
    def __init__(self,path):
        with open(path) as file:
            self.data=list(map(lambda x:list(map(lambda y:float(y),x)),csv.reader(file)))
            self.electrodeNumber=len(self.data[0])
            self.startPoint=0
            self.endPoint=len(self.data)

    #Function for iteratations
    def __iter__(self):
        self.auxPoint=self.startPoint
        return self
    
    #Function for iterations
    def __next__(self):
        if self.auxPoint>=self.endPoint:
            raise StopIteration
        self.moveEEGWindow(self.auxPoint)
        self.auxPoint+=self.step
        return self.eeg
    
    #This function prepares the object to be iterated from |startPoint| to |endPoint| by skyping |step|
    def prepareIterator(self,step=None,startPoint=0,endPoint=None):
        if endPoint!=None:
            self.endPoint=endPoint
        if step!=None:
            self.step=step
    
    #This function prepares the eeg object with a size of |windowSize|, having a sample rate of |sampleRate|
    #and using the specified |windowFunction|
    def prepareEEG(self, windowSize, sampleRate,windowFunction=None):
        self.eeg=EEG(windowSize,sampleRate,self.electrodeNumber,windowFunction=windowFunction)
        self.step=windowSize
        return self.eeg
    
    #This moves the current window to start at |startPoint|
    def moveEEGWindow(self, startPoint):
        self.eeg.set(self.data[startPoint:startPoint+self.eeg.windowSize])
        return self.eeg
    
    #This function returns the eeg object
    def getEEG(self):
        return self.eeg