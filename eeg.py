import numpy as np
import scipy.integrate as integrate

#Default bands ranges
defaultBands={"delta":(1,4),"theta":(4,7), "alpha":(8,12), "beta": (12,30)}


				
#Class for storing signal samples
class SampleWindow:
	#Takes as input the window size and the number of electrodes
	def __init__(self,windowSize,electrodeNumber):
		self.windowSize=windowSize
		self.electrodeNumber=electrodeNumber
		
		self.window=[[0 for i in range(windowSize)] for i in range(electrodeNumber)]
		self.means=[0 for j in range(electrodeNumber)]
	
	#Adds a sample to the begining of the window and pops the last one and recalculates the means
	def add(self,sample):
		if hasattr(sample,"__getitem__") and len(sample) == self.electrodeNumber:
			for i in range(self.electrodeNumber):
				poped=self.window[i].pop()
				self.window[i].insert(0,sample[i])
				self.means[i]+=(sample[i]-poped)/self.windowSize
		else:
			raise ValueError("sample must be a subscriptable object with a length equals to electrodeNumber ("+str(self.electrodeNumber)+")")
	
	#Sets multiple samples at a time. The sample number must be the same as the windows size
	def set(self,samples,columnMode=False):
		if hasattr(samples,"__getitem__") and hasattr(samples[0],"__getitem__"):
			if (len(samples) == self.windowSize and len(samples[0]) == self.electrodeNumber and not columnMode) or (len(samples) == self.electrodeNumber and len(samples[0]) == self.windowSize and columnMode):
				for i in range(self.electrodeNumber):
					if not columnMode:
						self.window[i]=getComponent(i,samples)
					else:
						self.window[i]=list(samples[i])
					self.means[i]=np.mean(self.window[i])
			else:
				raise ValueError("the number of samples must be equal to the window size and each sample length must be a equals to electrodeNumber ("+str(self.electrodeNumber)+") if not in columnMode and viceversa if in columnMode")
		else:
			raise ValueError("samples must be a subscriptable object wich contains subcriptable objects")

	#Returns a list of a especific electrode
	def getComponentAt(self,i):
		return self.window[i]
	
	def getNormalizedComponentAt(self,i):
#		return list(map(lambda v:v-self.means[i],self.getComponentAt(i)))
		return list(map(lambda x: x-self.means[i],self.getComponentAt(i)))
	#Class for iterations
	class SampleWindowIterator:
		def __init__(self,iterWindow):
			self.iterWindow=iterWindow
			self.i=0
		def __next__(self):
			if self.i<len(self.iterWindow):
				self.i+=1
				return self.iterWindow[self.i-1]
			else:
				raise StopIteration
	#Returns the iteration class
	def __iter__(self):
		return self.SampleWindowIterator(self)
	
	def __getitem__(self,n):
		return [win[n] for win in self.window]
	
	#Returns a string representation of the inner list
	def __str__(self):
		return str(self.window)

#Class for calculating EEG features
class EEG:
	def __init__(self,windowSize,sampleRate, electrodeNumber,windowFunction=None):
		self.windowSize=windowSize
		self.sampleRate=sampleRate
		self.electrodeNumber=electrodeNumber
		self.window=SampleWindow(windowSize,electrodeNumber)
		self.__handleWindowFunction__(windowFunction,windowSize)
	
	#Function to handle the windowFunction parameter	
	def __handleWindowFunction__(self, windowFunction, windowSize):
		if windowFunction == None:
			self.windowFunction=np.ones(windowSize)
		elif type(windowFunction) == str:
			if windowFunction is "hamming":
				self.windowFunction = np.hamming(windowSize)
			else:
				raise ValueError("the option chosen is not valid")
		elif type(windowFunction) == np.ndarray:
			if len(windowFunction) == windowSize:
				self.windowFunction=windowFunction
			else:
				raise ValueError("the size of windowFunction is not the same as windowSize")
			
		else:
			raise ValueError("not a valid type for windowFunction")
	
	#Sets multiple samples at a time. The sample number must be the same as the windows size
	def set(self,samples,columnMode=False):
		self.window.set(samples, columnMode)
	
	#Adds a sample into the window
	def add(self,sample):
		self.window.add(sample)
	
	#Gets the raw data values of the component i
	def getRawDataAt(self,i):
		return self.window.getComponentAt(i)
		
	#Gets the normalized data values of the component i
	def getNormalizedDataAt(self,i):
		return self.window.getNormalizedComponentAt(i)
		
	#Gets the complex value resulting from a Fourier Transform from a component i
	def getFourierTransformAt(self,i):
		return np.fft.fft(self.windowFunction*self.window.getNormalizedComponentAt(i))
	
	#Gets the magnitude of each complex value resulting from a Fourier Transform from a component i
	def getMagnitudesAt(self,i):
		return abs(self.getFourierTransformAt(i))
	
	#Gets the average magnitude of each band from a component i
	def getAverageBandValuesAt(self,i,bands=defaultBands):
		magnitudes=self.getMagnitudesAt(i)
		bandsValues={}
		for key in bands:
			bounds=self.getBoundsForBand(bands[key])
			bandsValues[key]=np.mean(magnitudes[bounds[0]:bounds[1]]/self.windowSize)
#			bandsValues[key]=integrate.trapz(y = magnitudes[bounds[0]:bounds[1]]/self.windowSize,
#									 x = [v*self.sampleRate/self.windowSize for v in range(bounds[0],bounds[1])])
		return bandsValues
	
	#Gets the average magnitude of each band from all components
	def getAverageBandValues(self,bands=defaultBands):
		return [self.getAverageBandValuesAt(i) for i in range(self.electrodeNumber)]
	
	def getBoundsForBand(self,bandBounds):
		return tuple(map(lambda val:int(val*self.windowSize/self.sampleRate),bandBounds))
	

	
	#Rebuilds the signal from a component i but only in the specified frequency bands
	def getBandsSignalsAt(self,i,bands=defaultBands):
		fft=self.getFourierTransformAt(i)
		bandsSignals={}
		for key in bands:
			bounds=self.getBoundsForBand(bands[key])
			bandsSignals[key]=rebuildSignalFromDFT(fft,bounds)
			
		return bandsSignals
	
	#Returns the rebuilded signals in the specified frequency bands of all the components
	def getBandsSignals(self,bands=defaultBands):
		return [self.getBandSignalsAt(i) for i in range(self.electrodeNumber)]
		
	#The Petrosian Fractal Dimension is an algorithm used to calculate the fractal dimension
	def getPFDAt(self,i):
		derivative=np.gradient(self.getRawDataAt(i))
		return np.log(self.windowSize)/(np.log(self.windowSize)+np.log(self.windowSize/(self.windowSize+0.4*countSignChanges(derivative))))
	
	#The Higuchi Fractal Dimension is an algorithm used to calculate the fractal dimension
	def getHFDAt(self,i,kMax=None):
		X=self.getRawDataAt(i)
		L,x = [],[]
		N = len(X)
		kMax=N//2 if kMax == None else kMax
		for k in range(1,kMax+1):
			Lk = []
			for m in range(0,k):
				Lmk = 0
				for i in range(1,(N-m)//k):
					Lmk += abs(X[m+i*k] - X[m+i*k-k])
				Lmk = Lmk*(N - 1)/(((N - m) // k)*k)
				Lk.append(Lmk)
			L.append(np.log(np.mean(Lk)))
			x.append([np.log(1 / k), 1])
		
		(p, r1, r2, s)=np.linalg.lstsq(x, L)
		return p[0]
	
	#Hjorth Parameters:
	#Hjorth Activity
	def hjorthActivityAt(self,i):
		return hjorthActivity(self.getRawDataAt(i))
	#Hjorth Mobility
	def hjorthMobilityAt(self,i):
		return hjorthMobility(self.getRawDataAt(i))
	#Hjorth Complexity
	def hjorthComplexityAt(self,i):
		return hjorthComplexity(self.getRawDataAt(i))
	
	#Synchronization likelihood
	def synchronizationLikelihood(self,i1,i2,bandBounds=None, pRef=0.05):
		if bandBounds == None:
			l=1
			m=16
			c1,c2=self.getRawDataAt(i1),self.getRawDataAt(i2)
		else:
			bounds=self.getBoundsForBand(bandBounds)
			l=self.sampleRate//(3*bounds[1])
			m=int(3*bounds[1]/bounds[0])
			c1,c2= rebuildSignalFromDFT(self.getFourierTransformAt(i1),bounds), rebuildSignalFromDFT(self.getFourierTransformAt(i2),bounds)
		l = 1 if l==0 else l
		w1=int(2*l*(m-1))
		w2=int(10//pRef+w1)
		return synchronizationLikelihood(c1,c2,self.sampleRate,m,l,w1,w2,pRef)

#Rebuilds a signal given the Discrete Fourier Transform having the option of giving the bounds
def rebuildSignalFromDFT(dft,bounds=None):
		if bounds==None: return np.fft.ifft(dft)
		
		auxArray=np.zeros(len(dft),dtype=complex)
		for i in range(bounds[0],bounds[1]):
			auxArray[i]=dft[i]
		return np.fft.ifft(auxArray)

#Synchronization Likeihood
def synchronizationLikelihood(c1,c2,sampleRate,m,l,w1,w2,pRef=0.05):
	X1=getEmbeddedVectors(c1,m,l)
	E1=[getEpsilon(X1,i,pRef) for i in range(len(X1))]
	X2=getEmbeddedVectors(c2,m,l)
	E2=[getEpsilon(X2,i,pRef) for i in range(len(X2))]
	
	size=len(X1)
	
	SL=0
	SLMax=0
	for i in range(size):
		Sij=0
		SijMax=0
#		wSize=0
		for j in range(size):
			if w1<abs(j-i)<w2:
				if np.linalg.norm(X1[i]-X1[j])<E1[i]:
					if np.linalg.norm(X2[i]-X2[j])<E2[i]:
						Sij+=1
					SijMax+=1
#				wSize+=1
#		SL+=0 if wSize==0 else Sij/wSize
#		SLMax+=0 if wSize==0 else SijMax/wSize
		SL+=Sij
		SLMax+=SijMax
	return SL/SLMax
	
#Auxiliar functions for Synchronization Likeihood	
def getHij(X,i,e):
	summ=0
	for j in range(len(X)):
		if np.linalg.norm(X[i]-X[j])<e: summ+=1
	return summ

def getProbabilityP(X,i,e):
	return getHij(X,i,e)/len(X)

def getEmbeddedVectors(x,m,l):
	X=[]
	size=len(x)
	for i in range(size-(m-1)*l):
		X.append(np.array(x[i:i+m*l:l]))
	
	return X
	
def getEpsilon(X,i,pRef,iterations=20):
	eInf=0
	eSup=None
	e=1
	p=1
	minP=1/len(X)
	for _ in range(iterations):
		p=getProbabilityP(X,i,e)
		if pRef<minP==p:
			break	
		elif e<0.0001:
			break
		elif p<pRef:
			eInf=e
		elif p>pRef:
			eSup=e
		else:
			break
		e = e*2 if eSup==None else (eInf+eSup)/2
	return e

#Detrented Fluctuation Analysis
def DFA():
	pass

#Returns a list with the i component of each list contained in the list given
def getComponent(i,data):
	return list(map(lambda row:row[i],data))

#Counts the sign changes of a data stream
def countSignChanges(data):
	signChanges=0
	for i in range(1,len(data)):
		if data[i]*data[i-1]<0: signChanges+=1
	return signChanges

#HjorthParameters
def hjorthActivity(data):
	return np.var(data)

def hjorthMobility(data):
	return np.sqrt(np.var(np.gradient(data))/np.var(data))

def hjorthComplexity(data):
	return hjorthMobility(np.gradient(data))/hjorthMobility(data)