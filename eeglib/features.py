"This module define the functions of the features for EEG analysis"

import numpy as np
from scipy.spatial.distance import chebyshev
from numba import jit, float64,int64


def averageBandValues(fft,bands):
    """
    Given a Fourier Transform, this function returns it averaged across the
    bands selected. The final value is calculated by multiplying the mean of
    the data in the given bounds by 2/len(fft).
    
    Parameters
    ----------
    fft: array_like
        The result of applying the fft function to a signal
    bands: dict{key:tuple}
        The bands names(key) and bounds(value)
    
    Returns
    -------
    dict{key:float}
        The average value for each band
    """
    bandsValues = {}
    for key,bounds in bands.items():
        bandsValues[key] = (2 / len(fft)) * np.mean(
                fft[bounds[0]:bounds[1]])
    
    return bandsValues

def PFD(data):
    """
    Returns the Petrosian Fractal Dimension of the signal given in data.
    
    Parameters
    ----------
    data: array_like
        Signal
    
    Returns
    -------
    float
        The resulting value
    """
    derivative = np.diff(data)
    size=len(data)
    return np.log(size) / (np.log(size) + np.log(size / (size + 0.4 * countSignChanges(derivative))))

@jit
def HFD(data,kMax=None):
    """
    Returns the Higuchi Fractal Dimension of the signal given data.

    Parameters
    ----------
    data: array_like
        signal

    kMax: int, optional
        By default it will be windowSize//4.

    Returns
    -------
    float
        The resulting value
    """
    #Inicializations
    data=np.array(data) 
    N = len(data)
    kMax = N // 4 if kMax is None else kMax     #Default kMax
    L = np.zeros(kMax-1)
    x = np.array((-np.log(np.arange(2,kMax+1)),np.ones(kMax-1))).transpose()
    # Loop from 2 to kMax
    for k in range(2, kMax + 1):
        Lk = np.zeros(k)
        #Loop for compute the lenght of Lmk
        for m in range(0, k):
            Lmk = 0            
            for i in range(1, (N - m) // k):
                Lmk += abs(data[m + i * k] - data[m + i * k - k])
            Lk[m]=Lmk * (N - 1) / (((N - m) // k) * k * k)
        Laux=np.mean(Lk)
        Laux=0.01/k if Laux==0 else Laux
        L[k-2]=np.log(Laux)

    p, _, _, _ = np.linalg.lstsq(x, L)
    return p[0]

def synchronizationLikelihood(c1, c2, m, l, w1, w2, pRef=0.05, epsilonIterations=20):
    """
    Returns the Synchronization Likelihood between c1 and c2. This is a
    modified version of the original algorithm.

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
    pRef: float, optional
        The pRef param of the synchronizationLikelihood. Default 0.05
    epsilonIterations: int,optional
        Number of iterations used to determine the value of epsilon. Default:20

    Returns
    -------
    float
        A value between 0 and 1. 0 means that the signal are not synchronized
        at all and 1 means that they are totally synchronized.
    """
    if len(c1)!=len(c2):
        raise ValueError("c1 and c2 must have the same lenght")        

    return __SL(c1, c2, m, l, w1, w2, pRef,epsilonIterations)

# Auxiliar functions for Synchronization Likeihood
@jit(float64(float64[:],float64[:],int64,int64,int64,int64,float64,int64))
def __SL(c1, c2, m, l, w1, w2, pRef, epsilonIterations):
    X1 = __getEmbeddedVectors(c1, m, l)
    X2 = __getEmbeddedVectors(c2, m, l)
    
    D1 = __getDistances(X1)
    D2 = __getDistances(X2)
    
    size=len(X1)
    E1 = np.zeros(size) 
    E2 = np.zeros(size)
    for i in range(size):
        E1[i]=__getEpsilon(D1, i, pRef,epsilonIterations)
        E2[i]=__getEpsilon(D2, i, pRef,epsilonIterations)
    
    SL = 0
    SLMax = 0
    for i in range(size):
        Sij = 0
        SijMax = 0
        for j in range(size):
            if w1 < abs(j - i) < w2:
                if D1[i,j] < E1[i]:
                    if D2[i,j] < E2[i]:
                        Sij += 1
                    SijMax += 1
        SL += Sij
        SLMax += SijMax
    return SL / SLMax if SLMax>0 else 0

@jit(int64(float64[:,:],int64,float64))
def __getHij(D, i, e):
    summ = 0
    for j in range(len(D)):
        if D[i,j] < e:
            summ += 1
    return summ

@jit(float64[:,:](float64[:,:]))
def __getDistances(X):
    t=len(X)
    D=np.zeros((t,t),dtype=np.float)
    for i in range(t):
        for j in range(i):
            D[j,i]=D[i,j]=np.linalg.norm(X[i]-X[j])

    return D
@jit(float64(float64[:,:],int64,float64))
def __getProbabilityP(D, i, e):
    return __getHij(D, i, e) /len(D) 

@jit(int64[:,:](float64[:],int64,int64))
def __getEmbeddedVectors(x, m, l):
    size = len(x)- (m - 1) * l
    X = np.zeros((size,m))
    for i in range(size):
        X[i]=np.array(x[i:i + m * l:l])

    return X

@jit(float64(float64,float64))
def __logDiference(p1,p2):
    return abs(np.log(p2/p1))

@jit(float64(float64[:,:],int64,float64,int64))
def __getEpsilon(D, i, pRef, iterations):
    eInf = 0
    eSup = None
    bestE=e = 1
    bestP=p = 1
    minP = 1 / len(D)
    for _ in range(iterations):
        p = __getProbabilityP(D, i, e)
        if pRef < minP == p:
            break
        elif p < pRef:
            eInf = e
        elif p > pRef:
            eSup = e
        else:
            bestP=p
            bestE=e
            break
        if __logDiference(bestP,pRef) > __logDiference(p,pRef):
            bestP=p
            bestE=e
        e = e * 2 if eSup is None else (eInf + eSup) / 2
        
    return bestE


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


# Sample Entropy
def MSE(data, m = 2, l = 1, r = None, fr = 0.2, eps = 10e-2):
    """
    Returns Multiscale Sample Entropy of the given data.
    
    Parameters
    ----------
    data: array_like
        The signal  
    m: int, optional
        Size of the embedded vectors. By default 2.
    l: int, optional
        Lag beetwen elements of embedded vectors. By default 1.
    r: float, optional
        Tolerance. By default fr*std(data)
    fr: float, optional
        Fraction of std(data) used as tolerance. If r is passed, this
        parameter is ignored. By default, 0.2.
    eps: float, optional
        Small number added to avoid infinite results. If 0 infinite results can
        appear. Default: 10e-2.
    
    Returns
    -------
    float
        The resulting value    
    """
    if not r:
        r = fr * np.std(data)
    
    A = __countEmbeddedDistances(data, m+1, l, r) + eps
    B = __countEmbeddedDistances(data, m  , l, r) + eps
    
    return -np.log(A/B) if A != 0 else float("inf")

def __countEmbeddedDistances(data, m, l, r):
    X = __getEmbeddedVectors(data , m, l)
    
    D = np.array([chebyshev(X[i], X[i+1]) for i in range(len(X)-1)])
    
    return np.sum(D < r)


# Lempel-Ziv Complexity
def LZC(data, threshold = None):
    """
    Returns the Lempel-Ziv Complexity of the given data.
    
    Parameters
    ----------
    data: array_like
        The signal.
    theshold: numeric, optional
        A number use to binarize the signal. The values of the signal above
        threshold will be converted to 1 and the rest to 0. By default, the 
        median of the data.
    """
    if not threshold:
        threshold=np.median(data)
    
    sequence = __binarize(data, threshold)
    
    c = __LZC(sequence)
    
 
    n = len(data)
    lzc = c*(np.log2(c)+1)/n
    
    return lzc

def __LZC(sequence):
    subsequences = set()
    subsequence  = []
    for e in sequence:
        subsequence.append(e)
        if str(subsequence) not in subsequences:
            subsequences.add(str(subsequence))
            subsequence=[]
            
    return len(subsequences)

#def __minLZC(size):
#    return int((np.sqrt(1 + 8*size) - 1)/2)
#
#def __maxLZC(size):
#    v = 0
#    i = 1
#    n = 2
#    while n*i < size:
#        size-=n*i
#        v+=n
#        i += 1
#        n = 2**i
#    v += size // i
#    
#    return v

def __binarize(data, threshold):
    array = np.array(data)
    return np.array(array > threshold, int)


# Detrended Fluctuation Analysis
def DFA(data, fit_degree = 1, min_window_size = 4, max_window_size = None,
        fskip = 1):
    """
    Applies Detrended Fluctuation Analysis algorithm to the given data.
    
    Parameters
    ----------
    data: array_like
        The signal.
    fit_degree: int, optional
        Degree of the polynomial used to model de local trends. Default: 1.
    min_window_size: int, optional
        Size of the smallest window that will be used. Default: signalSize//2.
    fskip: float, optional
        Fraction of the window that will be skiped in each iteration for each
        window size. Default: 1.
    
    Returns
    -------
    float
        The resulting value
    """
    data = np.array(data)
    size=len(data)
        
    if not max_window_size:
        max_window_size = size//2
        
    Y = np.cumsum(data - np.mean(data))
    
    F = np.zeros(max_window_size - min_window_size)
    
    ns = np.arange(min_window_size, max_window_size)
    
    for n in ns:
        itskip = int(fskip * n)
        nWindows = int(np.ceil((size - n + 1) / itskip))
        Fn = np.zeros(nWindows)
        x = np.arange(n)
        for i in range(0, nWindows):
            y  = Y[i*n:i*n+n]
            p  = np.polyfit(x, y, fit_degree)
            yn = np.polyval(p, x)
            Fn[i] = np.sqrt(np.sum((y-yn)**2)/n)
        F[n-min_window_size] = np.mean(Fn)
    
    return np.polyfit(np.log(ns), np.log(F), 1)[0]

# Cross Correlation Coeficient
def CCC(c1, c2):
    """
    Computes the Cross Correlation Coeficient between the data in c1 and the 
    data in c2.
    
    Parameters
    ----------
    c1: array_like
        One of the signals
    c2: array_like
        The other signal
    
    Returns
    -------
    float
        The Cross Correlation Coeficient.
    """
    meanC1 =  np.mean(c1)
    meanC2 =  np.mean(c2)
    
    num = np.mean((c1 - meanC1)*(c2-meanC2))
    den = np.sqrt(np.var(c1)*np.var(c2))
    
    return num/den if num != 0 else 0