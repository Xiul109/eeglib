"This module define the functions for preprocessing the signal data"

import numpy as np

def bandPassFilter(data,sampleRate=None,highpass=None,lowpass=None):
    """
    Return the signal filtered between highpass and lowpass. Note that neither
    highpass or lowpass should be above sampleRate/2.

    Parameters
    ----------
    data: numpy.ndarray
        The signal
    sampleRate: numeric, optional
        The frequency at which the signal was recorded. By default it is the
        same as the number of samples of the signal.
    highpass: numeric, optional
        The signal will be filtered above this value.
    lowpass: numeric, optional
        The signal will be filtered bellow this value.

    Returns
    -------
    numpy.ndarray
        The signal filtered betwen the highpass and the lowpass
    """
    size=len(data)
    if not sampleRate:
        sampleRate=size
    else:
        if highpass:
            highpassP=int(highpass*size/sampleRate)
            highpassN=-highpassP
        else:
            highpassP=highpassN=None
        if lowpass:
            lowpassP=int(lowpass*size/sampleRate)
            lowpassN=-lowpassP
        else:
            lowpassP=lowpassN=None
            
    fft=np.fft.fft(data)
    
    window=np.zeros(size)
    window[highpassP:lowpassP]=1
    window[lowpassN:highpassN]=1
    
    filtered_fft=fft*window
    return np.real(np.fft.ifft(filtered_fft))  
