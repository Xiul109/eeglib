"This module define the functions for preprocessing the signal data"


from scipy.signal import butter, filtfilt

def bandPassFilter(data,sampleRate=None,highpass=None,lowpass=None, order=2):
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
    order: int, optional
        Butterworth

    Returns
    -------
    numpy.ndarray
        The signal filtered betwen the highpass and the lowpass
    """
    size=len(data)
    if not sampleRate:
        sampleRate=size

    #nyquist frequency
    nyq = 0.5*sampleRate

    if highpass:
        highpass=highpass/nyq

    if lowpass:
        lowpass=lowpass/nyq


    if lowpass and highpass:
        b,a = butter(order, [highpass, lowpass], btype="band")
    elif lowpass:
        b,a = butter(order, lowpass, btype="low")
    elif highpass:
        b,a = butter(order, highpass, btype="high")
    else:
        return data


    return filtfilt(b, a, data)
