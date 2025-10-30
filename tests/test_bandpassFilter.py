import unittest

import numpy as np

from eeglib.preprocessing import bandPassFilter

def energy(signal):
    return sum(signal**2)

class TestBandpassFilter(unittest.TestCase):
    
    def setUp(self):
        self.sampleRate = 128
        self.duration = 10
        x = np.linspace(0, self.duration, self.sampleRate*self.duration)
        self.signal = 2*np.cos(10*2*np.pi*x) +\
                      np.cos(5*2*np.pi*x+np.pi) +\
                      0.7*np.cos(2*2*np.pi*x+np.pi/2)
        self.signalHigh = 2*np.cos(10*2*np.pi*x) +\
                          np.cos(5*2*np.pi*x+np.pi)
        self.signalLow = np.cos(5*2*np.pi*x+np.pi) +\
                         0.7*np.cos(2*2*np.pi*x+np.pi/2)
        self.signalBand = np.cos(5*2*np.pi*x+np.pi)
        
        self.highpass = 3.5
        self.lowpass = 7.5
    
    def test_lowpass(self):
        filtered = bandPassFilter(self.signal, sampleRate=self.sampleRate, 
                                  lowpass=self.lowpass)

        self.assertGreater(np.corrcoef(filtered, self.signalLow)[0,1], 0.9)
        self.assertAlmostEqual(energy(filtered)/energy(self.signalLow), 1, 
                               delta = 0.1)
        
    def test_highpass(self):
        filtered = bandPassFilter(self.signal, sampleRate=self.sampleRate, 
                                  highpass=self.highpass)

        self.assertGreater(np.corrcoef(filtered, self.signalHigh)[0,1], 0.9)
        self.assertAlmostEqual(energy(filtered)/energy(self.signalHigh), 1, 
                               delta = 0.1)
    
    def test_bandpass(self):
        filtered = bandPassFilter(self.signal, sampleRate=self.sampleRate, 
                                lowpass = self.lowpass, highpass=self.highpass)

        self.assertGreater(np.corrcoef(filtered, self.signalBand)[0,1], 0.9)
        self.assertAlmostEqual(energy(filtered)/energy(self.signalBand), 1, 
                               delta = 0.1)
    
    def test_nopass(self):
        filtered = bandPassFilter(self.signal)
        np.testing.assert_array_equal(self.signal, filtered)
    

if __name__ == "__main__":
    unittest.main()