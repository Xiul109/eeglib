import unittest

import numpy as np

import eeglib.features as features

class TestBandpower(unittest.TestCase):
    n_points = 1000
    duration = 10
    freq_res = 1/duration
    sampleRate = n_points/duration
    
    error = 0.07
    
    def setUp(self):
        self.x = np.pi*np.linspace(0, self.duration, self.n_points)
        bands = [(2,4), (8,10), (12,14)]
        signal = np.zeros(self.n_points)
        
        np.random.seed(42)
        for band in bands:
            for f in np.arange(band[0], band[1], self.freq_res):
                signal+= np.cos(f*2*self.x+np.random.uniform(0, 2*np.pi)) \
                    * np.random.uniform(0.9,1.1)
        self.test_signal = {"data":signal, "bands":bands}
    
    def test_bandpower(self):
        spectrum =np.abs(np.fft.fft(
                            self.test_signal["data"])[0:self.n_points//2+1])
        
        dur = self.duration
        bands = self.test_signal["bands"]
        n_bands = len(bands)
        bandLimits = {"test%d"%i:(int(band[0]*dur)-2, int(band[1]*dur)+2) 
                      for i, band in enumerate(bands)}
        res = features.bandPower(spectrum, bandsLimits=bandLimits,
                                 freqRes=0.1, normalize=True)
        
        totalPower = 0
        for power in res.values():
            totalPower+=power
            self.assertAlmostEqual(power, 1/n_bands, delta=self.error/n_bands)
            self.assertLess(power, 1/n_bands)
        
        self.assertAlmostEqual(totalPower, 1, delta=self.error)
        self.assertLess(totalPower, 1)


if __name__ == "__main__":
    unittest.main()
