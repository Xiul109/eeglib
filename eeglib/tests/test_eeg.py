import unittest

import numpy as np
import pandas as pd

import eeglib.eeg as eeg



class TestEEG(unittest.TestCase):
    windowSize = 256
    channels = 4
    sampleRate = 128
    signal_data = np.random.random((windowSize,channels))
    names = ["C1", "C2", "C3", "C4"]
    
    def setUp(self):
        self.eeg = eeg.EEG(self.windowSize, self.sampleRate, self.channels,
                           names = self.names)
        self.eeg.set(self.signal_data)
        
        self.eegNoNames = eeg.EEG(self.windowSize, self.sampleRate,
                                  self.channels)
        self.eegNoNames.outputMode = "dict"
        self.eegNoNames.set(self.signal_data)
    
    def test_eeg_constructor(self):
        eegNoNames = eeg.EEG(self.windowSize, self.sampleRate, self.channels)
        self.assertEqual(eegNoNames.window._names, None)
        eegNames = eeg.EEG(self.windowSize, self.sampleRate, self.channels,
                           names = self.names)
        self.assertEqual(eegNames.window._names, self.names)
        
    def test_eeg_set(self):
        self.eeg.set(self.signal_data)
        d1 = self.eeg.window.window
        self.eeg.set(self.signal_data.T, columnMode=True)
        d2 = self.eeg.window.window
        np.testing.assert_array_equal(d1, d2)
        
        with self.assertRaises(ValueError) as cm:
            self.eeg.set(self.signal_data.T)
        self.assertIn("The number of samples must be equal to the",
                      str(cm.exception))
        
        with self.assertRaises(ValueError) as cm:
            self.eeg.set(42)
        self.assertIn("Samples must be a subscriptable object wich contains",
                      str(cm.exception))
    
    def test_applyFunction(self):
        v1 = self.eeg._applyFunctionTo(np.mean,0)
        self.assertEqual(type(v1), np.float64)
        v1_n = self.eeg._applyFunctionTo(np.mean,"C4")
        self.assertEqual(type(v1_n), np.float64)
        v2 = self.eeg._applyFunctionTo(np.mean,[1,2])
        self.assertEqual(len(v2), 2)
        v3 = self.eeg._applyFunctionTo(np.mean)
        self.assertEqual(len(v3), 4)
        v4 = self.eeg._applyFunctionTo(np.mean, slice(0,3))
        self.assertEqual(len(v4), 3)
        
        with self.assertRaises(ValueError):
            self.eeg._applyFunctionTo(np.mean, 3.4)
            
        v5 = self.eegNoNames._applyFunctionTo(np.mean, slice(0,3))
        self.assertEqual(list(v5.keys()), [0, 1, 2])
                   
        with self.assertRaises(ValueError):
            self.eegNoNames._applyFunctionTo(np.mean, "C1")
    
    def test_applyFunction2CGeneral(self):
        f = lambda x,y : np.corrcoef(x,y)[0,1]
        
        v1 = self.eeg._applyFunctionTo2C(f, (2,3))
        self.assertEqual(type(v1), np.float64)
        v2 = self.eeg._applyFunctionTo2C(f, [0,1,2])
        self.assertEqual(len(v2), 3)
        v3 = self.eeg._applyFunctionTo2C(f)
        self.assertEqual(len(v3), 6)
        v4 = self.eeg._applyFunctionTo2C(f, allPermutations=True)
        self.assertEqual(len(v4), 12)
        v5 = self.eeg._applyFunctionTo2C(f, [(0,2), (1,3)])
        self.assertEqual(len(v5), 2)
        v6 = self.eeg._applyFunctionTo2C(f, slice(0,3))
        self.assertEqual(len(v6), 3)
    
    def test_applyFunction2CDict(self):
        self.eeg.outputMode = "dict"      
        f = lambda x,y : np.corrcoef(x,y)[0,1]
        
        v1 = self.eeg._applyFunctionTo2C(f, (2,3))
        self.assertEqual(type(v1), np.float64)
        v2 = self.eeg._applyFunctionTo2C(f, [0,1,2])
        self.assertEqual(list(v2.keys()),
                         [("C1", "C2"), ("C1", "C3"), ("C2", "C3")])
        v3 = self.eeg._applyFunctionTo2C(f, [(0,2), (1,3)])
        self.assertEqual(list(v3.keys()), [("C1", "C3"), ("C2", "C4")])
        v4 = self.eeg._applyFunctionTo2C(f, slice(0,3))
        self.assertEqual(list(v4.keys()), 
                         [("C1", "C2"), ("C1", "C3"), ("C2", "C3")])
        
        self.eeg.outputMode = "list"
        
        
        vNM = self.eegNoNames._applyFunctionTo2C(f)
        self.assertEqual(list(vNM.keys()), [(0, 1), (0, 2), (0, 3),
                                           (1, 2), (1, 3), (2, 3)])
        
    def test_applyFunction2CError(self):
        f = lambda x,y : np.corrcoef(x,y)[0,1]
        
        with self.assertRaises(ValueError):
            self.eeg._applyFunctionTo2C(f, (2, 3, 4))
        
        with self.assertRaises(ValueError):
            self.eeg._applyFunctionTo2C(f, [(0, 1, 2), (2, 3, 4)])
        
        with self.assertRaises(ValueError):
            self.eeg._applyFunctionTo2C(f, [0, 1, (2,3)])
            
        with self.assertRaises(ValueError):
            self.eeg._applyFunctionTo2C(f, 0.1)
    
    def test_eeg_features(self):
        self.eeg.getChannel()
        self.eeg.DFA()
        self.eeg.CCC()
        self.eeg.LZC()
        self.eeg.sampEn()
        self.eeg.engagementLevel()
        self.eeg.synchronizationLikelihood()
        self.eeg.hjorthComplexity()
        self.eeg.hjorthMobility()
        self.eeg.hjorthActivity()
        self.eeg.HFD()
        self.eeg.PFD()
        self.eeg.getSignalAtBands()
        
        
    def test_DFT(self):
        self.eeg.DFT(output="complex")
        self.eeg.DFT(output="pHase")
        self.eeg.DFT(output="MAGNITUDE")
        
        with self.assertRaises(ValueError):
            self.eeg.DFT(output="incorrect")
            
        self.eeg.DFT(windowFunction="hamming")
        self.eeg.DFT(windowFunction=np.hamming(self.windowSize))
        
        with self.assertRaises(ValueError) as cm:
            self.eeg.DFT(windowFunction=np.hamming(self.windowSize+1))
        error = "The size of windowFunction is not the same as windowSize."
        self.assertEqual(str(cm.exception), error)
        
        with self.assertRaises(ValueError) as cm:
            self.eeg.DFT(windowFunction=23)
        error = "Not a valid type for windowFunction."
        self.assertEqual(str(cm.exception), error)
    
    def test_PSD(self):
        psd1 = self.eeg.PSD(0)
        self.assertEqual(type(psd1), np.ndarray)
        
        psd2 = self.eeg.PSD(0,retFrequencies=True)
        self.assertEqual(type(psd2), tuple)
    
    def test_DTW(self):
        dtw1 = self.eeg.DTW((0,1))
        dtw2 = self.eeg.DTW((0,1), normalize=True)
        self.assertEqual(dtw1,dtw2*self.windowSize)
        
        dtw3 = self.eeg.DTW(returnOnlyDistances=False)
        dtw4 = self.eeg.DTW(returnOnlyDistances=False, normalize=True)
        self.assertEqual(type(dtw3), type(dtw4))
        self.assertEqual(type(dtw3), list)
        
        self.assertEqual(type(dtw3[0]), type(dtw4[0]))
        self.assertEqual(type(dtw3[0]), tuple)
    
    def test_bandPower(self):
        bp = self.eeg.bandPower()
        self.assertEqual(type(bp), list)
        self.assertEqual(type(bp[0]), dict)
        
        self.assertEqual(type(self.eeg.bandPower(0)), dict)
        
        self.eeg.outputMode = "dict"
        self.assertEqual(type(self.eeg.bandPower(["C1","C2"])), dict)
        self.eeg.outputMode = "list"
        
        self.eeg.bandPower(0, spectrumFrom="DFT")
        self.eeg.bandPower(0, spectrumFrom="PSD")
        
        with self.assertRaises(ValueError) as _:
            self.eeg.bandPower(spectrumFrom = "Incorrect")

    def test_window(self):
        v = self.eeg.window[0]
        self.assertEqual(v.shape, (4,))
    
        
if __name__ == "__main__":
    unittest.main()