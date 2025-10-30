import unittest
import os

import numpy as np

import eeglib.helpers as helpers

import datetime


class TestHelpers(unittest.TestCase):
    test_file_name = os.path.join(os.path.dirname(__file__), "fake_EEG_signal.csv")
    test_file_edf = os.path.join(os.path.dirname(__file__), "fake_EEG_signal.edf")
    test_file_noname = os.path.join(os.path.dirname(__file__),"fake_EEG_signal_no_names.csv")
    test_data = np.random.random((3,1024))
    
    def test_channels_names(self):
        h = helpers.CSVHelper(self.test_file_name, windowSize=64, 
                              names = ["F1", "F2", "F3", "F4", "F5"],
                              selectedSignals=["F1","F3"])
        
        self.assertListEqual(h.getNames(), ["F1","F3"])
        self.assertListEqual(h.getNames([1]), ["F3"])
        
        # Not sure if this should be allowed, but it will be keeped by the
        # moment.
        h.selectSignals([1,0,"F1"])
        self.assertListEqual(h.getNames(), ["F3", "F1", "F1"])
    
    def test_EDFHelper(self):
        h = helpers.EDFHelper(self.test_file_edf)
        self.assertEqual(128, h.sampleRate)
        self.assertListEqual(h.getNames(), ["C1", "C2", "C3", "C4", "C5"])
    
    def test_CSVHelperNames(self):
        h = helpers.CSVHelper(self.test_file_name, sampleRate=128)
        self.assertListEqual(h.getNames(), ["C1", "C2", "C3", "C4", "C5"])
        
        h = helpers.CSVHelper(self.test_file_noname, sampleRate=128)
        self.assertListEqual(h.getNames(), ["0", "1", "2", "3", "4"])
    
    def test_branches(self):
        helpers.CSVHelper(self.test_file_name, sampleRate=128, ICA=True, 
                          lowpass=24, highpass=2, normalize=True)
        h = helpers.Helper(self.test_data)
        len(h)
        
        
    
    def test_indexing_bad_index_type(self):
        h = helpers.Helper(self.test_data)
        with self.assertRaises(ValueError) as cm:
            h["1"]
        
        errorMsg = "Only slices can be used."
        self.assertEqual(str(cm.exception), errorMsg)
        
        with self.assertRaises(ValueError) as cm:
            h[0:3.4]
        
        errorMsg ="An index can only be a int, a str or a datetime.timedelta."
        self.assertEqual(str(cm.exception), errorMsg)
    
    def test_indexing_no_datetime(self):
        h = helpers.Helper(self.test_data, sampleRate=128, windowSize=128)
        t0 = datetime.timedelta(seconds=1)
        tf = datetime.timedelta(seconds=2)
        iterator = iter(h[t0:tf])
        eeg = next(iterator)
        np.testing.assert_array_equal(eeg.window.window, 
                                      self.test_data[:,128:256])
    
    def test_indexing_bad_strings(self):
        h = helpers.Helper(self.test_data)

        with self.assertRaises(ValueError) as cm:
            h["a":]
        
        errorMsg = "could not convert string to float: 'a'"
        self.assertTrue(str(cm.exception), errorMsg)
        
        with self.assertRaises(ValueError) as cm:
            h["0:0:0:10":]
        
        errorMsg= "Wrong value for the index"
        self.assertTrue(str(cm.exception), errorMsg)
        
    
    def test_indexing_negative(self):
        h = helpers.Helper(self.test_data, sampleRate=128, windowSize=128)
        iterator = iter(h[-128::])
        eeg = next(iterator)
        np.testing.assert_array_equal(eeg.window.window, 
                                      self.test_data[:,-128:])
        
        iterator = iter(h["-0:0:1"::])
        eeg = next(iterator)
        np.testing.assert_array_equal(eeg.window.window, 
                                      self.test_data[:,-128:])
        
    def test_move_window(self):
        h = helpers.Helper(self.test_data, sampleRate=128, windowSize=128)
        h.moveEEGWindow(256)
        self.assertEqual(h.windowPosition, 256)
        
        with self.assertRaises(ValueError) as cm:
            h.moveEEGWindow(1024)
        
        errorMsg = "The start point is too near to the end."
        self.assertEqual(str(cm.exception), errorMsg)
        
            
    

    
        
if __name__ == "__main__":
    unittest.main()