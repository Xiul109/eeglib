import unittest

import numpy as np
import pandas as pd

import eeglib.wrapper as wrapper
import eeglib.helpers as helpers


class TestWrapper(unittest.TestCase):
    test_file_name = "fake_EEG_signal.csv"
    
    def test_wrapper_add_features_errors(self):
        helper = helpers.CSVHelper(self.test_file_name, sampleRate=128)
        wrap = wrapper.Wrapper(helper[::128])
        
        with self.assertRaises(ValueError) as _:
            wrap.addFeatures([("synchronizationLikelihood",[(0,3)]),])
        
        with self.assertRaises(AttributeError) as _:
            wrap.addFeatures("PFD")
    
    def test_wrapper_all_features_stored(self):
        helper = helpers.CSVHelper(self.test_file_name, sampleRate=128)
        wrap = wrapper.Wrapper(helper[::128])
        
        wrap.addFeature("bandPower", 0)
        wrap.addFeature.DFA([1,2])
        wrap.addFeature.HFD(kMax=8)
        wrap.addFeatures([("synchronizationLikelihood",[(0,3)],{}),
                          "PFD"])
        wrap.addCustomFeature(np.mean)
        wrap.addCustomFeature(lambda x, y: np.linalg.norm(x-y), 
                              channels=[0,1,2], twoChannels=True)
        
        features = wrap.getAllFeatures()
        stored = wrap.getStoredFeatures()
        
        pd.testing.assert_frame_equal(features, stored)
        pd.testing.assert_series_equal(features.iloc[-1], wrap.getFeatures(),
                                       check_names=False)
        
        
    def test_wrapper_segmentation(self):
        helper = helpers.CSVHelper(self.test_file_name, sampleRate=128)
        wrap = wrapper.Wrapper(helper[::128], label="test",
                               segmentation=[(("0","2"),"l1"), 
                                             ((384,512),"l2")])
        
        wrap.addFeature("bandPower", 0)
        
        label = wrap.getAllFeatures()["segment_label"]
        
        self.assertEqual(label[0], "l1")
        self.assertEqual(label[1], "l1")
        self.assertEqual(label[3], "l2")
    
    def test_wrapper_names_base(self):
        helper = helpers.CSVHelper(self.test_file_name, sampleRate=128)
        wrap = wrapper.Wrapper(helper)
        wrap.addFeature("bandPower", 0, bandLimits={"band1":(3,4),
                                                    "band2":(10,20)})
        
        featureName="bandPower(0,){'bandLimits': {'band1': (3, 4), 'band2': (10, 20)}}"
        self.assertEqual(list(wrap.featuresNames())[0], featureName)

    def test_wrapper_names_noargs(self):
        helper = helpers.CSVHelper(self.test_file_name, sampleRate=128)
        wrap = wrapper.Wrapper(helper)
        wrap.addFeature("bandPower", 0, bandLimits={"band1":(3,4),
                                                    "band2":(10,20)},
                        hideArgs=True)
        
        featureName="bandPower"
        self.assertEqual(list(wrap.featuresNames())[0], featureName)
        
if __name__ == "__main__":
    unittest.main()