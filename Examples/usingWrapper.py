from eeglib import wrapper, helpers

helper = helpers.CSVHelper("fake_EEG_signal.csv", windowSize=128)

wrap = wrapper.Wrapper(helper)

wrap.addFeature.HFD()
wrap.addFeature.DFT()
wrap.addFeature.synchronizationLikelihood()

features=wrap.getAllFeatures()