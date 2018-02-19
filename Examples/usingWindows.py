from eeglib.helpers import CSVHelper

helper= CSVHelper("fake_EEG_signal.csv", windowSize=256)

for eeg in helper:
    print(eeg.PFD())