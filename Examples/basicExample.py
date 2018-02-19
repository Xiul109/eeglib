from eeglib.helpers import CSVHelper

helper= CSVHelper("fake_EEG_signal.csv")

for eeg in helper:
    print(eeg.PFD())