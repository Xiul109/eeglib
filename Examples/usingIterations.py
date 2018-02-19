from eeglib.helpers import CSVHelper

helper= CSVHelper("fake_EEG_signal.csv", windowSize=256)

for eeg in helper[::128]:
    print(eeg.PFD())