import unittest

import numpy as np
import colorednoise

from itertools import product

import eeglib.features as features

#Supress warnings
import warnings
warnings.filterwarnings("ignore")

class TestDFA(unittest.TestCase):
    n_tests  = 100
    n_points = 1000
    
    def test_white_noise(self):
        results = []
        for _ in range(self.n_tests):
            samples = np.random.normal(0, 1, self.n_points)
            results.append(features.DFA(samples))
        
        results_mean = np.mean(results)
        self.assertAlmostEqual(results_mean, 0.5, delta=0.05)

        
    def test_brownian_noise(self):
        results = []
        for _ in range(self.n_tests):
            samples = np.cumsum(np.random.normal(0, 1, self.n_points))
            results.append(features.DFA(samples))
        
        results_mean = np.mean(results)
        self.assertAlmostEqual(results_mean, 1.5, delta=0.05)
        
    def test_pink_noise(self):
        results = []
        for _ in range(self.n_tests):
            samples = colorednoise.powerlaw_psd_gaussian(1,self.n_points)
            results.append(features.DFA(samples))
        
        results_mean = np.mean(results)
        self.assertAlmostEqual(results_mean, 1, delta=0.05)
        
    def test_strong_negative_correlation(self):
        samples = np.zeros(self.n_points)
        results = []
        for _ in range(self.n_tests):
            x = np.random.random()
            for i in range(self.n_points):
                x = -x + np.random.random()
                samples[i] = x
            results.append(features.DFA(samples)) 
        
        results_mean = np.mean(results)
        self.assertLessEqual(results_mean, 0.1)
        
    def test_parameters(self):
        samples = np.random.normal(0, 1, self.n_points)
        
        #Params to test
        fit_degrees = [1,2,4,8]
        min_window_sizes = [2,4,8,16]
        max_window_sizes = [self.n_points, self.n_points//2, self.n_points//4]
        fskip = [0.01, 0.1, 0.5, 1,2]
        max_n_windows_sizess = [None, 10, 30, 50]
        
        for parameters in product(fit_degrees, min_window_sizes,
                                  max_window_sizes, fskip, 
                                  max_n_windows_sizess):
            features.DFA(samples, *parameters)
    

if __name__ == "__main__":
    unittest.main()
