import unittest

import numpy as np
import colorednoise


import eeglib.features as features

class TestPFD(unittest.TestCase):
    n_tests  = 100
    n_points = 1000
    
    def test_white_noise(self):
        results = []
        for _ in range(self.n_tests):
            samples = np.random.normal(0, 1, self.n_points)
            results.append(features.PFD(samples))
        
        results_mean = np.mean(results)
        self.assertAlmostEqual(results_mean, 1.0355, delta=0.001)

        
    def test_brownian_noise(self):
        results = []
        for _ in range(self.n_tests):
            samples = np.cumsum(np.random.normal(0, 1, self.n_points))
            results.append(features.PFD(samples))
        
        results_mean = np.mean(results)
        self.assertAlmostEqual(results_mean, 1.027, delta=0.001)
        
    def test_pink_noise(self):
        results = []
        for _ in range(self.n_tests):
            samples = colorednoise.powerlaw_psd_gaussian(1,self.n_points)
            results.append(features.PFD(samples))
        
        results_mean = np.mean(results)
        self.assertAlmostEqual(results_mean, 1.031, delta=0.001)    

    

if __name__ == "__main__":
    unittest.main()
