import unittest

import numpy as np
import colorednoise

from itertools import product

import eeglib.features as features

class TestLZC(unittest.TestCase):
    n_tests  = 50
    n_points = 1000
    
    def test_white_noise(self):
        results = []
        for _ in range(self.n_tests):
            samples = np.random.normal(0, 1, self.n_points)
            results.append(features.LZC(samples))
        
        results_mean = np.mean(results)
        self.assertAlmostEqual(results_mean, 1.05, delta=0.05)

        
    def test_brownian_noise(self):
        results = []
        for _ in range(self.n_tests):
            samples = np.cumsum(np.random.normal(0, 1, self.n_points))
            results.append(features.LZC(samples))
        
        results_mean = np.mean(results)
        self.assertAlmostEqual(results_mean, 0.15, delta=0.05)
        
    def test_pink_noise(self):
        results = []
        for _ in range(self.n_tests):
            samples = colorednoise.powerlaw_psd_gaussian(1,self.n_points)
            results.append(features.LZC(samples))
        
        results_mean = np.mean(results)
        self.assertAlmostEqual(results_mean, 0.75, delta=0.05)


if __name__ == "__main__":
    unittest.main()
