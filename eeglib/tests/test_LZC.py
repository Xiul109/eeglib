import unittest

import numpy as np
import colorednoise

import eeglib.features as features

class TestLZC(unittest.TestCase):
    n_tests  = 100
    n_points = 1000
    
    error = 0.06
    
    def test_white_noise(self):
        results = []
        for _ in range(self.n_tests):
            samples = np.random.normal(0, 1, self.n_points)
            results.append(features.LZC(samples))
        
        results_mean = np.mean(results)
        self.assertAlmostEqual(results_mean, 1.05, delta=self.error)

        
    def test_brownian_noise(self):
        results = []
        for _ in range(self.n_tests):
            samples = np.cumsum(np.random.normal(0, 1, self.n_points))
            results.append(features.LZC(samples))
        
        results_mean = np.mean(results)
        self.assertAlmostEqual(results_mean, 0.15, delta=self.error)
        
    def test_pink_noise(self):
        results = []
        for _ in range(self.n_tests):
            samples = colorednoise.powerlaw_psd_gaussian(1,self.n_points)
            results.append(features.LZC(samples))
        
        results_mean = np.mean(results)
        self.assertAlmostEqual(results_mean, 0.75, delta=self.error)
        
    def test_binarize(self):
        testData = [*range(10)]
        expected = [0,0,0,0,0,1,1,1,1,1]
        binar = features._binarize(testData, 4)
        np.testing.assert_array_equal(binar, expected)


if __name__ == "__main__":
    unittest.main()
