import unittest

import numpy as np

import eeglib.features as features

class TestHjorthParameters(unittest.TestCase):
    n_tests = 100
    n_points = 10000
    
    
    def test_activity(self):
        results = []
        for _ in range(self.n_tests):
            samples = np.random.normal(0, 1, self.n_points)
            results.append(features.hjorthActivity(samples))
        
        results_mean = np.mean(results)
        self.assertAlmostEqual(results_mean, 1, delta=0.005)
    
    def test_mobility(self):
        results = []
        for _ in range(self.n_tests):
            samples = np.random.normal(0, 1, self.n_points)
            results.append(features.hjorthMobility(samples))
        
        results_mean = np.mean(results)
        self.assertAlmostEqual(results_mean, 0.707, delta=0.005)
    
    def test_complexity(self):
        results = []
        for _ in range(self.n_tests):
            samples = np.random.normal(0, 1, self.n_points)
            results.append(features.hjorthComplexity(samples))
        
        results_mean = np.mean(results)
        self.assertAlmostEqual(results_mean, 1.224, delta=0.005)


if __name__ == "__main__":
    unittest.main()
