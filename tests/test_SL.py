import unittest

import numpy as np

import eeglib.features as features


def henon_system(size=1000, C=1, B=0.3):
    x = np.zeros(size)
    u = np.zeros(size)
    y = np.zeros(size)
    v = np.zeros(size)
    for t in range(0,size-1):
        x[t+1] = 1 - 1.4*x[t]**2 + 0.3*u[t]
        u[t+1] = x[t]
        y[t+1] = 1 - 1.4*(C*x[t]+(1-C)*y[t])*y[t]+B*v[t]
        v[t+1] = y[t]
    return x, u, y, v

class TestSL(unittest.TestCase):
    n_points = 1000
    n_Cs = 100
    
    error = 0.05
    
    def test_henon_non_identical(self):
        Cs = np.linspace(0, 1, num=self.n_Cs)
        values=[]
        for C in Cs:
            x, u, y, v = henon_system(size=self.n_points, C=C, B=0.1)
            values.append(features.synchronizationLikelihood(x,y,10,1,5,100))
        corr = np.corrcoef(Cs,values)[0,1]
        self.assertLessEqual(1-corr, self.error)
    
    def test_henon_identical(self):
        Cs = np.linspace(0, 1, num=self.n_Cs)
        values=[]
        for C in Cs:
            x, u, y, v = henon_system(size=self.n_points, C=C, B=0.3)
            values.append(features.synchronizationLikelihood(x,y,10,1,5,100))
        corr = np.corrcoef(Cs,values)[0,1]
        self.assertLessEqual(1-corr, self.error)

    def test_error_diff_lens(self):
        v1 = np.arange(0,100)
        v2 = np.arange(1,100)
        
        with self.assertRaises(ValueError) as cm:
            features.synchronizationLikelihood(v1, v2, 10, 1, 5, 100)

            

if __name__ == "__main__":
    unittest.main()