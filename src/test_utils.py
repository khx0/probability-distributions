#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################################
# author: Nikolas Schnellbaecher
# contact: khx0@posteo.net
# date: 2018-10-15
# file: test_utils.py
# tested with python 2.7.15
# tested with python 3.7.0
##########################################################################################

import sys
import os
import numpy as np
import unittest

from scipy.stats import poisson

from utils import checkPMFnorm

class utils_test(unittest.TestCase):
    
    """
    Test cases for the utils module.
    """
            
    def test_pmf_norm_01(self):
        
        xVals = np.arange(0, 30, 1)
        
        mu = 0.0
        yVals = poisson.pmf(xVals, mu)
        assert xVals.shape == yVals.shape, "Error: Shape assertion failed."
        norm = checkPMFnorm(yVals, 'Poisson dist with mu = %.2f' %(mu))
        self.assertTrue(np.isclose(norm, 1.0))
        
        mu = 1.0
        yVals = poisson.pmf(xVals, mu)
        assert xVals.shape == yVals.shape, "Error: Shape assertion failed."
        norm = checkPMFnorm(yVals, 'Poisson dist with mu = %.2f' %(mu))
        self.assertTrue(np.isclose(norm, 1.0))
    
        xVals = np.arange(0, 50, 1)
    
        mu = 5.0
        yVals = poisson.pmf(xVals, mu)
        assert xVals.shape == yVals.shape, "Error: Shape assertion failed."
        norm = checkPMFnorm(yVals, 'Poisson dist with mu = %.2f' %(mu))
        self.assertTrue(np.isclose(norm, 1.0))
        
        mu = 9.0
        yVals = poisson.pmf(xVals, mu)
        assert xVals.shape == yVals.shape, "Error: Shape assertion failed."
        norm = checkPMFnorm(yVals, 'Poisson dist with mu = %.2f' %(mu))
        self.assertTrue(np.isclose(norm, 1.0))
        
        return None
        
if __name__ == '__main__':
    
    unittest.main()
