#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################################
# author: Nikolas Schnellbaecher
# contact: khx0@posteo.net
# date: 2018-10-15
# file: utils.py
# tested with python 2.7.15
# tested with python 3.7.0
##########################################################################################

import math
import numpy as np

def checkPMFnorm(PMF, textString = '', verbose = False):
    '''
    Checks for proper normalization of a given probability mass function (PMF).
    '''
    norm = np.sum(PMF)
    print("norm(" + textString + ") =", norm)
    if (not np.isclose(norm, 1.0)):
        print(textString + " distribution seems NOT to be normalized to 1.")
        if (verbose):
            sys.exit(1)
    return norm

def checkPDFnorm(PDF, xGrid, textString = '', verbose = False):
    '''
    Checks for proper normalization of a given probability density function (PDF).
    '''
    norm = np.trapz(PDF, xGrid)
    print("norm(" + textString + ") =", norm)
    if (not np.isclose(norm, 1.0)):
        print(textString + " distribution seems NOT to be normalized to 1.")
        if (verbose):
            sys.exit(1)
    return norm

if __name__ == '__main__':

    pass
