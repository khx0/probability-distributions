#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################################
# author: Nikolas Schnellbaecher
# contact: khx0@posteo.net
# date: 2019-05-18
# file: univariate_poisson_sampling.py
# tested with python 2.7.15 in conjunction with mpl version 2.2.3
##########################################################################################

import os
import platform
import datetime
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import legend

import scipy
from scipy.stats import poisson

now = datetime.datetime.now()
now = "{}-{}-{}".format(str(now.year), str(now.month).zfill(2), str(now.day).zfill(2))

BASEDIR = os.path.dirname(os.path.abspath(__file__))
RAWDIR = os.path.join(BASEDIR, 'raw')
OUTDIR = os.path.join(BASEDIR, 'out')

os.makedirs(OUTDIR, exist_ok = True)

def getFigureProps(width, height, lFrac = 0.17, rFrac = 0.9, bFrac = 0.17, tFrac = 0.9):
    '''
    True size scaling auxiliary function to setup mpl plots with a desired size.
    Specify widht and height in cm.
    lFrac = left fraction   in [0, 1]
    rFrac = right fraction  in [0, 1]
    bFrac = bottom fraction in [0, 1]
    tFrac = top fraction    in [0, 1]
    returns:
        fWidth = figure width
        fHeight = figure height
    These figure width and height values can then be used to create a figure instance
    of the desired size, such that the actual plotting canvas has the specified
    target width and height, as provided by the input parameters of this function.
    '''
    axesWidth = width / 2.54    # convert to inches
    axesHeight = height / 2.54  # convert to inches
    fWidth = axesWidth / (rFrac - lFrac)
    fHeight = axesHeight / (tFrac - bFrac)
    return fWidth, fHeight, lFrac, rFrac, bFrac, tFrac

def Plot(titlestr, X, S, params, outname, outdir, pColors,
         grid = False, drawLegend = True, xFormat = None, yFormat = None,
         savePDF = True, savePNG = False, datestamp = True):

    mpl.rcParams['xtick.top'] = False
    mpl.rcParams['xtick.bottom'] = True
    mpl.rcParams['ytick.right'] = False
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'

    mpl.rc('font', **{'size': 10})
    mpl.rc('legend', **{'fontsize': 6.0})
    mpl.rc("axes", linewidth = 0.5)

    # mpl.rc('font', **{'family' : 'sans-serif', 'sans-serif' : ['Myriad Pro']})
    mpl.rc('font', **{'family' : 'sans-serif', 'sans-serif' : ['Helvetica']})
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['mathtext.fontset'] = 'cm'
    fontparams = {'text.latex.preamble': [r'\usepackage{cmbright}',
                                          r'\usepackage{amsmath}']}
    mpl.rcParams.update(fontparams)

    ######################################################################################
    # set up figure
    fWidth, fHeight, lFrac, rFrac, bFrac, tFrac =\
        getFigureProps(width = 4.1, height = 2.9,
                       lFrac = 0.17, rFrac = 0.96, bFrac = 0.17, tFrac = 0.93)
    f, ax1 = plt.subplots(1)
    f.set_size_inches(fWidth, fHeight)
    f.subplots_adjust(left = lFrac, right = rFrac)
    f.subplots_adjust(bottom = bFrac, top = tFrac)
    ######################################################################################
    labelfontsize = 5.0

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)

    ax1.tick_params('both', length = 2.5, width = 0.5, which = 'major', pad = 3.0)
    ax1.tick_params('both', length = 1.5, width = 0.25, which = 'minor', pad = 3.0)

    ax1.tick_params(axis = 'x', which = 'major', pad = 1.5)
    ax1.tick_params(axis = 'y', which = 'major', pad = 1.5, zorder = 10)
    ######################################################################################
    # labeling
    plt.title(titlestr)
    ax1.set_xlabel(r'$k$', fontsize = 6.0)
    ax1.set_ylabel(r'$p(k\, |\, \lambda)$', fontsize = 6.0)
    ax1.xaxis.labelpad = 2.0
    ax1.yaxis.labelpad = 2.0
    ######################################################################################
    # plotting

    lineWidth = 0.5

    ax1.plot([-1.0, 21.0], [0.0, 0.0],
             color = pColors[0],
             alpha = 1.0,
             lw = lineWidth,
             zorder = 2,
             dashes = [4.0, 2.0])

    for i in range(len(muValues)):

        ax1.plot(X[:, 0], X[:, i + 1],
                 color =  pColors[i + 1],
                 alpha = 1.0,
                 lw = lineWidth,
                 zorder = 2,
                 label = r'')
                 #drawstyle = 'steps-mid')
                    
        ax1.scatter(S[:, 0], S[:, i + 1],
                    s = 6.0,
                    marker = 'o',
                    lw = lineWidth,
                    facecolor = 'None',
                    edgecolor = pColors[i + 1],
                    zorder = 12,
                    label = labels[i])

    p1, = ax1.plot([-99.9, -99.9], [-99.9, -99.9],
                  color = pColors[0],
                  alpha = 1.0,
                  lw = lineWidth,
                  zorder = 2)
    p1 = [p1]

    ######################################################################################
    # annotations

    label = r'$p(k\, |\, \lambda) = \dfrac{\lambda^k}{k!} e^{-\lambda}$'

    x_pos = 0.25

    ax1.annotate(label,
                 xy = (x_pos, 0.85),
                 xycoords = 'axes fraction',
                 fontsize = 6.0,
                 horizontalalignment = 'left')

    ax1.annotate(r'# samples {:d}'.format(nSamples),
                 xy = (0.0, 1.025),
                 xycoords = 'axes fraction',
                 fontsize = 4.0,
                 horizontalalignment = 'left')

    ######################################################################################
    # legend
    if drawLegend:
        leg = ax1.legend(# bbox_to_anchor = [0.7, 0.8],
                         # loc = 'upper left',
                         handlelength = 0.25,
                         scatterpoints = 1,
                         markerscale = 1.0,
                         ncol = 1)
        leg.draw_frame(False)
        plt.gca().add_artist(leg)

    label1 = [r'analytical']

    leg1 = plt.legend(p1, label1,
                      bbox_to_anchor = [0.68, 0.65],
                      loc = 'upper left',
                      handlelength = 1.75,
                      fontsize = 4.0)

    for legobj in leg1.legendHandles:
        legobj.set_linewidth(1.0)

    leg1.draw_frame(False)
    plt.gca().add_artist(leg1)

    ######################################################################################
    # set plot range
    if (xFormat == None):
        pass
    else:
        major_x_ticks = np.arange(xFormat[2], xFormat[3], xFormat[4])
        minor_x_ticks = np.arange(xFormat[2], xFormat[3], xFormat[5])
        ax1.set_xticks(major_x_ticks)
        ax1.set_xticks(minor_x_ticks, minor = True)
        ax1.set_xlim(xFormat[0], xFormat[1])

    if (yFormat == None):
        pass
    else:
        major_y_ticks = np.arange(yFormat[2], yFormat[3], yFormat[4])
        minor_y_ticks = np.arange(yFormat[2], yFormat[3], yFormat[5])
        ax1.set_yticks(major_y_ticks)
        ax1.set_yticks(minor_y_ticks, minor = True)
        ax1.set_ylim(yFormat[0], yFormat[1])

    ax1.set_axisbelow(False)

    for spine in ax1.spines.values(): # ax1.spines is a dictionary
        spine.set_zorder(10)

    ######################################################################################
    # grid options
    if grid:
        ax1.grid(color = 'gray', linestyle = '-', alpha = 0.2, which = 'major',
                 linewidth = 0.2)
        ax1.grid('on')
        ax1.grid(color = 'gray', linestyle = '-', alpha = 0.05, which = 'minor',
                 linewidth = 0.1)
        ax1.grid('on', which = 'minor')
    ######################################################################################
    # save to file
    if datestamp:
        outname += '_' + now
    if savePDF:
        f.savefig(os.path.join(outdir, outname) + '.pdf', dpi = 300, transparent = True)
    if savePNG:
        f.savefig(os.path.join(outdir, outname) + '.png', dpi = 600, transparent = False)
    ######################################################################################
    # close handles
    plt.cla()
    plt.clf()
    plt.close()
    return outname

def sample(mu):
    u = np.random.uniform()    
    x = poisson.ppf(u, mu)
    return x

if __name__ == '__main__':

    print("using np.__version__ =", np.__version__)
    print("using scipy.__version__ =", scipy.__version__)

    nSamples = 5000 # 100 # 2000 # 1000 # 50000

    # create Poisson distributed random sample
    muValues = [1.0, 5.0, 9.0]

    nBins = 20
    S = np.zeros((nBins, len(muValues) + 1))
    xVals = np.arange(0, 30, 1)
    X = np.zeros((len(xVals), len(muValues) + 1))
    X[:, 0] = xVals

    for i, mu in enumerate(muValues):

        samples = []
        for cc in range(nSamples):
            samples.append(sample(mu))

        # bin the data
        hist, bin_edges = np.histogram(samples, bins = nBins, range = (-0.5, 19.5),
                                       density = True)
        
        bin_centers = (bin_edges[1:] + bin_edges[0:-1]) / 2.0
        assert hist.shape == bin_centers.shape, "Error: Shape assertion failed."
            
        S[:, 0] = bin_centers
        S[:, i + 1] = hist
    
        yVals = poisson.pmf(xVals, mu)
        assert xVals.shape == yVals.shape, "Error: Shape assertion failed."

        X[:, i + 1] = yVals

        # check for normalization
        norm = np.sum(yVals)
        print("Normalization = np.sum(yVals) = ", norm)
        assert np.isclose(norm, 1.0), \
               "Error: Poisson distribution seems NOT to be normalized."
    
    # call plotting function
    
    labels = [r'$\lambda = 1$',
              r'$\lambda = 5$',
              r'$\lambda = 9$']

    outname = 'mpl_discrete_poisson_pmf_A_nSamples_{:d}'.format(nSamples)
    outname += '_Python_' + platform.python_version() + \
               '_mpl_' + mpl.__version__

    xFormat = (-0.5, 19.5, 0.0, 19.1, 5.0, 1.0)
    yFormat = (-0.0175, 0.42, 0.0, 0.405, 0.1, 0.05)

    pColors = ['#CCCCCC', 'C0', 'C1', 'C2']

    outname = Plot(titlestr = '',
                   X = X,
                   S = S,
                   params = [],
                   outname = outname,
                   outdir = OUTDIR,
                   pColors = pColors,
                   grid = False,
                   drawLegend = True,
                   xFormat = xFormat,
                   yFormat = yFormat)
