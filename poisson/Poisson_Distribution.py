
# coding: utf-8

# # Probability Distributions
# ## Poisson Distribution
# ### Nikolas Schnellbächer (last revision 2018-10-19)

# In[59]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import poisson


# The Poisson distribution is a very common probability distribution to describe stochastic processes. Hence it is often used as a noise model for physical processes.
# We call a discrete random variable $K$ Poisson distributed, if its probability mass function (PMF) is given by
# \begin{align}
# p(K=k) = \dfrac{e^{-\mu} \mu^k}{k!} \, ,
# \end{align}
# where $\mu$ is the only shape parameter of this distribution. The random variable $K$ can take any discrete integer value $k\ge 0$.

# Using a python environment we use scipy's built in functionality to work with Poisson distributions.
# For this purpose use the 
# ```python
# from scipy.stats import poisson
# ```
# statement. Then you can access the probability mass function in the following way
# ```python
# poisson.pmf(k, mu, loc)
# ```
# where $k$ is the discrete random variable, $\mu$ the shape parameter and `loc` the location parameter of the Poisson distribution.

# In[60]:


def plot_pmfs(X, muVals, labels):
    """
    plot Poisson probability mass functions
    """
    
    pColors = ['#CCCCCC', 'C0', 'C1', 'C2']
    
    f, ax = plt.subplots(1)
    f.set_size_inches(5.5, 3.5)
    
    ax.set_xlabel(r'$k$', fontsize = 18.0)
    ax.set_ylabel(r'$p(k\, ; \mu)$', fontsize = 18.0)
    ax.xaxis.labelpad = 4.0
    ax.yaxis.labelpad = 4.0 
    
    labelfontsize = 15.0
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(labelfontsize)
    
    lineWidth = 1.5    
    
    ax.plot([-5.0, 35.0], [0.0, 0.0], 
             color = pColors[0],
             alpha = 1.0,
             lw = lineWidth,
             zorder = 2,
             dashes = [4.0, 2.0])
    
    for i in range(len(muVals)):
    
        ax.plot(X[:, 0], X[:, i + 1], 
                 color = pColors[i + 1],
                 alpha = 1.0,
                 lw = lineWidth,
                 zorder = 2,
                 label = r'')
             
        ax.scatter(X[:, 0], X[:, i + 1],
                    s = 20.0,
                    lw = lineWidth,
                    facecolor = pColors[i + 1],
                    edgecolor = 'None',
                    zorder = 11,
                    label = labels[i])
        
    leg = ax.legend(handlelength = 0.25, 
                    scatterpoints = 1,
                    markerscale = 1.0,
                    ncol = 1,
                    fontsize = 14.0)
    leg.draw_frame(False)
    plt.gca().add_artist(leg)
        
    ax.set_xlim(-0.5, 19.0)

    return None


# In[61]:


get_ipython().run_cell_magic('time', '', '# create Poisson distribution\nmuVals = [1.0, 5.0, 9.0]   \nxVals = np.arange(0, 30, 1)\n\nX = np.zeros((len(xVals), len(muVals) + 1))\nX[:, 0] = xVals\n\nfor i, mu in enumerate(muVals):\n\n    yVals = poisson.pmf(xVals, mu)\n    assert xVals.shape == yVals.shape, "Error: Shape assertion failed."\n    \n    X[:, i + 1] = yVals\n    \nlabels = [r\'$\\mu = 1$\',\n          r\'$\\mu = 5$\',\n          r\'$\\mu = 9$\']\n\nplot_pmfs(X, muVals, labels)')

