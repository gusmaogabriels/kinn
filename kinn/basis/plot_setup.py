# -*- coding: utf-8 -*-
"""Matplotlib and IPython display setup for notebooks.

Import this module explicitly in notebooks to get the plot styling:

    import kinn.basis.plot_setup

This file was extracted from kinn/basis/__init__.py so that importing
kinn.basis for library use does not pull matplotlib or set global rc params.
"""

from matplotlib import pyplot as plt
from matplotlib import animation, cm
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import HTML, display, Image

plt.style.use('seaborn-v0_8-white')

SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 13

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

left   = 0.05
right  = 0.925
bottom = 0.15
top    = 0.85
wspace = 0.225
hspace = 0.25
