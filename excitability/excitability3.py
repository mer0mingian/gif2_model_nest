# coding=utf-8
"""
the script is supposed to be configured with names of parameter scan results
and give a plot to compare firing rate responses in the relevant regime.
"""

import numpy as np
import matplotlib.pyplot as plt
import excitability1 as ex1
import excitability2 as ex2
from itertools import product

namelist = list()
namelist.append('fitted_excitability_722850.csv')
namelist.append('fitted_excitability_753857.csv')
namelist.append('fitted_excitability_746025.csv')
namelist.append('fitted_excitability_weighted_746025.csv')
namelist.append('fitted_excitability_weighted_722850.csv')
namelist.append('fitted_excitability_weighted_753857.csv')

colorlist = ['r', 'g', 'c', 'k', 'y', 'm']
linestyles = ['-', '--', '-.']
presentations = product(colorlist, linestyles)

for index, name in enumerate(namelist):
    if index == 0:
        fig = ex1.makeoneplotfromsolutions(name, color=colorlist[ index ], lif=True)
    else:
        fig = ex1.makeoneplotfromsolutions(name, color=colorlist[ index ], lif=False)
plt.ion()
plt.show()
