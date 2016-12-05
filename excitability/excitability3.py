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

def make_plots_from_file(filename, color='k', lif=True):
    """
    work in progress
    """
    datamat = np.loadtxt(filename)
    for i in np.arange(0, datamat.shape[ 0 ]):
        c = datamat[ i, : ]
        results = ex1.run_specific_comparison(
                c[ 1 ], c[ 2 ], c[ 3 ],
                tau_1=c[ 4 ], V_dist=c[ 5 ], runs=30)
        add_label = ': C={0}, g={1}, g1={2}, t1={3}, dV={4}'.format(
                c[ 1 ], c[ 2 ], c[ 3 ], c[ 4 ], c[ 5 ])
        fig1 = ex1.plot_comparison_resultarray(results,
                color=color, limity=True, lif=lif, add_label=add_label)
    return fig1


namelist = list()
#namelist.append('fitted_excitability_722850.csv')
#namelist.append('fitted_excitability_753857.csv')
#namelist.append('fitted_excitability_746025.csv')
#namelist.append('fitted_excitability_weighted_746025.csv')
#namelist.append('fitted_excitability_weighted_722850.csv')
#namelist.append('fitted_excitability_weighted_753857.csv')
namelist.append('fitted_excitability_weighted_753857.csv')


colorlist = ['r', 'g', 'c', 'k', 'y', 'm']
linestyles = ['-', '--', '-.']
presentations = product(colorlist, linestyles)

for index, name in enumerate(namelist):
    if index == 0:
        fig = make_plots_from_file(name, color=colorlist[ index ], lif=True)
    else:
        fig = make_plots_from_file(name, color=colorlist[ index ], lif=False)
plt.ion()
plt.show()
