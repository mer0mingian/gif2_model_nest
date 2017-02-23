# coding=utf-8
"""
Collection of tools pasted together from various locations to analyse the Brunel network
"""
import os
import copy
import numpy as np
import time
import sys
import shutil
import contextlib
import nest
# import matplotlib as mpl
if ('blaustein' in os.environ['HOSTNAME'].split('.')[ 0 ]):
        cluster = True
        # mpl.use('Agg')
        print('recognised cluster')
else: 
    cluster = False
    print('working locally')
# import matplotlib.pyplot as plt
try:
    from elephant.statistics import isi, cv
except ImportError:
    print('Warning: elephant functions not available!')
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
# import correlation_toolbox.correlation_analysis as corr
# import correlation_toolbox.helper as ch
from mingtools1 import *


# ****************************************************************************************************************************
# BRUNEL NETWORK HELPER FUNCTIONS
# ****************************************************************************************************************************

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)

def define_synweights(Cm, tau_m, tauSyn, J=0.15, g=4.0):
    J_ex = J  # amplitude of excitatory postsynaptic potential
    J_in = -g * J
    # Compute the synaptic weights
    PSP_e = J
    PSP_ext = J
    re = tau_m / tauSyn
    de = tauSyn - tau_m
    PSC_e_over_PSP_e = 1 / (
        1 / Cm * tau_m * tauSyn / de * (np.power(re, tau_m / de) - 
                                         np.power(re, tauSyn / de)))
    PSC_i_over_PSP_i = PSC_e_over_PSP_e
    PSC_e = PSC_e_over_PSP_e * PSP_e
    PSP_i = - PSP_e * g
    PSC_i = PSC_i_over_PSP_i * PSP_i
    PSC_ext = PSC_e_over_PSP_e * PSP_ext
    #  standard deviation of PSC amplitudes relative to mean PSC amplitudes:
    PSC_rel_sd = 0.1
    # this are the 87.8 pA:
    weight_dict = {'E': {'distribution': 'normal_clipped', 'low': 0.0, 
                         'mu': PSC_e, 'sigma': PSC_e * PSC_rel_sd},
                   'R': {'distribution': 'normal_clipped', 'low': 0.0,
                         'mu': PSC_e, 'sigma': PSC_e * PSC_rel_sd},
                   'I': {'distribution': 'normal_clipped', 'high': 0.0,
                         'mu': PSC_i, 'sigma': np.abs(PSC_i * PSC_rel_sd)},
                   'ext': {'distribution': 'normal_clipped', 'low': 0.0, 
                           'mu': PSC_ext, 'sigma': PSC_ext * PSC_rel_sd}
                  }
    return weight_dict


def define_structures(NE, NI, NR):
    """define all the structures for the MC connectivity"""
    """ Previously, we assumed a microcircuit connectivity of just on layer.
    Now we extract L5 from the full microcircuit instead."""

    #     2/3e      2/3i    4e      4i      5e      5i      6e      6i
    conn_probs_old = [ 
        [ 0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.,     0.0076, 0. ],
        [ 0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.,     0.0042, 0. ],
        [ 0.0077, 0.0059, 0.0497, 0.135,  0.0067, 0.0003, 0.0453, 0. ],
        [ 0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.,     0.1057, 0. ],
        [ 0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0. ],
        [ 0.0548, 0.0269, 0.0257, 0.0022, 0.06,   0.3158, 0.0086, 0. ],
        [ 0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252 ],
        [ 0.0364, 0.001,  0.0034, 0.0005, 0.0277, 0.008,  0.0658, 0.1443 ] ]

    layers = {'L23': 0, 'L4': 1, 'L5': 2, 'L6': 3}
    pops_old = {'E': 0, 'I': 1}
    pops_new = {'E': 0, 'I': 1, 'R': 2}

    structure_new = {'L23': {'E': 0, 'I': 1, 'R': 2},
                     'L4':  {'E': 3, 'I': 4, 'R': 5},
                     'L5':  {'E': 6, 'I': 7, 'R': 8},
                     'L6':  {'E': 9, 'I': 10, 'R': 11}}

    structure_old = {'L23': {'E': 0, 'I': 1},
                     'L4':  {'E': 2, 'I': 3},
                     'L5':  {'E': 4, 'I': 5},
                     'L6':  {'E': 6, 'I': 7}}


    # Numbers of neurons in full-scale model
    neuron_nums_new = {
        'L23': {'E': 20683, 'I': 5834, 'R': 0},
        'L4':  {'E': 21915, 'I': 5479, 'R': 0},
        'L5':  {'E': NE,    'I': NI,   'R': NR},
        'L6':  {'E': 14395, 'I': 2948, 'R': 0}
    }

    neuron_nums_old = {
        'L23': {'E': 20683, 'I': 5834},
        'L4':  {'E': 21915, 'I': 5479},
        'L5':  {'E': 4850, 'I': 1065},
        'L6':  {'E': 14395, 'I': 2948}
    }

    # the rates extracted from the microcircuit
    rates = {
             'L23': {'E':0.971, 'I':2.868},
             'L4' : {'E':4.746, 'I':5.396},
             # 'L5' : {'E':8.142, 'I':9.078},
             'L6' : {'E':0.991, 'I':7.523}
            }

    return conn_probs_old, layers, pops_old, pops_new, structure_new, structure_old, neuron_nums_new, neuron_nums_old, rates


def verbosity(input, verbosity=False, designation=None):
    """function for giving verbose ouput in run_brunel"""
    if verbosity:
        if designation:
            # with printoptions(precision=3, suppress=True):
            print('{0} {1}'.format(designation, input))
        else:
            pass
#            print(input)

def connect(source, target, syn_spec, conn_spec=None):
    """nest.Connect wrapper to catch 0-pops"""
    if (len(source) == 0 or len(target) == 0):
        pass
    else:
        nest.Connect(source, target, 
                     syn_spec=syn_spec,
                     conn_spec=conn_spec)

def create(neuron_model, number=1):
    """nest.Connect wrapper to catch 0-pops"""
    if number == 0:
        return ()
    else:
        return nest.Create(neuron_model, number)


def writeout(simulation_index, resultarray, path=None):
    if resultarray[ 10 ] > 0.0:  # if resonance occurs
        with open(path + 'brunel_array_results_{0}.csv'.format(
                simulation_index), 'a') as output:
            np.savetxt(output, resultarray, fmt="%12.6G",
                       newline=' ')
            output.write(' \n')
            output.close()
    else:
        with open(path + 'brunel_array_results_{0}_0.csv'.format(
                simulation_index), 'a') as output:
            np.savetxt(output, resultarray, fmt="%12.6G",
                       newline=' ')
            output.write(' \n')
            output.close()


def construct_resultarray(resultlist, networkparamdict):
    resultarray = np.array(resultlist)
    paramlist = [ networkparamdict[ 'p_rate' ],
                  networkparamdict[ 'C_m' ],
                  networkparamdict[ 'g' ],
                  networkparamdict[ 'g_1' ],
                  networkparamdict[ 'tau_1' ],
                  networkparamdict[ 'V_dist' ],
                  networkparamdict[ 'V_dist2' ] ]
    paramarray = np.array(paramlist, dtype=float)
    resultarray = np.hstack((paramarray, resultarray))
    return resultarray

def configure_nest_kernel(cluster=cluster):
    dt = 0.1
    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    if cluster:
        nest.SetKernelStatus(
                {"resolution":        dt,
                 "print_time":        False,
                 "overwrite_files":   True,
                 "local_num_threads": 16})
    else:
        nest.SetKernelStatus(
                {"resolution":        dt,
                 "print_time":        True,
                 "overwrite_files":   True})
    try:
        nest.Install("gif2_module")
    except:
        pass


def convert_to_paramdict(input):
    """
    Convert a row of an configuration_extractor output to run_brunel input.
    """
    networkparamdict = {
        'p_rate':   0,
        'C_m':      input[ 1 ],
        'g':        input[ 2 ],
        'g_1':      input[ 3 ],
        'tau_1':    input[ 4 ],
        'V_dist':   input[ 5 ],
        'V_dist2':  input[ 6 ],
        'K_ext':   {'E': 2000, 'I': 1900, 'R': 2000},
        'bg_rate':  8., 
        'fraction': input[ 7 ]}
    return networkparamdict

def print_results(i, somedict=None):
    NE = int(4850 * (1-i[7]))
    NR = int(4850 * i[7])
    print('Fraction of resonating {1} among excitatory {2} neurons: {0}'.format(i[7], NR, NE+NR))
    print('Simulated time: {0}'.format(i[16]))
    print('Pop rates (E | I | R): {0} | {1} | {2}'.format(i[7+1], i[7+2], i[7+3]))
    print('CVs (E | I | R | total): {0} | {1} | {2} | {3}'.format(i[7+4], i[7+5], i[7+7], i[7+7]))


# ****************************************************************************************************************************
# FORMER BETTER CONFIGURATION GENERATOR
# ****************************************************************************************************************************
def looptester():
    """
    This is a default script that should be adapted to the respective purpose.

    It is intended to compute the array length required to adequately scan
    parameter space for a suitable configuration of a GIF2 neuron in a Brunel
    network.

    Insert k as the array length in one of the shell scripts in this folder!
    """

    from mingtools1 import predict_str_freq

    dV1_range = np.arange(3.0, 9.5, 0.05)
    dV2_range = np.arange(4.0, 9.5, 0.05)
    k = 0
    # conn_probs_new_L5
    for i in dV1_range:
        for j in dV2_range:
            k += 1
    print(k)


def configuration_extractor(resultfile, criterion='new'):
    data = np.loadtxt('brunel_results/brunel_array_results_{0}.csv'.format(
                            resultfile))
    if criterion == 'old':
        """
        Detects three test arrays with rows containing parameter configs.
        Use function convert_to_paramdict to obtain dict for run_brunel.
        """
        exin_rates_lower_bound = 5.0
        exin_rates_upper_bound = 7.5
        distance_r_to_exin_bound = 2.0

        # find all rows with acceptable E/I rates:
        mean_exin = np.mean((data[:,8:9]),axis=1)
        mean_exin_accept = np.all((
            mean_exin <= exin_rates_upper_bound, 
            mean_exin >=exin_rates_lower_bound), axis=0)

        # is R rate also acceptable?
        dist_r_exin = np.abs(data[:,10] - mean_exin)
        dist_r_exin_accept = np.abs(dist_r_exin) <= distance_r_to_exin_bound

    elif criterion == 'new':
        # 'L5' : {'E':8.142, 'I':9.078},
        ex_lower_bound = 8.142 - 0.25
        ex_upper_bound = 8.142 + 0.25
        in_lower_bound = 9.078 - 0.25
        in_upper_bound = 9.078 + 0.25

        mean_ex = data[ :, 8 ]
        mean_in = data[ :, 9 ]
        mean_re = data[ :, 10 ]

        mean_in_accept = np.all((
            mean_in <= in_upper_bound, mean_in >= in_lower_bound), axis=0)
        mean_re_accept = np.all((
            (mean_ex + mean_re) / 2.0 <= ex_upper_bound,
            (mean_ex + mean_re) / 2.0 >= ex_lower_bound), axis=0)

    # where do both criteria hold?
    rates_accept = np.all((mean_in_accept, mean_re_accept), axis=0)
    print('We have {0} results with acceptable rates.'.format(
        rates_accept.sum()))

    cv_lower_bound = 0.65
    cv_upper_bound = 1.25
    distance_penalty = 6.0

    # if the rates fit, what are the resulting CVs?
    cvs = data[:, 14]
    cvs_accept = np.all((
        cvs <= cv_upper_bound, cvs >= cv_lower_bound), axis=0)

    all_accept = np.all((rates_accept, cvs_accept), axis=0)
    # also acceptable rates?
    print('{0} among these have CVs between {1} and {2}'.format(
        all_accept.sum(), cv_lower_bound, cv_upper_bound))

    if criterion == 'old':
        # of the remaining configurations, which has...
        # ... the largest dV2?
        testindices1 = data[ :, 6 ] == np.amax(data[ all_accept, 6 ])
        testconfigs1 = data[ testindices1, : ]
        # ... the largest total dV?
        testvalue2 =  np.amax(data[ all_accept, 6 ] + data[ all_accept, 5 ])
        testindices2 = (data[ :, 5 ] + data[ :, 6 ]) == testvalue2
        testconfigs2 = data[ testindices2, : ]
        # ... the lowest RMSE of rate difference and dV total?
        # ... not yet implemented
        # ... the lower RMSE of rate difference and dV2?
        testvalue3 = np.sqrt((distance_penalty * np.ones_like(data[ 
            all_accept, 6 ]) - data[ all_accept, 6 ])**2 + dist_r_exin[
            all_accept ]**2)
        testindices3 = np.sqrt((distance_penalty * np.ones_like(data[
         :, 6 ]) - data[ :, 6 ])**2 + dist_r_exin**2) == np.amin(testvalue3)
        testconfigs3 = data[ testindices3, : ]
    elif criterion == 'new':
        # of the reamaining configurations, which has...
        # ... the closest mean rates?
        mean_reex = np.abs(np.mean(
                            [ data[ :, 8 ], data[ :, 10 ] ], axis=0) - 8.142)
        testvalue1 = np.amin(mean_reex[ all_accept ])
        testindeices1 = (mean_reex == testvalue1)
        testconfigs1 = data[ testindeices1, : ]
        # the closest rates in E and R to previous E?
        testconfigs2 = testconfigs1
        # the smallest RMSE between difference in E and mean difference in E/R?
        dist_ex = np.abs(mean_ex - 8.142)
        total_dist = dist_ex**2 + mean_reex**2
        testvalue3 = np.amin(total_dist[ all_accept ])
        testindices3 = (total_dist == testvalue3)
        testconfigs3 = data[ testindices3, : ]

    return testconfigs1, testconfigs2, testconfigs3

# ****************************************************************************************************************************
# MICROCIRCUIT CONNECTIVITY TRANSFORMER FUNCTIONS
# ****************************************************************************************************************************

def find_Nsyn(Cold, Npre, Npost):
    """
    compute h3 = syn_nums = actual number of synapses = K_in * Npost
    """
    assert Npost != 0
    assert Npre != 0
    h1 = np.log(1. - Cold)
    h2 = (Npre * Npost - 1.) / (Npre * Npost)
    h3 = h1 / np.log(h2)
    return h3


def find_new_C(K, Npre, Npost):
    """
    find new connection probability
    """
    assert Npost != 0
    assert Npre != 0
    h1 = (Npre * Npost - 1.) / float(Npre * Npost)
    h2 = np.power(h1, K)
    h3 = 1. - h2
    return h3


def compute_new_connectivity(conn_probs_old, neuron_nums_new, neuron_nums_old, layers, 
                             pops_old, pops_new, structure_new, structure_old):
    """
    the core function.
    """
    n_new_pops_per_layer = len(pops_new)
    n_pops_per_layer = len(pops_old)
    n_layers = len(layers)
    n_pops_new = n_layers * n_new_pops_per_layer
    n_pops_old = n_layers * n_pops_per_layer
    conn_probs_new = np.zeros((n_pops_new, n_pops_new))
    syn_nums_new = np.zeros((n_pops_new, n_pops_new))
    syn_nums_old = np.zeros((n_pops_old, n_pops_old))

    for tl, tl_dict in neuron_nums_new.iteritems():
        for tp, tp_size in tl_dict.iteritems():
            bindex_t = structure_new[ tl ][ tp ]

            for sl, sl_dict in neuron_nums_new.iteritems():
                for sp, sp_size in sl_dict.iteritems():
                    bindex_s = structure_new[ sl ][ sp ]
                    if (sp_size == 0 or tp_size == 0):                         # target or source have size 0?
                        conn_probs_new[ bindex_t, bindex_s ] = 0
                        syn_nums_new[ bindex_t, bindex_s ] = 0
                    else:
                        # collect the old connectivity:
                        if 'R' in sp:                                          # take source connectivity from E for R?
                            source_index = structure_old[ sl ][ 'E' ]          # get the source index for the old connectivity list
                            Npre_old = neuron_nums_old[ sl ][ 'E' ]
                        else:
                            source_index = structure_old[ sl ][ sp ]           # get the source index for the old connectivity list
                            Npre_old = neuron_nums_old[ sl ][ sp ]
                        if 'R' in tp:                                          # take target connectivity from E for R?
                            target_index = structure_old[ tl ][ 'E' ]          # get the target index for the old connectivity list
                            Npost_old = neuron_nums_old[ tl ][ 'E' ]
                        else:                                                  # just E and I populations connecting
                            target_index = structure_old[ tl ][ tp ]           # get the target index for the old connectivity list
                            Npost_old = neuron_nums_old[ tl ][ tp ]
                        Cold = conn_probs_old[ target_index ][ source_index ]  # get the 'old' connectivity list entry

                        # compute new connectivity:
                        if Cold == 0:                                          # was connectivity 0 anyway?
                            conn_probs_new[ bindex_t, bindex_s ] = 0
                        else:                                                  # compute new connectivity
                            n_syn = find_Nsyn(Cold, Npre_old, Npost_old)       # replaces K with Nsyn, find_C with find_Nsyn
                            rel = float(sp_size) / float(Npre_old) * float(tp_size) / float(Npost_old)  # number of synapses with same relation as between old population sizes
                            # put synapse number into matrix
                            syn_nums_new[ bindex_t, bindex_s ] = n_syn * rel
                            syn_nums_old[ target_index, source_index ] = n_syn

                            conn_probs_new_L5 = find_new_C(
                                float(syn_nums_new[ bindex_t, bindex_s ]), #* rel, 
                                neuron_nums_new[ sl ][ sp ], 
                                neuron_nums_new[ tl ][ tp ])
                            conn_probs_new[ bindex_t, bindex_s ] = conn_probs_new_L5
    return conn_probs_new, syn_nums_new, syn_nums_old



