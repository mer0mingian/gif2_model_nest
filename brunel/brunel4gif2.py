# coding=utf-8
"""
This file collects all currerntly relevant functions for Simulating the
isolated L5 from the Microcircuit, including the resonating population,
as a Brunel network to find a reasonable parameter configuration for
the GIF2 model for application in the multi-area model.

options to take sys.argv's, e.g. from gif2brunel.sh:
1 - cluster variable
"""

import os
import copy
import numpy as np
import time
import sys
import shutil
import contextlib
import matplotlib as mpl
if ('blaustein' in os.environ['HOSTNAME'].split('.')[ 0 ] or
    int(sys.argv[ 1 ]) == 0):
        cluster = True
        mpl.use('Agg')
        print('recognised cluster')
else: 
    cluster = False
    print('working locally')
import matplotlib.pyplot as plt
import nest
from mingtools1 import *
from elephant.statistics import isi, cv
from itertools import product
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


# ****************************************************************************************************************************
# BRUNEL NETWORK HELPER FUNCTIONS
# ****************************************************************************************************************************

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)

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

def configuration_extractor():
    """
    Detects three test arrays with rows containing parameter configs.
    Use function convert_to_paramdict to obtain dict for run_brunel.
    """
    exin_rates_lower_bound = 5.0
    exin_rates_upper_bound = 7.5
    distance_r_to_exin_bound = 2.0
    cv_lower_bound = 0.85
    cv_upper_bound = 1.15
    distance_penalty = 6.0

    data = np.loadtxt('brunel_results/brunel_array_results_15.csv')
    # find all rows with acceptable E/I rates:
    mean_exin = np.mean((data[:,8:9]),axis=1)
    mean_exin_accept = np.all((
        mean_exin <= exin_rates_upper_bound, 
        mean_exin >=exin_rates_lower_bound), axis=0)

    # is R rate also acceptable?
    dist_r_exin = np.abs(data[:,10] - mean_exin)
    dist_r_exin_accept = np.abs(dist_r_exin) <= distance_r_to_exin_bound

    # where do both criteria hold?
    rates_accept = np.all((mean_exin_accept, dist_r_exin_accept), axis=0)
    print('We have {0} results with acceptable rates.'.format(
        rates_accept.sum()))

    # if the rates fit, what are the resulting CVs?
    cvs = data[:, 14]
    cvs_accept = np.all((
        cvs <= cv_upper_bound, cvs >= cv_lower_bound), axis=0)

    all_accept = np.all((rates_accept, cvs_accept), axis=0)
    # also acceptable rates?
    print('{0} among these have CVs between {1} and {2}'.format(
        all_accept.sum(), cv_lower_bound, cv_upper_bound))


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



# ****************************************************************************************************************************
# THE L5 BRUNEL NETWORK
# ****************************************************************************************************************************

def run_brunel(networkparamdict, external_drive=True, plotting=True, verbose=True):
    """ a version that is driven just as the MC L5
    !!! WARNING !!!
    Possibly faulty connection for the poisson generators ->
    return to all_to_all conn_rule.
    """
    assert 'K_ext' in networkparamdict, 'K_ext missing, invalid networkparamdict'
    assert 'bg_rate' in networkparamdict, 'bg_rate missing, invalid networkparamdict'
    fraction = networkparamdict[ 'fraction' ]
    p_rate = networkparamdict[ 'p_rate' ]
    C_m = networkparamdict[ 'C_m' ]
    gm = networkparamdict[ 'g' ]
    g1 = networkparamdict[ 'g_1' ]
    tau_1 = networkparamdict[ 'tau_1' ]
    V_dist = networkparamdict[ 'V_dist' ]
    Vdist2 = networkparamdict[ 'V_dist2' ]
    K_ext = networkparamdict[ 'K_ext' ]
    bg_rate = networkparamdict[ 'bg_rate' ]
    v = verbose

    recstart = 2500.0
    simtime = 2500.0  # Simulation time in ms
    delay = 0.8  # synaptic delay in ms
    delay_ex = 1.5
    delay_in = 0.8
    g = 4.0  # ratio inhibitory weight/excitatory weight

    N = 4850
    NE = int((1.0 - fraction) * N)
    NR = int(fraction * N)
    NI = 1065
    N_neurons = [ NE, NI, NR ] # number of neurons in total
    verbosity(N_neurons, v, 'Neuron numbers: ')

    # PSP_ext = 0.15  # mean EPSP amplitude in mV for external drive
    theta = -50.0  # membrane threshold potential in mVfrom stats
    tauSyn = 0.5
    tau_m = 10.0
    Cm = 250.0

    neuron_params = {
        "C_m":        Cm,
        "tau_m":      tau_m,
        "tau_syn_ex": tauSyn,
        "tau_syn_in": tauSyn,
        "t_ref":      2.0,
        "E_L":        -65.0,
        "V_reset":    -65.0,
        "V_m":        -58.0,
        "V_th":       theta}

    neuron_params2 = {
        "tau_1":      tau_1,
        "C_m":        C_m,
        "tau_syn_ex": tauSyn,
        "tau_syn_in": tauSyn,
        "g_rr":       g1,
        "g":          gm,
        "V_m":        -58.0,
        "V_reset":    theta - V_dist,
        "E_L":        theta - V_dist - Vdist2,
        "V_th":       theta,
        "t_ref":      2.0}

    J = 0.15  # postsynaptic potential amplitude in mV
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

    # this are the 87.8 pA
    synweight_ex = PSC_e
    synweight_in = PSC_i
    synweight_ext = PSC_ext
    # verbosity([ synweight_ex, synweight_in, synweight_ext ], v, 'weights:\n')


    conn_probs_old, layers, pops_old, pops_new, structure_new, structure_old, neuron_nums_new, neuron_nums_old, rates = define_structures(NE, NI, NR)

    conn_probs_new, syn_nums_new, syn_nums_old = compute_new_connectivity(
                                     conn_probs_old, neuron_nums_new,
                                     neuron_nums_old, layers, pops_old,
                                     pops_new, structure_new, structure_old)
    conn_probs_old = np.asarray(conn_probs_old)
    conn_probs_new_L5 = conn_probs_new[ 6:9, 6:9 ]
    verbosity(conn_probs_new_L5, v, 'Potjans-Connectivity: \n')
    # Compute the synapse numbers:
    """ Compute numbers of synapses assuming binomial degree distributions and
    allowing for multapses(see Potjans and Diesmann 2012 Cereb Cortex Eq.1)"""

    # import pdb; pdb.set_trace()
    # These are the synapse numbers (K only in the Potjans paper)
    syn_nums_L5 = np.array(syn_nums_new[ 6:9, 6:9 ], dtype=int)
    # Compute the syn_nums as syn_num / Npost:
    # if verbose:
    #     print('{0} {1}'.format('Syn_nums: \n', np.array(syn_nums_new, dtype=int)))
    #     print('{0} {1}'.format('Syn_nums_old: \n', np.array(syn_nums_old, dtype=int)))

    # ###################### Setting up Nodes ######################

    startbuild = time.time()
    nest.SetDefaults("iaf_psc_exp", neuron_params)
    nest.SetDefaults("gif2_psc_exp", neuron_params2)
    nodes_ex = create("iaf_psc_exp", NE)
    nodes_re = create("gif2_psc_exp", NR)
    nodes_in = create("iaf_psc_exp", NI)
    nodes = [ nodes_ex, nodes_in, nodes_re ]
    noise = nest.Create("sinusoidal_poisson_generator", 3)
    espikes = nest.Create("spike_detector")
    rspikes = nest.Create("spike_detector")
    ispikes = nest.Create("spike_detector")
    layer_nodes = nest.Create("sinusoidal_poisson_generator", 6) # 9 before

    # ###################### Configuring ######################

    # set the drive levels for all populations
    for i, j in pops_new.iteritems():
        nest.SetStatus([ noise[ j ] ],
            {"rate":      bg_rate * K_ext[ i ],
             "amplitude": 0.0, "frequency": 10.0, "phase": 0.0} )
        # if verbose:
        #     print('Set noise to {0} with rate {1} and syn_nums {2}'.format(
        #         i, bg_rate, K_ext[ i ]))

    for spikedetect in [espikes, rspikes, ispikes]:
        nest.SetStatus(spikedetect, [
            {"label":   "brunel-py-in", "withtime": True, "withgid": True,
             "to_file": False, 'start': recstart} ])

    # try, in case we loop over the function, not the script
    try:
        nest.CopyModel("static_synapse", "excite",
                       {"weight": synweight_ex, "delay": delay_in})
    except:
        pass

    # ###################### Connecting ######################
    
    # Connect pops to detectors
    connect(nodes_ex, espikes, syn_spec="excite")
    connect(nodes_in, ispikes, syn_spec="excite")
    connect(nodes_re, rspikes, syn_spec="excite")


    # connect the background drive 
    for i in np.arange(len(nodes)):
        connect([ noise[ i ] ], list(nodes[ i ]), 
                syn_spec={'weight': synweight_ext, "delay": delay_ex}, 
                conn_spec={'rule': 'all_to_all'})

    # Connect populations amongst each other
    for i in np.arange(len(nodes)):
        for j in np.arange(len(nodes)):
            if i == 1:              # if source inhibitory, set delays
                delay = delay_in
                weight = synweight_in
            else:
                delay = delay_ex
                weight = synweight_ex
            # connect populations in L5
            connect(list(nodes[ i ]), list(nodes[ j ]),
                conn_spec={'rule': 'fixed_total_number', 
                           'N': syn_nums_L5[ j, i ]},
                syn_spec={'weight': weight, "delay": delay})
                # syn_numss K: source = col, target = row
                # Weights J: source = row, target = col

    # connect drive from other layers
    verbosity(external_drive, v, "Using external layers' drive: ")
    if external_drive:
        k = 0
        external_layers = copy.copy(layers)
        external_layers.pop('L5')
        if verbose:
            print('Connecting external layers all_to_all')
        for target_pop in pops_new.keys():
            for source_pop in pops_old.keys():
                virtual_rate = 0.0
                for layer in external_layers.keys():
                    syns = syn_nums_new[ structure_new[ 'L5' ][ target_pop ],
                                      structure_new[ layer ][ source_pop ] ]
                    # if no synapses or one pop is empty
                    if (verbose and 'R' in target_pop):
                        print(neuron_nums_new[ 'L5' ][ target_pop ], syns)
                    if np.any([syns == 0.0, 
                              neuron_nums_new[ layer ][ source_pop ] == 0,
                              neuron_nums_new[ 'L5' ][ target_pop ] == 0]):
                        indegree = 0.0
                    else:
                    # compute the indegree from synapse numbers
                        indegree = syns / float(
                                        neuron_nums_new[ 'L5' ][ target_pop ])
                    if verbose:
                        print("INDEGREE {3}{1}->{2}: {0}".format(
                                    indegree, source_pop, target_pop, layer))
                    rate = rates[ layer ][ source_pop ]
                    virtual_rate += rate * indegree
                # distinguish between weights for excitatroy/inhibitory drive:
                if source_pop == 'I':
                    delay = delay_in
                    weight = synweight_in
                else:
                    delay = delay_ex
                    weight = synweight_ex
                # connect:
                nest.SetStatus([ layer_nodes[ k ] ],
                    {"rate":      virtual_rate  , "frequency": 10.0,
                     "amplitude": 0.025 * 0.0, "phase": 0.0})
                connect([ layer_nodes[ k ] ], 
                     nodes[ pops_new[ target_pop ] ],
                     conn_spec={'rule': 'all_to_all'},
                     syn_spec={'weight': weight, "delay": delay})
                if verbose:
                    print(
                        'Connecting {0}->{1} with {2} rate'.format(
                            source_pop, target_pop, virtual_rate))
                k += 1


    endbuild = time.time()

    # simulate
    if verbose:
        print('Starting simulation')
    nest.Simulate(simtime + recstart)
    endsimulate = time.time()

    # read out events
    events_ex = nest.GetStatus(espikes, "events")[ 0 ]
    events_re = nest.GetStatus(rspikes, "events")[ 0 ]
    events_in = nest.GetStatus(ispikes, "events")[ 0 ]
    nevents_ex = nest.GetStatus(espikes, "n_events")[ 0 ]
    nevents_re = nest.GetStatus(rspikes, "n_events")[ 0 ]
    nevents_in = nest.GetStatus(ispikes, "n_events")[ 0 ]

    if NE > 0:
        rate_ex = nevents_ex / simtime * 1000.0 / NE
    else:
        rate_ex = 0.0
    if NR > 0:
        rate_re = nevents_re / simtime * 1000.0 / NR
    else: 
        rate_re = 0.0
    rate_in = nevents_in / simtime * 1000.0 / NI


    # CVs:
    spiketrains_ex = list()
    spiketrains_in = list()
    spiketrains_re = list()
    for gid in nodes_ex:
        spiketrains_ex.append(
                events_ex[ 'times' ][ events_ex[ 'senders' ] == gid ])
    for gid in nodes_re:
        spiketrains_re.append(
                events_re[ 'times' ][ events_re[ 'senders' ] == gid ])
    for gid in nodes_in:
        spiketrains_in.append(
                events_in[ 'times' ][ events_in[ 'senders' ] == gid ])
    spiketrains_allex = spiketrains_ex + spiketrains_re
    spiketrains_all = spiketrains_allex + spiketrains_in

    if len(spiketrains_ex) > 0:
        cv_ex = np.nanmean(
                [ cv(isi(spiketrain)) for spiketrain in spiketrains_ex ])
    else:
        cv_ex = 0.0
    if len(spiketrains_re) > 0:
        cv_re = np.nanmean(
                [ cv(isi(spiketrain)) for spiketrain in spiketrains_re ])
    else:
        cv_re = 0.0
    cv_in = np.nanmean(
            [ cv(isi(spiketrain)) for spiketrain in spiketrains_in ])
    cv_all = np.nanmean(
            [ cv(isi(spiketrain)) for spiketrain in spiketrains_all ])

    if plotting:
        plt.clf()
        # Raster:
        figu = plt.figure("Rasterplot")
        offset = 0
        for spiketrain in spiketrains_ex:
            if np.any(spiketrain):
                offset += 1
                figu.add_subplot(1, 1, 1)
                plt.plot(spiketrain - recstart,
                         offset * np.ones_like(spiketrain),
                         'b.',
                         markersize=2)
        for spiketrain in spiketrains_re:
            if np.any(spiketrain):
                offset += 1
                figu.add_subplot(1, 1, 1)
                plt.plot(spiketrain - recstart,
                         offset * np.ones_like(spiketrain),
                         'g.',
                         markersize=2)
        for spiketrain in spiketrains_in:
            if np.any(spiketrain):
                offset += 1
                figu.add_subplot(1, 1, 1)
                plt.plot(spiketrain - recstart,
                         offset * np.ones_like(spiketrain),
                         'r.',
                         markersize=2)
        plt.ylim(0, offset)
        plt.xlim(0, simtime)
        plt.savefig('driven_plots/Rasterplot_{2}_{0}_{1}.png'.format(
                int(fraction) * 10, 10, int(p_rate)))

    return [ fraction, rate_ex, rate_in, rate_re, cv_ex, cv_in, cv_re, cv_all,
             recstart, simtime ], [ spiketrains_all, spiketrains_ex,
                                    spiketrains_in, spiketrains_re, espikes,
                                    ispikes, rspikes ]

# ****************************************************************************************************************************
# MAIN FILE CODE
# ****************************************************************************************************************************


if __name__ == '__main__':
    paramscan = True
    if paramscan:
        """ PARAMETERSCAN
            argvalues:  1 - cluster
                        2 - parameterscan
                        3 - scanning index """
        scanindex = int(sys.argv[ 3 ])

        dV1_range = np.arange(3.0, 9.5, 0.05)
        dV2_range = np.arange(4.0, 9.5, 0.05)
        k = 0
        # conn_probs_new_L5
        for dV1 in dV1_range:
            for dV2 in dV2_range:
                k += 1
                if np.all([ k >= scanindex * 100, k < scanindex * 100 + 100 ]):
                    nest.ResetKernel()
                    configure_nest_kernel(cluster=cluster)
                    K_ext = {'I': 1900, 'R': 2000, 'E': 2000}
                    networkdict = dict( fraction = 0.5, K_ext=K_ext, g = 25.0,
                            p_rate = 0.0, C_m = 250.0, g_1 = 32.5, bg_rate=8.0,
                            tau_1 = 50.0, V_dist = dV1, V_dist2 = dV2)
                    resultlist, spikelists = run_brunel(networkdict, 
                            external_drive=True, plotting=False, verbose=False)
                    resultarray = construct_resultarray(
                            resultlist, networkdict)
                    writeout(20, resultarray, path='brunel_results/')

    else: 
        """
        This extracts a set of possibly suitable configurations from one of the
        brunel_array_results_X files and runs a single network.
        """
        print('Importing suitable GIF2 configurations')
        testconfigs1, testconfigs2, testconfigs3 = configuration_extractor()
        print('Importing complete.')
        nest.ResetKernel()
        configure_nest_kernel(cluster=cluster)
        if int(sys.argv[ 1 ]):
            nest.SetKernelStatus({'print_time': True})
        print('Nest Kernel configured')

        networkdict = convert_to_paramdict(testconfigs1[0,:])
        networkdict[ 'V_dist2' ] = 6.3999
        networkdict[ 'V_dist' ] = 6.0
        networkdict[ 'fraction' ] = 0.5
        # print('Chosen network parameters:')
        # if sys.argv[ 1 ]:
        #     for i, j in networkdict.iteritems():
        #         print('{0} = {1}'.format(i,j))
        #     print('\n')

        print('Simulating')
        resultlist, spikelists = run_brunel(networkdict, 
                                            external_drive=True, verbose=False)

        print('Done. Congregating results. \n')
        resultarray = construct_resultarray(resultlist, networkdict)
        writeout(20, resultarray, path='brunel_results/')

        # print(resultarray)
        print_results(resultarray)

    """
    V_dist2 = 5.55078
    bg_rate = 8.0
    g = 25.0
    g_1 = 32.5
    V_dist = 6.1
    C_m = 250.0
    K_ext = {'I': 1900, 'R': 2000, 'E': 2000}
    tau_1 = 50.0
    Will have equal firing rates at fraction 0.5
    """