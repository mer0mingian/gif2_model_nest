# coding=utf-8
"""
This file collects all currerntly relevant functions for Simulating the
isolated L5 from the Microcircuit, including the resonating population,
as a Brunel network to find a reasonable parameter configuration for
the GIF2 model for application in the multi-area model.

options to take sys.argv's, e.g. from gif2brunel.sh:
1 - cluster variable

NEW:
- normal clipped distribution for synaptic weights and connections
"""

import os
import copy
import numpy as np
import time
import sys
import shutil
import contextlib
import matplotlib as mpl
if ('blaustein' in os.environ['HOSTNAME'].split('.')[ 0 ]):
        cluster = True
        mpl.use('Agg')
        print('recognised cluster')
else: 
    cluster = False
    print('working locally')
import matplotlib.pyplot as plt
import nest
from elephant.statistics import isi, cv
from itertools import product
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

from mingtools1 import *
from brunel_helper import *

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
# THE L5 BRUNEL NETWORK
# ****************************************************************************************************************************

def run_brunel(networkparamdict, external_drive=True, plotting=True, 
               verbose=True):
    """ a version that is driven just as the MC L5 """
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
    if 'simtime' in networkparamdict.keys():
        simtime = networkparamdict[ 'simtime' ]
    else:
        simtime = 2500.0  # Simulation time in ms

    # interpopulation delays from the Microcircuit:
    # synaptic delays in ms
    delay_dicts = {'E': {'distribution': 'normal_clipped', 
                        'low': 0.1, 
                        'mu': 1.5, 
                        'sigma': 0.75
                        },
                   'I': {'distribution': 'normal_clipped', 
                        'low': 0.1,
                        'mu': 0.8, 
                        'sigma': 0.4
                        } ,
                   'R': {'distribution': 'normal_clipped', 
                         'low': 0.1, 
                         'mu': 1.5, 
                         'sigma': 0.75
                        },
                  }

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


    weight_dict = define_synweights(Cm, tau_m, tauSyn, J=0.15, g=4.0)

    # verbosity([ weight_dict[ 'E' ], weight_dict[ 'I' ], 
    #             weight_dict[ 'ext' ] ], v, 'weights:\n')


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
    # These are the synapse numbers (K only in the Potjans paper):
    syn_nums_L5 = np.array(syn_nums_new[ 6:9, 6:9 ], dtype=int)

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

    for spikedetect in [espikes, rspikes, ispikes]:
        nest.SetStatus(spikedetect, [
            {"label":   "brunel-py-in", "withtime": True, "withgid": True,
             "to_file": False, 'start': recstart} ])

    # try, in case we loop over the function, not the script
    excite = {'model': 'static_synapse', 
              'weight': weight_dict[ 'E' ], 
              'delay': 0.8}

    # ###################### Connecting ######################
    
    # DETECTORS
    connect(nodes_ex, espikes, syn_spec=excite)
    connect(nodes_in, ispikes, syn_spec=excite)
    connect(nodes_re, rspikes, syn_spec=excite)

    # BACKGROUND 
    for i in np.arange(len(nodes)):
        connect([ noise[ i ] ], list(nodes[ i ]), 
                syn_spec={'weight': weight_dict[ 'ext' ], "delay": 0.75}, 
                conn_spec={'rule': 'all_to_all'})

    # POPULATIONS
    for sp, spx in pops_new.iteritems():
        for tp, tpx in pops_new.iteritems():
            # connect populations in L5
            connect(list(nodes[ spx ]), list(nodes[ tpx ]),
                    conn_spec={'rule': 'fixed_total_number', 
                               'N': syn_nums_L5[ tpx, spx ]},
                    syn_spec={'weight': weight_dict[ sp ], 
                              'delay': delay_dicts[ sp ]})
            # syn_numss K: source = col, target = row
            # Weights J: source = row, target = col

    # LAYERS
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
                    if np.any([syns == 0.0, 
                              neuron_nums_new[ layer ][ source_pop ] == 0,
                              neuron_nums_new[ 'L5' ][ target_pop ] == 0]):
                        indegree = 0.0
                    else:
                    # compute the indegree from synapse numbers
                        indegree = syns / float(
                                        neuron_nums_new[ 'L5' ][ target_pop ])
                    # if verbose:
                    #     print("INDEGREE {3}{1}->{2}: {0}".format(
                    #                 indegree, source_pop, target_pop, layer))
                    rate = rates[ layer ][ source_pop ]
                    virtual_rate += rate * indegree
                # distinguish between weights for excitatroy/inhibitory drive:
                # connect:
                nest.SetStatus([ layer_nodes[ k ] ],
                    {"rate":      virtual_rate  , "frequency": 10.0,
                     "amplitude": 0.025 * 0.0, "phase": 0.0})
                connect([ layer_nodes[ k ] ], 
                          nodes[ pops_new[ target_pop ] ],
                          conn_spec={'rule': 'all_to_all'},
                          syn_spec={'weight': weight_dict[ source_pop ], 
                                    'delay': delay_dicts[ source_pop ]})
                if verbose:
                    print(
                        'Connecting {0}->{1} with {2} rate'.format(
                            source_pop, target_pop, virtual_rate))
                k += 1


    endbuild = time.time()

    # SIMULATE
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

    # for sp in [ spiketrains_ex, spiketrains_in, spiketrains_re, spiketrains_all ]:
    #     sp = sp[ np.isnan(sp) == False ] 
    # cv_ex = np.mean([ cv(isi(spiketrain)) for spiketrain in spiketrains_ex ])
    # cv_in = np.mean([ cv(isi(spiketrain)) for spiketrain in spiketrains_in ])
    # cv_re = np.mean([ cv(isi(spiketrain)) for spiketrain in spiketrains_re ])
    # cv_all = np.mean([ cv(isi(spiketrain)) for spiketrain in spiketrains_all ])


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
                         markersize=1)
        for spiketrain in spiketrains_re:
            if np.any(spiketrain):
                offset += 1
                figu.add_subplot(1, 1, 1)
                plt.plot(spiketrain - recstart,
                         offset * np.ones_like(spiketrain),
                         'g.',
                         markersize=1)
        for spiketrain in spiketrains_in:
            if np.any(spiketrain):
                offset += 1
                figu.add_subplot(1, 1, 1)
                plt.plot(spiketrain - recstart,
                         offset * np.ones_like(spiketrain),
                         'r.',
                         markersize=1)
        plt.ylim(0, offset)
        plt.xlim(0, simtime)
        plt.savefig('driven_plots/Rasterplot_{2}_{0}_{1}.png'.format(
                int(fraction) * 10, 10, int(p_rate)))

    detectordict = {'ex_dict': events_ex,
                    'in_dict': events_in,
                    're_dict': events_re}

    return [ fraction, rate_ex, rate_in, rate_re, cv_ex, cv_in, cv_re, cv_all,
             recstart, simtime ], [ spiketrains_all, spiketrains_ex,
                                    spiketrains_in, spiketrains_re, espikes,
                                    ispikes, rspikes, detectordict]
    # TODO: actually, all the spiketrain outputs lack GIDs and are obsolete

# ****************************************************************************************************************************
# MAIN FILE CODE
# ****************************************************************************************************************************


if __name__ == '__main__':
    if int(sys.argv[ 1 ]) == 1:
        purpose = 'paramscan'
    elif int(sys.argv[ 1 ]) == 2:
        purpose = 'ratescan'
    elif int(sys.argv[ 1 ]) == 0:
        purpose = 'single sim'
    elif int(sys.argv[ 1 ]) == 3:
        purpose = 'fractionscan'


    if purpose == 'paramscan':
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
                    writeout(int(sys.argv[ 2 ]), resultarray, path='brunel_results/')

    elif purpose == 'single sim': 
        """
        This extracts a set of possibly suitable configurations from one of the
        brunel_array_results_X files and runs a single network.
        """
        print('Importing suitable GIF2 configurations')
        testconfigs1, testconfigs2, testconfigs3 = configuration_extractor(20)
        print('Importing complete.')
        nest.ResetKernel()
        configure_nest_kernel(cluster=cluster)
        nest.SetKernelStatus({'print_time': True})
        print('Nest Kernel configured')

        networkdict = convert_to_paramdict(testconfigs1[0,:])
        networkdict[ 'V_dist2' ] = 7.75
        networkdict[ 'V_dist' ] = 3.8
        networkdict[ 'fraction' ] = 0.5
        networkdict[ 'simtime' ] = 10000.
        print('Chosen network parameters:')
        if sys.argv[ 1 ]:
            for i, j in networkdict.iteritems():
                print('{0} = {1}'.format(i,j))

        print('Simulating')
        resultlist, spikelists = run_brunel(networkdict, 
            external_drive=True, verbose=True, plotting=True)

        print('Done. Congregating results. \n')
        resultarray = construct_resultarray(resultlist, networkdict)

        print_results(resultarray)


        # do we also want some plots?
        analytics = False
        if analytics == True:
            # First, collect all the spike data
            detectordict = spikelists[ -1 ]
            spikearray_all = np.array([[0.],[0.]])
            for key in detectordict.keys():
                detectordict[ key ][ 'spikearray' ] = np.vstack((
                               detectordict[ key ][ u'senders' ],
                               detectordict[ key ][ u'times' ]))
                spikearray_all = np.hstack((spikearray_all, 
                               detectordict[ key ][ 'spikearray' ]))
            # cut off the initialisation point:
            spikearray_all = spikearray_all[ :, 1: ]


            # spectrum
            t_min = 0.0 # recstart
            t_max = networkdict[ 'simtime' ] # t_min + simtime

            # concatenate all spikes
            timeseries_all = pop_rate_time_series(spikearray_all.T, 
                5915, t_min, t_max)

            # # compute the spectrum
            power, freq = spectrum(spikearray_all, 5915, t_min, t_max,
                               resolution=2.0, kernel='binned', Df=None)
            # # plotting
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('Power spectrum Layer 5')
            ax.plot(freq, power, color='k', markersize=3)
            ax.set_xlabel('Frequency [Hz]', size=16)
            ax.set_ylabel('Power', size=16)
            ax.set_xlim(0.0, 500.0)
            ax.set_yscale("Log")
            plt.savefig('driven_plots/spectrum_test.png')
            # plt.savefig(os.path.join(self.output_dir, self.label + 
            #     '_power_spectrum_' + area + '_' + pop + '.' + 
            #     keywords['output']))



            # USE NITIME FOR PLOTTING THE SPECTRUM:
            # import scipy.signal as sig
            # import scipy.stats.distributions as dist
            # import nitime.algorithms as tsa
            # # import nitime.utils as utils
            # # from nitime.viz import winspect
            # from nitime.viz import plot_spectral_estimate

            # # def dB(x, out=None):
            # #     if out is None:
            # #         return 10 * np.log10(x)
            # #     else:
            # #         np.log10(x, out)
            # #         np.multiply(out, 10, out)
            # # ln2db = dB(np.e)
            # # freqs, d_psd = tsa.periodogram(timeseries_all)

            # f, adaptive_psd_mt, nu = tsa.multi_taper_psd(timeseries_all, 
            #                                 adaptive=True, jackknife=False)
            # # dB(adaptive_psd_mt, adaptive_psd_mt)
            # # p975 = dist.chi2.ppf(.975, nu)
            # # p025 = dist.chi2.ppf(.025, nu)
            # # l1 = ln2db * np.log(nu / p975)
            # # l2 = ln2db * np.log(nu / p025)
            # # hyp_limits = (adaptive_psd_mt + l1, adaptive_psd_mt + l2)

            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.set_title('Power spectrum Layer 5')
            # ax.plot(f, adaptive_psd_mt, color='k', markersize=3)
            # ax.set_yscale("Log")
            # # fig06 = plot_spectral_estimate(freqs, psd, (adaptive_psd_mt,), hyp_limits,
            # #                        elabels=('MT with adaptive weighting and 95% interval',))
            # plt.savefig('driven_plots/spectrum_test.png')


    elif purpose == 'rate_scan': 
        pass

    elif purpose == 'fraction_scan':
        pass

    """
    fraction = 0.5
    V_dist2 = 7.95
    p_rate = 0
    bg_rate = 8.0
    g = 25.0
    g_1 = 32.5
    V_dist = 3.6
    C_m = 250.0
    K_ext = {'I': 1900, 'R': 2000, 'E': 2000}
    tau_1 = 50.0

    'V_dist': 3.8,
    'V_dist2': 7.75,
    'g_1': 32.5,
    'tau_1': 50.0

    Will have equal firing rates at fraction 0.5
    """