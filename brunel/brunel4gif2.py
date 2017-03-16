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
from itertools import product
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

from mingtools1 import *
from brunel_helper import *
from brunel_vistools import *

# ****************************************************************************************************************************
# THE L5 BRUNEL NETWORK
# ****************************************************************************************************************************

def run_brunel(networkparamdict, external_drive=True, plotting=True, 
               verbose=True):
    """ a version that is driven just as the MC L5 """
    assert 'K_ext' in networkparamdict, 'K_ext missing, invalid networkparamdict'
    assert 'bg_rate' in networkparamdict, 'bg_rate missing, invalid networkparamdict'
    fraction = networkparamdict[ 'fraction' ]
    modulation = networkparamdict[ 'modulation' ]
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
             "amplitude": bg_rate * K_ext[ i ] * modulation * 0.0, 
             "frequency": 10.0, "phase": 0.0} )

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
                     "amplitude": 0.0, "phase": 0.0})
                # OSCILLATORY DRIVE
                if np.all([ target_pop == 'R', source_pop == 'E' ]):
                    print('Inserting oscillatory drive')
                    nest.SetStatus([ layer_nodes[ k ] ], 
                        {"amplitude": virtual_rate * modulation})

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

    # ###################### SIMULATE ######################
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

    detectordict = {'ex_dict': events_ex,
                    'in_dict': events_in,
                    're_dict': events_re}

    # ###################### EVALUATE ######################
    # Rates:
    if NE > 0:
        rate_ex = nevents_ex / simtime * 1000.0 / NE
    else:
        rate_ex = 0.0
    if NR > 0:
        rate_re = nevents_re / simtime * 1000.0 / NR
    else: 
        rate_re = 0.0
    rate_in = nevents_in / simtime * 1000.0 / NI

    # Define Spiketrains
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

    # CVs:
    try:
        from elephant.statistics import isi, cv

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


        return [ fraction, rate_ex, rate_in, rate_re, cv_ex, cv_in, cv_re, 
                 cv_all, recstart, simtime ], [ spiketrains_all, 
                 spiketrains_ex, spiketrains_in, spiketrains_re, espikes,
                 ispikes, rspikes, detectordict]
    except ImportError:
        return [ fraction, rate_ex, rate_in, rate_re, -1, -1, -1, -1,
                 recstart, simtime ], [ spiketrains_all, spiketrains_ex,
                 spiketrains_in, spiketrains_re, espikes, ispikes, rspikes, 
                 detectordict]

    # TODO: actually, all the spiketrain outputs lack GIDs and are obsolete

# ****************************************************************************************************************************
# MAIN FILE CODE
# ****************************************************************************************************************************


if __name__ == '__main__':
    if int(sys.argv[ 1 ]) == 1:        purpose = 'paramscan'
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
                            modulation = 0.0, C_m = 250.0, g_1 = 32.5, bg_rate=8.0,
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
        testconfigs1, testconfigs2, testconfigs3 = configuration_extractor(21)
        print('Importing complete.')
        nest.ResetKernel()
        configure_nest_kernel(cluster=cluster)
        nest.SetKernelStatus({'print_time': True})
        print('Nest Kernel configured')

        networkdict = convert_to_paramdict(testconfigs1[0,:])
        networkdict[ 'V_dist2' ] = 7.75
        networkdict[ 'V_dist' ] = 3.8
        networkdict[ 'fraction' ] = 0.5
        networkdict[ 'simtime' ] = 7500.
        networkdict[ 'modulation' ] = float(sys.argv[ 3 ])
        print('Chosen network parameters:')
        if int(sys.argv[ 1 ]):
            for i, j in networkdict.iteritems():
                print('{0} = {1}'.format(i,j))

        print('Simulating')
        resultlist, spikelists = run_brunel(networkdict, 
            external_drive=True, verbose=True, plotting=True)

        print('Done. Congregating results. \n')
        resultarray = construct_resultarray(resultlist, networkdict)

        print_results(resultarray)

        # do we also want some plots?
        plotting = True
        targetname = str(sys.argv[ 2 ])
        if plotting:
            plt.clf()
            print('Creating plots.')
            rasterplot(spikelists, networkdict, targetname, length=1000)
            multi_taper_spectrum(networkdict, spikelists, targetname,
                                hist_binwidth=2.0, 
                                NW=16, 
                                xlims=[ 0.0, 150.0 ])

    elif purpose == 'rate_scan': 
        pass

    elif purpose == 'fraction_scan':
        pass

    """
    fraction = 0.5
    V_dist2 = 7.95
    modulation = 0
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