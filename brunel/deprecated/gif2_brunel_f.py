"""
Version 9.5
Option
alternative_connection = False

This version omits the fourth spike detector for the rasterplot and extracts
all data from the three spike detectors.

With p_rate2 = 85000. and h = 1.5, the firing rates of the pops are equal.
With p_rate2 = 70000. and h = 2.5, the firing rates of the pops are equal.
With p_rate2 = 66000. and h = 1.9, the firing rates of the pops are equal.


IMPORTANT!!! 
Moved the deprecated version of the run brunel function and renamed the 
currently included one.
"""

import os
import numpy as np
import time
import sys
import shutil
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import nest
from mingtools1 import *
from elephant.statistics import isi, cv
from mc_connectivity_transformer import compute_new_connectivity
from itertools import product


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

def configure_nest_kernel():
    dt = 0.1
    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    nest.SetKernelStatus(
            {"resolution":        dt,
             "print_time":        False,
             "overwrite_files":   True,
             "local_num_threads": 16})
    try:
        nest.Install("gif2_module")
    except:
        pass


def run_brunel(networkparamdict, plotting=True):
    """ a version that is driven just as the MC L5"""
    assert 'K_ext' in networkparamdict, 'invalid networkparamdict'
    assert 'bg_rate' in networkparamdict, 'invalid networkparamdict'
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

    recstart = 7500.0
    simtime = 10000.0  # Simulation time in ms
    delay = 0.8  # synaptic delay in ms
    delay_ex = 1.5
    delay_in = 0.8
    g = 4.0  # ratio inhibitory weight/excitatory weight

    N = 4850
    NE = int((1.0 - fraction) * N)
    NR = int(fraction * N)
    NI = 1065
    N_neurons = [ NE, NI, NR ] # number of neurons in total
    # record from N_rec neurons per population:
    N_rec = {'NE': int(NE / 10), 'NI': int(NI / 10), 'NR': int(NR / 10)}

    PSP_ext = 0.15  # mean EPSP amplitude in mV for external drive
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
        "V_m":        -65.0,
        "V_th":       theta}

    neuron_params2 = {
        "tau_1":      tau_1,
        "C_m":        C_m,
        "tau_syn_ex": tauSyn,
        "tau_syn_in": tauSyn,
        "g_rr":       g1,
        "g":          gm,
        "V_m":        theta - V_dist,
        "V_reset":    theta - V_dist,
        "E_L":        theta - V_dist - Vdist2,
        "V_th":       theta,
        "t_ref":      2.0}

    synweight = 87.8
    J = 0.15  # postsynaptic amplitude in mV
    J_ex = J  # amplitude of excitatory postsynaptic potential
    J_in = -g * J

    J *= np.array([ [ J_ex, J_ex, J_ex ],
                   [ J_ex, J_ex, J_ex ],
                   [ J_in, J_in, J_in ] ]) # * synweight

    """ Previously, we assumed a microcircuit connectivity of just on layer.
    Now we extract L5 from the full microcircuit instead."""

    #             2/3e      2/3i    4e      4i      5e      5i      6e      6i
    conn_probs = [ [ 0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0., 0.0076, 0. ],
                   [ 0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0., 0.0042, 0. ],
                   [ 0.0077, 0.0059, 0.0497, 0.135, 0.0067, 0.0003, 0.0453,
                     0. ],
                   [ 0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0., 0.1057, 0. ],
                   [ 0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204,
                     0. ],
                   [ 0.0548, 0.0269, 0.0257, 0.0022, 0.06, 0.3158, 0.0086,
                     0. ],
                   [ 0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396,
                     0.2252 ],
                   [ 0.0364, 0.001, 0.0034, 0.0005, 0.0277, 0.008, 0.0658,
                     0.1443 ] ]

    layers = {'L23': 0, 'L4': 1, 'L5': 2, 'L6': 3}
    pops = {'E': 0, 'I': 1}
    newpopdict = {'E': 0, 'I': 1, 'R': 2}

    new_structure = {'L23': {'E': 0, 'I': 1, 'R': 2},
                     'L4':  {'E': 3, 'I': 4, 'R': 5},
                     'L5':  {'E': 6, 'I': 7, 'R': 8},
                     'L6':  {'E': 9, 'I': 10, 'R': 11}}

    # Numbers of neurons in full-scale model
    N_full_new = {
        'L23': {'E': 20683, 'I': 5834, 'R': 0},
        'L4':  {'E': 21915, 'I': 5479, 'R': 0},
        'L5':  {'E': NE, 'I': NI, 'R': NR},
        'L6':  {'E': 14395, 'I': 2948, 'R': 0}
    }

    N_full_old = {
        'L23': {'E': 20683, 'I': 5834},
        'L4':  {'E': 21915, 'I': 5479},
        'L5':  {'E': 4850, 'I': 1065},
        'L6':  {'E': 14395, 'I': 2948}
    }

    K_bg = [ [ 1600, 1500 ], [ 2100, 1900 ], [ 2000, 1900 ], [ 2900, 2100 ] ]

    connmat, K_bg_new = compute_new_connectivity(conn_probs, N_full_new,
                                                 N_full_old, layers, pops,
                                                 newpopdict, new_structure,
                                                 K_bg)
    C = np.array(connmat[ 6:9, 6:9 ], dtype=int)

    """ Compute numbers of synapses assuming binomial degree distributions and
    allowing for multapses(see Potjans and Diesmann 2012 Cereb Cortex Eq.1)"""
    syn_nums = np.zeros((3, 3))
    for k in np.arange(3):
        for j in np.arange(3):
            conn_prb = C[ k, j ]
            prod = N_neurons[ k ] * N_neurons[ j ]
            syn_nums[ k, j ] = np.log(
                    1. - conn_prb) / np.log((prod - 1.) / prod)
    # These are the synapse numbers (K only in the Potjans paper)
    syn_nums = np.array(syn_nums, dtype=int)
    # Compute the indegree as syn_num / Npost:
    K = np.array(np.diag(np.array([ 1./NE, 1./NI, 1./NR ]).dot(syn_nums)), 
        dtype=int)

    # print"Building network")
    startbuild = time.time()
    nest.SetDefaults("iaf_psc_exp", neuron_params)
    nest.SetDefaults("gif2_psc_exp", neuron_params2)
    nodes_ex = nest.Create("iaf_psc_exp", NE)
    nodes_re = nest.Create("gif2_psc_exp", NR)
    nodes_in = nest.Create("iaf_psc_exp", NI)
    noise = nest.Create("sinusoidal_poisson_generator", 3)
    espikes = nest.Create("spike_detector")
    rspikes = nest.Create("spike_detector")
    ispikes = nest.Create("spike_detector")

    # set the drive levels for all populations
    nest.SetStatus([ noise[ 0 ] ],
        {"rate":      bg_rate,
         "amplitude": 0.025 * 0.0, "frequency": 10.0, "phase": 0.0} )
    nest.SetStatus([ noise[ 1 ] ],
        {"rate":      bg_rate,
         "amplitude": 0.025 * 0.0, "frequency": 10.0, "phase": 0.0} )
    nest.SetStatus([ noise[ 2 ] ],
        {"rate":      bg_rate,
         "amplitude": 0.025 * 0.0, "frequency": 10.0, "phase": 0.0} )

    nest.SetStatus(espikes, [
        {"label":   "brunel-py-ex", "withtime": True, "withgid": True,
         "to_file": False, 'start': recstart} ])
    nest.SetStatus(rspikes, [
        {"label":   "brunel-py-res", "withtime": True, "withgid": True,
         "to_file": False, 'start': recstart} ])
    nest.SetStatus(ispikes, [
        {"label":   "brunel-py-in", "withtime": True, "withgid": True,
         "to_file": False, 'start': recstart} ])

    # try, in case we loop over the function, not the script
    try:
        nest.CopyModel("static_synapse", "excite",
                       {"weight": J_ex * synweight, "delay": delay})
    except:
        pass
    
    nest.Connect([ noise[ 0 ] ], nodes_ex,
                 conn_spec={'rule': 'fixed_indegree', 'indegree': K_ext[ 'E' ]}, 
                 # 'rule': 'all_to_all'},
                 syn_spec={'weight': J_ex, "delay": delay_ex})
    nest.Connect([ noise[ 1 ] ], nodes_in,
                 conn_spec={'rule': 'fixed_indegree', 'indegree': K_ext[ 'I' ]}, 
                 # conn_spec={'rule': 'all_to_all'},
                 syn_spec={'weight': J_ex, "delay": delay_ex})
    nest.Connect([ noise[ 2 ] ], nodes_re,
                 conn_spec={'rule': 'fixed_indegree', 'indegree': K_ext[ 'R' ]}, 
                 # conn_spec={'rule': 'all_to_all'},
                 syn_spec={'weight': J_ex, "delay": delay_ex})

    nest.Connect(nodes_ex[ 0:N_rec[ 'NE' ] ], espikes, syn_spec="excite")
    nest.Connect(nodes_re[ 0:N_rec[ 'NR' ] ], rspikes, syn_spec="excite")
    nest.Connect(nodes_in[ 0:N_rec[ 'NI' ] ], ispikes, syn_spec="excite")

    nest.Connect(nodes_ex, nodes_ex,
                 conn_spec={'rule': 'fixed_indegree', 'indegree': K[ 0, 0 ]},
                 syn_spec={'weight': J[ 0, 0 ], "delay": delay_ex})
    nest.Connect(nodes_ex, nodes_re,
                 conn_spec={'rule': 'fixed_indegree', 'indegree': K[ 0, 1 ]},
                 syn_spec={'weight': J[ 0, 1 ], "delay": delay_ex})
    nest.Connect(nodes_ex, nodes_in,
                 conn_spec={'rule': 'fixed_indegree', 'indegree': K[ 0, 2 ]},
                 syn_spec={'weight': J[ 0, 2 ], "delay": delay_ex})
    # print"Resonating connections")
    nest.Connect(nodes_re, nodes_ex,
                 conn_spec={'rule': 'fixed_indegree', 'indegree': K[ 1, 0 ]},
                 syn_spec={'weight': J[ 1, 0 ], "delay": delay_ex})
    nest.Connect(nodes_re, nodes_re,
                 conn_spec={'rule': 'fixed_indegree', 'indegree': K[ 1, 1 ]},
                 syn_spec={'weight': J[ 1, 1 ], "delay": delay_ex})
    nest.Connect(nodes_re, nodes_in,
                 conn_spec={'rule': 'fixed_indegree', 'indegree': K[ 1, 2 ]},
                 syn_spec={'weight': J[ 1, 2 ], "delay": delay_ex})
    # print"Inhibitory connections")
    nest.Connect(nodes_in, nodes_ex,
                 conn_spec={'rule': 'fixed_indegree', 'indegree': K[ 2, 0 ]},
                 syn_spec={'weight': J[ 2, 0 ], "delay": delay_in})
    nest.Connect(nodes_in, nodes_re,
                 conn_spec={'rule': 'fixed_indegree', 'indegree': K[ 2, 1 ]},
                 syn_spec={'weight': J[ 2, 1 ], "delay": delay_in})
    nest.Connect(nodes_in, nodes_in,
                 conn_spec={'rule': 'fixed_indegree', 'indegree': K[ 2, 2 ]},
                 syn_spec={'weight': J[ 2, 2 ], "delay": delay_in})

    endbuild = time.time()

    # print"Simulating")
    nest.Simulate(simtime + recstart)
    endsimulate = time.time()

    # print'Computing results')
    events_ex = nest.GetStatus(espikes, "events")[ 0 ]
    events_re = nest.GetStatus(rspikes, "events")[ 0 ]
    events_in = nest.GetStatus(ispikes, "events")[ 0 ]
    nevents_ex = nest.GetStatus(espikes, "n_events")[ 0 ]
    nevents_re = nest.GetStatus(rspikes, "n_events")[ 0 ]
    nevents_in = nest.GetStatus(ispikes, "n_events")[ 0 ]
    rate_ex = nevents_ex / simtime * 1000.0 / N_rec[ 'NE' ]
    rate_re = nevents_re / simtime * 1000.0 / N_rec[ 'NR' ]
    rate_in = nevents_in / simtime * 1000.0 / N_rec[ 'NI' ]

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
    cv_ex = np.nanmean(
            [ cv(isi(spiketrain)) for spiketrain in spiketrains_ex ])
    cv_re = np.nanmean(
            [ cv(isi(spiketrain)) for spiketrain in spiketrains_re ])
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
                         markersize=3)
        for spiketrain in spiketrains_re:
            if np.any(spiketrain):
                offset += 1
                figu.add_subplot(1, 1, 1)
                plt.plot(spiketrain - recstart,
                         offset * np.ones_like(spiketrain),
                         'g.',
                         markersize=3)
        for spiketrain in spiketrains_in:
            if np.any(spiketrain):
                offset += 1
                figu.add_subplot(1, 1, 1)
                plt.plot(spiketrain - recstart,
                         offset * np.ones_like(spiketrain),
                         'r.',
                         markersize=3)
        plt.ylim(0, offset)
        plt.xlim(0, simtime)
        plt.savefig('driven_plots/Rasterplot_{2}_{0}_{1}.png'.format(
                int(fraction) * 10, 10, int(p_rate)))
    return [ fraction, rate_ex, rate_in, rate_re, cv_ex, cv_in, cv_re, cv_all,
             recstart, simtime ], [ spiketrains_all, spiketrains_ex,
                                    spiketrains_in, spiketrains_re, espikes,
                                    ispikes, rspikes ]


if __name__ == '__main__':
    configure_nest_kernel()
    loop = 2
    try:
        simulation_index = int(sys.argv[ 3 ])
        simarray_index = int(sys.argv[ 2 ])
        path = 'brunel_results/'
        if not os.path.isfile(path + 'brunel_array_results_{'
                              '0}.csv'.format(simulation_index)):
            shutil.copy2('brunel_results/brunel_array_results_.csv',
                         path + 'brunel_array_results_{0}.csv'.format(
                                 simulation_index))
        if not os.path.isfile(path + 'brunel_array_results_{'
                              '0}_0.csv'.format(simulation_index)):
            shutil.copy2('brunel_results/brunel_array_results_.csv',
                         path + 'brunel_array_results_{0}_0.csv'.format(
                                 simulation_index))


        fractionindex = int(sys.argv[ 1 ])
        fraction = np.arange(0.0, 10.0)[ fractionindex + 1 ] / 10.0
    except IndexError:
        fraction = 0.5
        loop = False
        writeoutflag = False

    if loop == 0:
        for i in np.arange(63000.0, 70500.0, 750.0):
            networkparamdict = {'p_rate':  i, 'C_m': 250.0, 'g': 25.0,
                                'g_1':     60.0, 'tau_1': 92.5,
                                'V_dist':  4.025, 'V_dist2': 1.15,
                                'K_ext':   {'E': 2000, 'I': 1900, 'R': 2000},
                                'bg_rate': 8., 'fraction': fraction}

            resultlist, spikelists = run_brunel(networkparamdict)
            resultarray = construct_resultarray(resultlist, networkparamdict)
            writeout(simulation_index, resultarray, path=path)
    elif loop == 2:
        networkparamdict = {'p_rate':  0, 'C_m': 250.0, 'g': 25.0,
                            'g_1':     60.0, 'tau_1': 92.5,
                            'V_dist':  4.025, 'V_dist2': 1.15,
                            'K_ext':   {'E': 2000, 'I': 1900, 'R': 2000},
                            'bg_rate': 8., 'fraction': fraction}

        # simulate once for maximal dV2
        resultlist, spikelists = run_brunel(networkparamdict, plotting=False)
        # list everything of interest
        resultarray = construct_resultarray(resultlist, networkparamdict)
        # write it down
        if writeoutflag:
            writeout(simulation_index, resultarray, path=path)

    elif loop == 3:
        writeoutflag = True
        execfile('better_configuration_finder.py')

        print('Importing complete.')
        configure_nest_kernel()
        nest.SetKernelStatus({'print_time': True})

        print('Nest Kernel configures')
        networkdict = convert_to_paramdict(testconfigs1[0,:])

        print('Chosen network parameters:')
        for i, j in networkdict.iteritems():
            print('{0} = {1}'.format(i,j))
        print('\n')

        print('Simulating')
        resultlist, spikelists = run_brunel(networkdict)

        print('Done. Congregating results. \n')
        resultarray = construct_resultarray(resultlist, networkdict)

        print(resultarray)