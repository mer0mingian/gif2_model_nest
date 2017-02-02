"""
Version 9.5

The idea of this script is to check, what distance between resting and reset
potential most adequately copies the firing rate behaviour of a LIF neuron
with equal membrance capacity, conductance and time constant.

To run this script, use gif2brunel_array3.sh. Before sbatching the shell script
use looptester.py with the current configuration and adapt the array length in
the shell script accordingly.

The improvement over v3 of the script is that we do not claim that C and tau
need to be equal to the LIF parameters, but only tau. This should open up the
domain, where g and g1 are roughly equal and around 70nS for C=250pF.
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
from itertools import product
from mingtools1 import *
from elephant.statistics import isi, cv
from gif2_brunel_f import *
from mc_connectivity_transformer import compute_new_connectivity


# Kernel parameters:

def writeout(simulation_index, resultarray):
    if resultarray[ 10 ] > 0.0:  # if resonance occurs
        with open('brunel_array_results_{0}.csv'.format(
                simulation_index), 'a') as output:
            np.savetxt(output, resultarray, fmt="%12.6G",
                       newline=' ')
            output.write(' \n')
            output.close()
    else:
        with open('brunel_array_results_{0}_0.csv'.format(
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


# def run_brunel(networkparamdict, fraction):
#     p_rate = networkparamdict[ 'p_rate' ]
#     C_m = networkparamdict[ 'C_m' ]
#     gm = networkparamdict[ 'g' ]
#     g1 = networkparamdict[ 'g_1' ]
#     tau_1 = networkparamdict[ 'tau_1' ]
#     V_dist = networkparamdict[ 'V_dist' ]
#     Vdist2 = networkparamdict[ 'V_dist2' ]

#     recstart = 7500.0
#     simtime = 10000.0  # Simulation time in ms
#     delay = 0.8  # synaptic delay in ms
#     delay_ex = 1.5
#     delay_in = 0.8
#     g = 5.0  # ratio inhibitory weight/excitatory weight

#     N = 4850
#     NE = int((1.0 - fraction) * N)
#     NR = int(fraction * N)
#     NI = 1065
#     N_neurons = NE + NI + NR  # number of neurons in total
#     # record from N_rec neurons per population:
#     N_rec = {'NE': int(NE / 10), 'NI': int(NI / 10), 'NR': int(NR / 10)}

#     theta = -50.0  # membrane threshold potential in mVfrom stats
#     tauSyn = 0.5

#     neuron_params = {
#         "C_m":        250.0,
#         "tau_m":      10.0,
#         "tau_syn_ex": tauSyn,
#         "tau_syn_in": tauSyn,
#         "t_ref":      2.0,
#         "E_L":        -65.0,
#         "V_reset":    -65.0,
#         "V_m":        -65.0,
#         "V_th":       theta}

#     neuron_params2 = {
#         "tau_1":      tau_1,
#         "C_m":        C_m,
#         "tau_syn_ex": tauSyn,
#         "tau_syn_in": tauSyn,
#         "g_rr":       g1,
#         "g":          gm,
#         "V_m":        theta - V_dist,
#         "V_reset":    theta - V_dist,
#         "E_L":        theta - V_dist - Vdist2,
#         "V_th":       theta,
#         "t_ref":      2.0}

#     synweight = 87.8
#     J = 0.125  # postsynaptic amplitude in mV
#     J_ex = J  # amplitude of excitatory postsynaptic potential
#     J_in = -g * J

#     J = np.array([ [ J_ex, J_ex, J_ex ],
#                    [ J_ex, J_ex, J_ex ],
#                    [ J_in, J_in, J_in ] ])

#     # get the new connectivity:
#     conn_probs = [ [ 0.0831, 0.3726 ], [ 0.060, 0.3158 ] ]
#     layers = {'L5': 0}
#     pops = {'E': 0, 'I': 1}
#     newpopdict = {'E': 0, 'I': 1, 'R': 2}
#     new_structure = {'L5': {'E': 0, 'I': 1, 'R': 2}}
#     N_full_new = {'L5': {'E': NE, 'I': 1065, 'R': NR}}
#     N_full_old = {'L5': {'E': 4050, 'I': 1065}}
#     K_bg = [ [ 2000, 1900 ] ]
#     C = compute_new_connectivity(conn_probs, N_full_new, N_full_old,
#                                  layers, pops, newpopdict, new_structure)

#     C[ 0, : ] *= NE
#     C[ 1, : ] *= NR
#     C[ 2, : ] *= NI
#     C = np.array(C, dtype=int)
#     N_pre = np.array([ NE, 1065, NR ], dtype=int)
#     N_pp = np.outer(N_pre, N_pre)
#     # C_a = TODO!!
#     # print(C)

#     # print"Building network")
#     startbuild = time.time()
#     nest.SetDefaults("iaf_psc_exp", neuron_params)
#     nest.SetDefaults("gif2_psc_exp", neuron_params2)
#     nodes_ex = nest.Create("iaf_psc_exp", NE)
#     nodes_re = nest.Create("gif2_psc_exp", NR)
#     nodes_in = nest.Create("iaf_psc_exp", NI)
#     noise = nest.Create("sinusoidal_poisson_generator")
#     espikes = nest.Create("spike_detector")
#     rspikes = nest.Create("spike_detector")
#     ispikes = nest.Create("spike_detector")

#     nest.SetStatus(noise, [
#         {"rate":  p_rate, "amplitude": 0.025 * 0.0, "frequency": 10.0,
#          "phase": 0.0} ])
#     nest.SetStatus(espikes, [
#         {"label":   "brunel-py-ex", "withtime": True, "withgid": True,
#          "to_file": False, 'start': recstart} ])
#     nest.SetStatus(rspikes, [
#         {"label":   "brunel-py-res", "withtime": True, "withgid": True,
#          "to_file": False, 'start': recstart} ])
#     nest.SetStatus(ispikes, [
#         {"label":   "brunel-py-in", "withtime": True, "withgid": True,
#          "to_file": False, 'start': recstart} ])

#     # nest cannot does not overwrite the synapse model
#     try:
#         nest.CopyModel("static_synapse", "excite",
#                        {"weight": J_ex * synweight, "delay": delay})
#     except:
#         pass

#     nest.Connect(noise, nodes_ex, syn_spec="excite")
#     nest.Connect(noise, nodes_in, syn_spec="excite")
#     nest.Connect(noise, nodes_re, syn_spec="excite")
#     nest.Connect(nodes_ex[ 0:N_rec[ 'NE' ] ], espikes, syn_spec="excite")
#     nest.Connect(nodes_re[ 0:N_rec[ 'NR' ] ], rspikes, syn_spec="excite")
#     nest.Connect(nodes_in[ 0:N_rec[ 'NI' ] ], ispikes, syn_spec="excite")

#     nest.Connect(nodes_ex, nodes_ex,
#                  conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 0 ]},
#                  syn_spec={'weight': J[ 0, 0 ], "delay": delay_ex})
#     nest.Connect(nodes_ex, nodes_re,
#                  conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 1 ]},
#                  syn_spec={'weight': J[ 0, 1 ], "delay": delay_ex})
#     nest.Connect(nodes_ex, nodes_in,
#                  conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 2 ]},
#                  syn_spec={'weight': J[ 0, 2 ], "delay": delay_ex})
#     nest.Connect(nodes_re, nodes_ex,
#                  conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 0 ]},
#                  syn_spec={'weight': J[ 1, 0 ], "delay": delay_ex})
#     nest.Connect(nodes_re, nodes_re,
#                  conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 1 ]},
#                  syn_spec={'weight': J[ 1, 1 ], "delay": delay_ex})
#     nest.Connect(nodes_re, nodes_in,
#                  conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 2 ]},
#                  syn_spec={'weight': J[ 1, 2 ], "delay": delay_ex})
#     nest.Connect(nodes_in, nodes_ex,
#                  conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 2, 0 ]},
#                  syn_spec={'weight': J[ 2, 0 ], "delay": delay_in})
#     nest.Connect(nodes_in, nodes_re,
#                  conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 2, 1 ]},
#                  syn_spec={'weight': J[ 2, 1 ], "delay": delay_in})
#     nest.Connect(nodes_in, nodes_in,
#                  conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 2, 2 ]},
#                  syn_spec={'weight': J[ 2, 2 ], "delay": delay_in})

#     endbuild = time.time()

#     nest.Simulate(simtime + recstart)
#     endsimulate = time.time()

#     events_ex = nest.GetStatus(espikes, "events")[ 0 ]
#     events_re = nest.GetStatus(rspikes, "events")[ 0 ]
#     events_in = nest.GetStatus(ispikes, "events")[ 0 ]
#     nevents_ex = nest.GetStatus(espikes, "n_events")[ 0 ]
#     nevents_re = nest.GetStatus(rspikes, "n_events")[ 0 ]
#     nevents_in = nest.GetStatus(ispikes, "n_events")[ 0 ]
#     rate_ex = nevents_ex / simtime * 1000.0 / N_rec[ 'NE' ]
#     rate_re = nevents_re / simtime * 1000.0 / N_rec[ 'NR' ]
#     rate_in = nevents_in / simtime * 1000.0 / N_rec[ 'NI' ]

#     # CVs:
#     spiketrains_ex = list()
#     spiketrains_in = list()
#     spiketrains_re = list()
#     for gid in nodes_ex:
#         spiketrains_ex.append(
#                 events_ex[ 'times' ][ events_ex[ 'senders' ] == gid ])
#     for gid in nodes_re:
#         spiketrains_re.append(
#                 events_re[ 'times' ][ events_re[ 'senders' ] == gid ])
#     for gid in nodes_in:
#         spiketrains_in.append(
#                 events_in[ 'times' ][ events_in[ 'senders' ] == gid ])
#     spiketrains_allex = spiketrains_ex + spiketrains_re
#     spiketrains_all = spiketrains_allex + spiketrains_in
#     cv_ex = np.nanmean(
#             [ cv(isi(spiketrain)) for spiketrain in spiketrains_ex ])
#     cv_re = np.nanmean(
#             [ cv(isi(spiketrain)) for spiketrain in spiketrains_re ])
#     cv_in = np.nanmean(
#             [ cv(isi(spiketrain)) for spiketrain in spiketrains_in ])
#     cv_all = np.nanmean(
#             [ cv(isi(spiketrain)) for spiketrain in spiketrains_all ])

#     return [ fraction, rate_ex, rate_in, rate_re, cv_ex, cv_in, cv_re, cv_all,
#              recstart, simtime ], [ spiketrains_all, spiketrains_ex,
#                                     spiketrains_in, spiketrains_re, espikes,
#                                     ispikes, rspikes ]


if __name__ == '__main__':
    simulation_index = int(sys.argv[ 3 ])
    simarray_index = int(sys.argv[ 2 ])
    if not os.path.isfile('brunel_results/brunel_array_results_{'
                          '0}.csv'.format(simulation_index)):
        shutil.copy2('brunel_results/brunel_array_results_.csv',
                     'brunel_results/brunel_array_results_{0}.csv'.format(
                             simulation_index))
    if not os.path.isfile('brunel_results/brunel_array_results_{'
                          '0}_0.csv'.format(simulation_index)):
        shutil.copy2('brunel_results/brunel_array_results_.csv',
                     'brunel_results/brunel_array_results_{0}_0.csv'.format(
                             simulation_index))

    # print(simulation_index)
    dt = 0.1
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.SetKernelStatus(
            {"resolution":        dt,
             "print_time": 		  False,
             "overwrite_files":   True,
             "local_num_threads": 16})
    try:
        nest.Install("gif2_module")
    except:
        pass
    os.chdir(
        '/mnt/beegfs/home/d.mingers/gif2_model_nest/brunel/brunel_results')

    fractionindex = int(sys.argv[ 1 ])
    fraction = np.arange(0.0, 20.0)[ fractionindex + 1 ] / 20.0

    networkparamdict = {'p_rate': float(sys.argv[ 4 ])}
    C_range = np.arange(250.0, 550.0, 10.0)
    g_range = np.arange(5.0, 80.0, 2.5)
    dV1_range = np.arange(3.0, 9.5, 0.1)

    k = 0
    # C
    for C in C_range:
        # g
        g = C / 10.0
        # g1
        g1_range = np.arange(g, 80.0, 2.5)
        # dV1
        for i in product(g1_range, dV1_range):
            dV2_max = 20.0 - i[ 1 ]
            # t1
            t1_range = np.arange(50.0, 120.0, 10.0)
            for t1 in t1_range:
                if np.isclose(
                        predict_str_freq(t1, g, i[ 0 ], C, remote=True),
                        10.0, atol=0.05, rtol=0.0):
                    k += 1
                    if np.all([ simarray_index >= k * 10,
                                simarray_index < (k + 1) * 10]):
                        networkparamdict[ 'C_m' ] = C
                        networkparamdict[ 'g' ] = g
                        networkparamdict[ 'g_1' ] = i[ 0 ]
                        networkparamdict[ 'V_dist' ] = i[ 1 ]
                        networkparamdict[ 'V_dist2' ] = dV2_max  # i[ 2 ]
                        networkparamdict[ 'tau_1' ] = t1
                        networkparamdict[ 'faction' ] = fraction

                        # simulate once for maximal dV2
                        resultlist, spikelists = run_brunel(networkparamdict, 
                            plotting=False)
                        # list everything of interest
                        resultarray = construct_resultarray(resultlist,
                                                            networkparamdict)
                        # write it down
                        writeout(simulation_index, resultarray)
                        # most right point for scanning dV2
                        resrate_upper = resultarray[ 10 ]
                        dV2_upper = networkparamdict[ 'V_dist2' ]

                        # simulate once for minimal dV2
                        # distance is closest, firing the highest
                        networkparamdict[ 'V_dist2' ] = 0.0
                        resultlist, spikelists = run_brunel(networkparamdict,
                                                            plotting=False)
                        resultarray = construct_resultarray(resultlist,
                                                            networkparamdict)

                        writeout(simulation_index, resultarray)
                        resrate_lower = resultarray[ 10 ]
                        dV2_lower = networkparamdict[ 'V_dist2' ]

                        # picard iteration for dV2 to accelerate scanning
                        counter = 0
                        while True:
                            counter += 1
                            if counter >= 25:
                                print('ran out of trials')
                                break
                            # take the step in the middle
                            networkparamdict[ 'V_dist2' ] = (dV2_upper + dV2_lower) / 2.0
                            nest.ResetKernel()
                            nest.SetKernelStatus(
                                    {"resolution":        dt,
                                     "print_time":        False,
                                     "overwrite_files":   True,
                                     "local_num_threads": 16})
                            resultlist, spikelists = run_brunel(
                                    networkparamdict, fraction)
                            resultarray = construct_resultarray(resultlist, networkparamdict)
                            writeout(simulation_index, resultarray)
                            # our new rate
                            resrate_new = resultarray[ 10 ]
                            # find the next value for dV2:
                            if np.isclose(resrate_new, resultarray[ 8 ],
                                          atol=0.1, rtol=0.0):
                                break
                            if resrate_new - resultarray[ 8 ] > 0.15:
                                # if resrate is higher than exrate:
                                # move lower boundary to the right =>
                                # dV2 greater => resrate lower
                                dV2_lower = networkparamdict[ 'V_dist2' ]
                            else:
                                dV2_upper = networkparamdict[ 'V_dist2' ]

# A seriously well working configuration:
"""
                        networkparamdict = {'p_rate':  65000.0, 'C_m': 250.0,
                                            'g':       10.0, 'g_1': 40.0,
                                            'tau_1':   50.0, 'V_dist': 6.0,
                                            'V_dist2': 0.0}

       65000          250           25           60           70
       4.75        0.525          0.5      5.37934      5.38208      5.38017
           0.730388     0.729597      2.27803      1.36504         7500
             10000
"""
# Make the C_range more finegrained and instead loop over a range of k's
