"""
Version 9.5

The idea of this script is to check, what distance between resting and reset
potential most adequately copies the firing rate behaviour of a LIF neuron
with equal membrance capacity, conductance and time constant.

To run this script, use gif2brunel_array3.sh. Before sbatching the shell script
use looptester.py with the current configuration and adapt the array length in
the shell script accordingly.
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
from mc_connectivity_transformer import compute_new_connectivity


# Kernel parameters:

def run_brunel(networkparamdict, fraction):
	p_rate = networkparamdict[ 'p_rate' ]
	C_m = networkparamdict[ 'C_m' ]
	gm = networkparamdict[ 'g' ]
	g1 = networkparamdict[ 'g_1' ]
	tau_1 = networkparamdict[ 'tau_1' ]
	V_dist = networkparamdict[ 'V_dist' ]
	Vdist2 = networkparamdict[ 'V_dist2' ]

	recstart = 7500.0
	simtime = 10000.0  # Simulation time in ms
	delay = 1.0  # synaptic delay in ms
	delay_ex = 1.5
	delay_in = 0.8
	g = 5.0  # ratio inhibitory weight/excitatory weight

	N = 4850
	NE = int((1.0 - fraction) * N)
	NR = int(fraction * N)
	NI = 1065
	N_neurons = NE + NI + NR  # number of neurons in total
	# record from N_rec neurons per population:
	N_rec = {'NE': int(NE / 10), 'NI': int(NI / 10), 'NR': int(NR / 10)}

	theta = -50.0  # membrane threshold potential in mVfrom stats
	tauSyn = 0.5

	neuron_params = {
		"C_m":        250.0,
		"tau_m":      10.0,
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
	J = 0.125  # postsynaptic amplitude in mV
	J_ex = J  # amplitude of excitatory postsynaptic potential
	J_in = -g * J

	J = np.array([ [ J_ex, J_ex, J_ex ],
				   [ J_ex, J_ex, J_ex ],
				   [ J_in, J_in, J_in ] ])

	# get the new connectivity:
	conn_probs = [ [ 0.0831, 0.3726 ], [ 0.060, 0.3158 ] ]
	layers = {'L5': 0}
	pops = {'E': 0, 'I': 1}
	newpopdict = {'E': 0, 'I': 1, 'R': 2}
	new_structure = {'L5': {'E': 0, 'I': 1, 'R': 2}}
	N_full_new = {'L5': {'E': NE, 'I': 1065, 'R': NR}}
	N_full_old = {'L5': {'E': 4050, 'I': 1065}}
	K_bg = [ [ 2000, 1900 ] ]
	C = compute_new_connectivity(conn_probs, N_full_new, N_full_old,
								 layers, pops, newpopdict, new_structure)

	C[ 0, : ] *= NE
	C[ 1, : ] *= NR
	C[ 2, : ] *= NI
	C = np.array(C, dtype=int)
	N_pre = np.array([ NE, 1065, NR ], dtype=int)
	N_pp = np.outer(N_pre, N_pre)
	# C_a = TODO!!
	# print(C)

	# print"Building network")
	startbuild = time.time()
	nest.SetDefaults("iaf_psc_exp", neuron_params)
	nest.SetDefaults("gif2_psc_exp", neuron_params2)
	nodes_ex = nest.Create("iaf_psc_exp", NE)
	nodes_re = nest.Create("gif2_psc_exp", NR)
	nodes_in = nest.Create("iaf_psc_exp", NI)
	noise = nest.Create("sinusoidal_poisson_generator")
	espikes = nest.Create("spike_detector")
	rspikes = nest.Create("spike_detector")
	ispikes = nest.Create("spike_detector")

	nest.SetStatus(noise, [
		{"rate":  p_rate, "amplitude": 0.025 * 0.0, "frequency": 10.0,
		 "phase": 0.0} ])
	nest.SetStatus(espikes, [
		{"label":   "brunel-py-ex", "withtime": True, "withgid": True,
		 "to_file": False, 'start': recstart} ])
	nest.SetStatus(rspikes, [
		{"label":   "brunel-py-res", "withtime": True, "withgid": True,
		 "to_file": False, 'start': recstart} ])
	nest.SetStatus(ispikes, [
		{"label":   "brunel-py-in", "withtime": True, "withgid": True,
		 "to_file": False, 'start': recstart} ])

	# print"Connecting devices")
	nest.CopyModel("static_synapse", "excitatory",
				   {"weight": J_ex * synweight, "delay": delay})

	nest.Connect(noise, nodes_ex, syn_spec="excitatory")
	nest.Connect(noise, nodes_in, syn_spec="excitatory")
	nest.Connect(noise, nodes_re, syn_spec="excitatory")
	nest.Connect(nodes_ex[ 0:N_rec[ 'NE' ] ], espikes, syn_spec="excitatory")
	nest.Connect(nodes_re[ 0:N_rec[ 'NR' ] ], rspikes, syn_spec="excitatory")
	nest.Connect(nodes_in[ 0:N_rec[ 'NI' ] ], ispikes, syn_spec="excitatory")

	# print"Connecting network")
	# print"Excitatory connections")
	nest.Connect(nodes_ex, nodes_ex,
				 conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 0 ]},
				 syn_spec={'weight': J[ 0, 0 ], "delay": delay_ex})
	nest.Connect(nodes_ex, nodes_re,
				 conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 1 ]},
				 syn_spec={'weight': J[ 0, 1 ], "delay": delay_ex})
	nest.Connect(nodes_ex, nodes_in,
				 conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 2 ]},
				 syn_spec={'weight': J[ 0, 2 ], "delay": delay_ex})
	# print"Resonating connections")
	nest.Connect(nodes_re, nodes_ex,
				 conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 0 ]},
				 syn_spec={'weight': J[ 1, 0 ], "delay": delay_ex})
	nest.Connect(nodes_re, nodes_re,
				 conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 1 ]},
				 syn_spec={'weight': J[ 1, 1 ], "delay": delay_ex})
	nest.Connect(nodes_re, nodes_in,
				 conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 2 ]},
				 syn_spec={'weight': J[ 1, 2 ], "delay": delay_ex})
	# print"Inhibitory connections")
	nest.Connect(nodes_in, nodes_ex,
				 conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 2, 0 ]},
				 syn_spec={'weight': J[ 2, 0 ], "delay": delay_in})
	nest.Connect(nodes_in, nodes_re,
				 conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 2, 1 ]},
				 syn_spec={'weight': J[ 2, 1 ], "delay": delay_in})
	nest.Connect(nodes_in, nodes_in,
				 conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 2, 2 ]},
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

	return [ fraction, rate_ex, rate_in, rate_re, cv_ex, cv_in, cv_re, cv_all,
			 recstart, simtime ], [ spiketrains_all, spiketrains_ex,
									spiketrains_in, spiketrains_re, espikes,
									ispikes, rspikes ]


if __name__ == '__main__':
	simulation_index = int(sys.argv[ 3 ])
	if not os.path.isfile('brunel_results/brunel_array_results_{'
						  '0}.csv'.format(simulation_index)):
		shutil.copy2('brunel_results/brunel_array_results_.csv',
					 'brunel_results/brunel_array_results_{0}.csv'.format(
							 simulation_index))
		shutil.copy2('brunel_results/brunel_array_params_.csv',
					 'brunel_results/brunel_array_params_{0}.csv'.format(
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

	# p_rate, C_m, gm, g1, tau_1, V_dist
	g1_range = np.arange(5.0, 85.0, 2.5)
	dV1_range = np.arange(4.0, 9.5, 0.025)
	dV2_range = np.arange(0.0, 16.0, 0.025)

	networkparamdict = {'p_rate': 65000.0, 'C_m': 250.0, 'g': 25.0}
	# len(g_range) * len(g1_range) * len(t1_range)
	k = 0
	m = 0
	for i in product(g1_range, dV1_range):
		# min = np.amin([ 50.0, i[ 0 ] - 10.0 ])
		t1_range = np.arange(50.0, 120.0, 10.0)
		for j in t1_range:
			if np.isclose(
					predict_str_freq(j, 10.0, i[ 0 ], 250.0, remote=True),
					10.0, atol=0.05, rtol=0.0):
				k += 1
				if int(sys.argv[ 2 ]) == k:
					networkparamdict[ 'g_1' ] = i[ 0 ]
					networkparamdict[ 'V_dist' ] = i[ 1 ]
					networkparamdict[ 'V_dist2' ] = 0.0  # i[ 2 ]
					networkparamdict[ 'tau_1' ] = j

					fractionindex = int(sys.argv[ 1 ])
					fraction = np.arange(0.0, 20.0)[ fractionindex + 1 ] / 20.0
					resultlist, spikelists = run_brunel(networkparamdict,
														fraction)
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

					if resultarray[ 10 ] > 0.0:  # if resonating actvity occurs
						with open('brunel_array_results_{0}.csv'.format(
								simulation_index), 'a') as output:
							np.savetxt(output, resultarray, fmt="%12.6G",
									   newline=' ')
							output.write(' \n')
							output.close()

					for l in dV2_range[ 1: ]:
						if resultarray[ 10 ] > 0.0:  # if we have fired before
							networkparamdict[ 'V_dist2' ] = l

							nest.ResetKernel()
							resultlist, spikelists = run_brunel(
									networkparamdict, fraction)
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

							with open('brunel_array_results_{0}.csv'.format(
									simulation_index),
									'a') as output:
								np.savetxt(output, resultarray, fmt="%12.6G",
										   newline=' ')
								output.write(' \n')
								output.close()
						else:  # if the threshold is so far away, that we
							# don't fire anymore
							break

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
