"""
Version 9.5
Option
alternative_connection = False

This version omits the fourth spike detector for the rasterplot and extracts
all data from the three spike detectors.

With p_rate2 = 85000. and h = 1.5, the firing rates of the pops are equal.
With p_rate2 = 70000. and h = 2.5, the firing rates of the pops are equal.
With p_rate2 = 66000. and h = 1.9, the firing rates of the pops are equal.
"""

import numpy as np
import time
import sys
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

	recstart = 5000.0
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
	N_rec = {'NE': int(NE/10), 'NI': int(NI/10), 'NR': int(NR/10)}

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
	N_full_new = {'L5': {'E': 2425, 'I': 1065, 'R': 2425}}
	N_full_old = {'L5': {'E': 4050, 'I': 1065}}
	K_bg = [ [ 2000, 1900 ] ]
	C = compute_new_connectivity(conn_probs, N_full_new, N_full_old,
								 layers, pops, newpopdict, new_structure)

	C[ 0, : ] *= NE
	C[ 1, : ] *= NR
	C[ 2, : ] *= NI
	C = np.array(C, dtype=int)
	print(C)

	print("Building network")
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

	print("Connecting devices")
	nest.CopyModel("static_synapse", "excitatory",
				   {"weight": J_ex * synweight, "delay": delay})

	nest.Connect(noise, nodes_ex, syn_spec="excitatory")
	nest.Connect(noise, nodes_in, syn_spec="excitatory")
	nest.Connect(noise, nodes_re, syn_spec="excitatory")
	nest.Connect(nodes_ex[ 0:N_rec[ 'NE' ] ], espikes, syn_spec="excitatory")
	nest.Connect(nodes_re[ 0:N_rec[ 'NR' ] ], rspikes, syn_spec="excitatory")
	nest.Connect(nodes_in[ 0:N_rec[ 'NI' ] ], ispikes, syn_spec="excitatory")

	print("Connecting network")
	print("Excitatory connections")
	nest.Connect(nodes_ex, nodes_ex,
				 conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 0 ]},
				 syn_spec={'weight': J[ 0, 0 ], "delay": delay_ex})
	nest.Connect(nodes_ex, nodes_re,
				 conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 1 ]},
				 syn_spec={'weight': J[ 0, 1 ], "delay": delay_ex})
	nest.Connect(nodes_ex, nodes_in,
				 conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 2 ]},
				 syn_spec={'weight': J[ 0, 2 ], "delay": delay_ex})
	print("Resonating connections")
	nest.Connect(nodes_re, nodes_ex,
				 conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 0 ]},
				 syn_spec={'weight': J[ 1, 0 ], "delay": delay_ex})
	nest.Connect(nodes_re, nodes_re,
				 conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 1 ]},
				 syn_spec={'weight': J[ 1, 1 ], "delay": delay_ex})
	nest.Connect(nodes_re, nodes_in,
				 conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 2 ]},
				 syn_spec={'weight': J[ 1, 2 ], "delay": delay_ex})
	print("Inhibitory connections")
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

	print("Simulating")
	nest.Simulate(simtime + recstart)
	endsimulate = time.time()

	print('Computing results')
	events_ex = nest.GetStatus(espikes, "events")[ 0 ]
	events_re = nest.GetStatus(rspikes, "events")[ 0 ]
	events_in = nest.GetStatus(ispikes, "events")[ 0 ]
	nevents_ex = nest.GetStatus(espikes, "n_events")[ 0 ]
	nevents_re = nest.GetStatus(rspikes, "n_events")[ 0 ]
	nevents_in = nest.GetStatus(ispikes, "n_events")[ 0 ]
	rate_ex = nevents_ex / simtime * 1000.0 / N_rec[ 'NE' ]
	rate_re = nevents_re / simtime * 1000.0 / N_rec[ 'NR' ]
	rate_in = nevents_in / simtime * 1000.0 / N_rec[ 'NI' ]
	print('Done. \n')

	build_time = endbuild - startbuild
	sim_time = endsimulate - endbuild

	print("Brunel network simulation (Python)")
	print("Number of neurons : {0}".format(N_neurons))
	print("Excitatory rate   : %.2f Sp/s" % rate_ex)
	print("Resonating rate   : %.2f Sp/s" % rate_re)
	print("Inhibitory rate   : %.2f Sp/s" % rate_in)
	print("Building time     : %.2f s" % build_time)
	print("Simulation time   : %.2f s" % sim_time)

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
	cv_ex = np.mean([ cv(isi(spiketrain)) for spiketrain in spiketrains_ex ])
	cv_re = np.mean([ cv(isi(spiketrain)) for spiketrain in spiketrains_re ])
	cv_in = np.mean([ cv(isi(spiketrain)) for spiketrain in spiketrains_in ])
	cv_allex = np.mean([ cv(isi(spiketrain))
						 for spiketrain in spiketrains_allex ])
	cv_all = np.mean([ cv(isi(spiketrain)) for spiketrain in spiketrains_all ])
	print('mean CV for sim: {0}'.format(cv_all))
	print('CVex {0}, CVre {1}, CVin {2}, CVexall{3}'.format(
			cv_ex, cv_re, cv_in, cv_allex))
	print('resonating spikes: {0}'.format(nevents_re))
	print('expected resonance frequency : {0}'.format(predict_str_freq(
			tau_1, gm, g1, C_m, remote=True)))
	print('C = {0}, g = {2}, g_1 = {1}, t_1 = {4}, p = {3}, '.format(
			C_m, g1, gm, p_rate, tau_1))

	print('Creating plot')
	plt.clf()
	# Raster:
	figu = plt.figure("Rasterplot")
	j = 0
	for spiketrain in spiketrains_ex:
		if np.any(spiketrain):
			j += 1
			figu.add_subplot(1, 1, 1)
			plt.plot(spiketrain - recstart, j * np.ones_like(spiketrain), 'b.',
					 markersize=3)
	for spiketrain in spiketrains_re:
		if np.any(spiketrain):
			j += 1
			figu.add_subplot(1, 1, 1)
			plt.plot(spiketrain - recstart, j * np.ones_like(spiketrain), 'g.',
					 markersize=3)
	for spiketrain in spiketrains_in:
		if np.any(spiketrain):
			j += 1
			figu.add_subplot(1, 1, 1)
			plt.plot(spiketrain - recstart, j * np.ones_like(spiketrain), 'r.',
					 markersize=3)
	plt.ylim(0, j)
	plt.xlim(0, simtime)
	plt.savefig('Rasterplot_{0}_{1}.png'.format(fraction * 20, 20))
	return [ fraction, rate_ex, rate_in, rate_re, cv_ex, cv_in, cv_re,
			 recstart, simtime ], [ spiketrains_all, spiketrains_ex,
									spiketrains_in, spiketrains_re, espikes,
									ispikes, rspikes ]


if __name__ == '__main__':
	dt = 0.1
	nest.set_verbosity('M_WARNING')
	nest.ResetKernel()
	nest.SetKernelStatus(
			{"resolution": dt, "print_time": False, "overwrite_files": True})
	try:
		nest.Install("gif2_module")
	except:
		pass

	# p_rate, C_m, gm, g1, tau_1, V_dist
	g_range = np.arange(10.0, 85.0, 2.5)
	g1_range = np.arange(10.0, 85.0, 2.5)
	t1_range = np.arange(80.0, 100.0, 10.0)
	dV1_range = np.arange(5.0, 10.0, 0.5)
	dV2_range = np.arange(0.0, 20.0, 1.0)

	networkparamdict = {'p_rate': 65000.0, 'C_m': 250.0, 'g': 25.0,
						'g_1':    77.5, 'tau_1': 100.0, 'V_dist': 5.5,
						'V_dist2': 14.5}
	# len(g_range) * len(g1_range) * len(t1_range)
	j = 0
	for i in product(g_range, g1_range, t1_range):
		j += 1
		if int(sys.argv[ 3 ]) == j:
			if np.isclose(predict_str_freq(i[2], i[0], i[1], 250.0), 10.0,
						  atol=0.1, rtol=0.0):
				for j in product(dV1_range, dV2_range):
					networkparamdict[ 'g' ] = i[ 0 ]
					networkparamdict[ 'g_1' ] = i[ 1 ]
					networkparamdict[ 'tau_1' ] = i[ 2 ]
					networkparamdict[ 'V_dist' ] = j[ 0 ]
					networkparamdict[ 'V_dist2' ] = j[ 1 ]

					fractionindex = int(sys.argv[ 1 ])
					fraction = np.arange(0.0, 20.0)[ fractionindex + 1 ] / 20.0
					resultlist, spikelists = run_brunel(networkparamdict, fraction)
					resultarray = np.array(resultlist)
					with open('brunel_array_results_1.csv', 'a') as output:
						np.savetxt(output, resultarray, fmt="%12.6G", newline=' ')
						output.write(' \n')
						output.close()

					paramlist = [ networkparamdict['p_rate'], networkparamdict['C_m'],
								  networkparamdict['g'], networkparamdict['g_1'],
								  networkparamdict['tau_1'], networkparamdict['V_dist'],
								  networkparamdict['V_dist2'] ]
					with open('brunel_array_params_1.csv', 'a') as output:
						output.write(str(paramlist) + ' \n')
						output.close()
