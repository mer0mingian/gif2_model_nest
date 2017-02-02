# coding=utf-8
"""
This file collects all currently relevant functions for Simulating the
isolated L5 from the Microcircuit, including the resonating population,
as a Brunel network to find a reasonable parameter configuration for
the GIF2 model for application in the multi-area model.
"""

import os
import numpy as np
import time
import sys
import shutil
import matplotlib as mpl
cluster = False
if cluster:
	mpl.use('Agg')
import matplotlib.pyplot as plt
import nest
from mingtools1 import *
from elephant.statistics import isi, cv
from mc_connectivity_transformer import compute_new_connectivity
from itertools import product
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


# ****************************************************************************************************************************
# BRUNEL NETWORK HELPER FUNCTIONS
# ****************************************************************************************************************************


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

def configure_nest_kernel(cluter=False):
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
	# t1
			t1_range = np.arange(50.0, 120.0, 10.0)
			for j in t1_range:
				if np.isclose(
						predict_str_freq(j, g, i[ 0 ], C, remote=True),
						10.0, atol=0.05, rtol=0.0):
					k += 1
	print(k)

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
	    mean_exin <= exin_rates_upper_bound, mean_exin >=exin_rates_lower_bound), axis=0)

	# is R rate also acceptable?
	dist_r_exin = np.abs(data[:,10] - mean_exin)
	dist_r_exin_accept = np.abs(dist_r_exin) <= distance_r_to_exin_bound

	# where do both criteria hold?
	rates_accept = np.all((mean_exin_accept, dist_r_exin_accept), axis=0)
	print('We have {0} results with acceptable rates.'.format(rates_accept.sum()))

	# if the rates fit, what are the resulting CVs?
	cvs = data[:, 14]
	cvs_accept = np.all((cvs <= cv_upper_bound, cvs >= cv_lower_bound), axis=0)

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
	# testvalue3 = np.sqrt((distance_penalty * np.ones_like(data[ 
	#   all_accept, 6 ]) - data[ all_accept, 6 ])**2 + dist_r_exin[ all_accept ]**2)
	# testindices3 = np.sqrt((distance_penalty * np.ones_like(data[ :, 6 ]) - data[ 
	#   :, 6 ])**2 + dist_r_exin**2) == np.amin(testvalue3)
	# testconfigs3 = data[ testindices3, : ]
	# ... the lower RMSE of rate difference and dV2?
	testvalue3 = np.sqrt((distance_penalty * np.ones_like(data[ 
	    all_accept, 6 ]) - data[ all_accept, 6 ])**2 + dist_r_exin[ all_accept ]**2)
	testindices3 = np.sqrt((distance_penalty * np.ones_like(data[ :, 6 ]) - data[ 
	    :, 6 ])**2 + dist_r_exin**2) == np.amin(testvalue3)
	testconfigs3 = data[ testindices3, : ]
	return testconfigs1, testconfigs2, testconfigs3


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

# ****************************************************************************************************************************
# MICROCIRCUIT CONNECTIVITY TRANSFORMER FUNCTIONS
# ****************************************************************************************************************************

def find_Nsyn(Cold, Npre, Npost):
    """
    compute indegree * Npost = actual number of synapses
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
    h1 = (Npre * Npost - 1.) / (Npre * Npost)
    h2 = np.power(h1, K)
    h3 = 1. - h2
    return h3


def compute_new_connectivity(conn_probs, N_full_new, N_full_old, layers, pops, newpopdict, new_structure, K_bg=None):
    """
    the core function.
    """
    n_new_pops_per_layer = len(newpopdict)
    n_pops_per_layer = len(pops)
    n_layers = len(layers)
    connmat = np.zeros((n_layers * n_new_pops_per_layer, n_layers * n_new_pops_per_layer))
    if K_bg is not None:
        K_bg = np.array(K_bg, dtype=float)
        K_bg_new = np.zeros((n_layers, n_new_pops_per_layer))

    for target_layer, target_layer_dict in N_full_new.iteritems():
        for target_pop, target_pop_size in target_layer_dict.iteritems():
            bindex_t = new_structure[ target_layer ][ target_pop ]

            # here we compute the new background indegree
            if K_bg is not None:
                if target_pop_size == 0:
                    K_bg_new.flat[ bindex_t ] = 0
                else:
                    if 'R' in target_pop:  # take target connectivity from E for R?
                        target_index = n_pops_per_layer * layers[ target_layer ] + pops[ 'E' ]
                        Npost_old = N_full_old[ target_layer ][ 'E' ]
                    else:  # just E and I populations connecting
                        target_index = n_pops_per_layer * layers[ target_layer ] + pops[ target_pop ]
                        Npost_old = N_full_old[ target_layer ][ target_pop ]
                    rel_bg = float(target_pop_size) / float(Npost_old)
                    K_bg_new.flat[ bindex_t ] = K_bg.flat[ target_index ] * rel_bg

            for source_layer, source_layer_dict in N_full_new.iteritems():
                for source_pop, source_pop_size in source_layer_dict.iteritems():
                    bindex_s = new_structure[ source_layer ][ source_pop ]
                    if source_pop_size == 0 or target_pop_size == 0:                                        # target or source have size 0?
                        connmat[ bindex_t, bindex_s ] = 0
                    else:
                        # collect the old connectivity:
                        if 'R' in source_pop:                                                               # take source connectivity from E for R?
                            source_index = n_pops_per_layer * layers[ source_layer ] + pops[ 'E' ]          # get the source index for the old connectivity list
                            Npre_old = N_full_old[ source_layer ][ 'E' ]
                        else:
                            source_index = n_pops_per_layer * layers[ source_layer ] + pops[ source_pop ]   # get the source index for the old connectivity list
                            Npre_old = N_full_old[ source_layer ][ source_pop ]
                        if 'R' in target_pop:                                                               # take target connectivity from E for R?
                            target_index = n_pops_per_layer * layers[ target_layer ] + pops[ 'E' ]          # get the target index for the old connectivity list
                            Npost_old = N_full_old[ target_layer ][ 'E' ]
                        else:                                                                               # just E and I populations connecting
                            target_index = n_pops_per_layer * layers[ target_layer ] + pops[ target_pop ]   # get the target index for the old connectivity list
                            Npost_old = N_full_old[ target_layer ][ target_pop ]
                        Cold = conn_probs[ target_index ][ source_index ]                                   # get the 'old' connectivity list entry

                        # compute new connectivity:
                        if Cold == 0:                                                                       # was connectivity 0 anyway?
                            connmat[ bindex_t, bindex_s ] = 0
                        else:                                                                               # compute new connectivity
                            Nsyn_new = find_Nsyn(Cold, source_pop_size, target_pop_size)                    # replaces K with Nsyn, find_C with find_Nsyn
                            rel = float(source_pop_size) / float(Npre_old) * float(target_pop_size) / float(Npost_old)                  # number of synapses with same relation as between old population sizes

                            C = find_new_C(Nsyn_new * rel, N_full_new[ source_layer ][ source_pop ], N_full_new[ target_layer ][ target_pop ])
                            connmat[ bindex_t, bindex_s ] = C
    if K_bg is None:
        return connmat
    else:
        return connmat, K_bg_new


# ****************************************************************************************************************************
# THE L5 BRUNEL NETWORK
# ****************************************************************************************************************************

def run_brunel(networkparamdict, plotting=True):
    """ a version that is driven just as the MC L5
	!!! WARNING !!!
	Possibly faulty connection for the poisson generators ->
	return to all_to_all conn_rule.
    """
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

# ****************************************************************************************************************************
# MAIN FILE CODE
# ****************************************************************************************************************************


if __name__ == '__main__':

	print('Importing suitable GIF2 configurations')
	testconfigs1, testconfigs2, testconfigs3 = configuration_extractor()
	
	print('Importing complete.')
	
	configure_nest_kernel(cluster=False)
	print('Nest Kernel configured')

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