# -*- coding: utf-8 -*-
"""
This project intends to find parameter section, where the firing rate of the
generalised integrate-and-fire model with 2 variables is comparable to that of
the integrate-and-fire neuron.
(see Richardson, Brunel and Hakim, 2003, "From subthreshold resonance to firing
rate resonance." and Brunel et al., 2003.)

Parameters indices are deliverd via sys.argv to be run on a queing system.
"""
import numpy as np
import sys
import elephant.statistics as es
import csv
from itertools import product

cluster = True


def write_excitability_results(filename, resultarray):
    """
	exports the results for a single datapoint to the results file.
	"""
    with open(filename, 'a') as csvfile:
        somewriter = csv.writer(csvfile, delimiter=';')
        for rows in np.arange(0, resultarray.shape[1]):
            somewriter.writerow(list(resultarray[ :, rows ]))
    csvfile.close()


def read_excitability_results(filename='excitability_results1.csv'):
	"""
	imports simulation or neuron parameters from txt-file into distionary
	"""
	paramdict = dict()
	with open(filename, 'r') as jobhashes:
		lasthash = jobhashes.readlines()[ -1 ]
	lasthash = lasthash.strip('\n').strip('{').strip('}')
	splitstring = lasthash.split(',')
	for item in splitstring:
		splititem = item.strip(' ').split(':')
		paramdict[ splititem[ 0 ].strip('\'') ] = splititem[ 1 ].strip(
			' ').strip('\'')
	for item in paramdict:
		try:
			paramdict[ item ] = float(paramdict[ item ])
		except ValueError:
			pass
	return paramdict


def read_solution():
    solutionlyarray = np.genfromtxt('solutionarray.txt', unpack=True)
    return solutionlyarray



if __name__ == '__main__':
    # Configure Nest
    import nest
    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    dt = 0.1  # the resolution in ms
    nest.SetKernelStatus(
        {"resolution": dt, "print_time": False, "overwrite_files": True})
    if cluster:
        nest.SetKernelStatus({"local_num_threads": 4})
    try:
        nest.Install("gif2_module")
    except:
        pass


    # Simulation setup
    recstart = 1000.0
    simtime = 2500.0

    # Index variables
    p_rate = float(sys.argv[ 1 ])
    C_m2 = 100.0
    indexarray = np.zeros(10)
    C_range = np.linspace(100.0, 600.0, 20)  # np.arange(100.0, 600.0, 25.0)
    g_range = np.linspace(5.0, 90.0, 20)  # np.arange(5.0, 90.0, 5.0)
    cases = len(C_range) * len(g_range) * len(g_range)
    g_m = 5.0
    g_1 = 5.0

    # Shared/fixed neuron parameters
    tauSyn = 0.5
    V_theta = 15.0
    tau_1 = 100.0
    V_range = 6.0
    synweight = 87.8
    J_ex = 0.125

    j = 0
    p_range = np.linspace(5000.0, 20000.0, 20)
    for i in product(C_range, g_range, g_range, p_range):
        j += 1
        nest.ResetKernel()
        p_rate = i[ 3 ]

        # adopt neuron parameters
        gif_params = {"tau_1":      tau_1,
                      "C_m":        i[ 0 ],  # C_m2,
                      "tau_syn_ex": tauSyn,
                      "tau_syn_in": tauSyn,
                      "g_rr":       i[ 2 ],  # g_1,
                      "g":          i[ 1 ],  # g_m,
                      "V_m":        0.0,
                      "V_reset":    V_theta - V_range,
                      "E_L":        0.0,
                      "V_th":       V_theta,
                      "w":          0.0}

        iaf_params = {"C_m":        250.0,
                      "tau_m":      10.0,
                      "tau_syn_ex": tauSyn,
                      "tau_syn_in": tauSyn,
                      "t_ref":      2.0,
                      "E_L":        0.0,
                      "V_reset":    0.0,
                      "V_m":        0.0,
                      "V_th":       V_theta}

        det_params = {"withtime": True,
                      "withgid":  False,
                      "to_file":  False,
                      "start":    recstart}

        stm_params = {"rate":      p_rate,
                      "amplitude": 0.025 * p_rate,
                      "frequency": 10.0,
                      "phase":     0.0}

        nest.CopyModel("static_synapse", "excitatory",
                       {"weight": J_ex * synweight, "delay": 1.0})

        # Create devices and neurons
        drive = nest.Create("sinusoidal_poisson_generator", params=stm_params)
        iafspikes = nest.Create("spike_detector", params=det_params)
        gifspikes = nest.Create("spike_detector", params=det_params)
        gif = nest.Create("gif2_psc_exp")
        iaf = nest.Create("iaf_neuron")

        # Connect everything
        nest.Connect(drive, gif + iaf, syn_spec="excitatory")
        nest.Connect(gif, gifspikes)
        nest.Connect(iaf, iafspikes)

        # Simulate
        if not cluster:
            sys.stdout.write('\r')
            sys.stdout.write('case {0}/{1}'.format(j, cases))
            sys.stdout.flush()
        nest.Simulate(recstart + simtime)

        # Evaluate
        rate_iaf = nest.GetStatus(iafspikes, "n_events")[ 0 ] / simtime * 1000.
        rate_gif = nest.GetStatus(gifspikes, "n_events")[ 0 ] / simtime * 1000.
        cv_gif = es.cv(es.isi(nest.GetStatus(gifspikes, "events")
                              [ 0 ][ "times" ]))
        cv_iaf = es.cv(es.isi(nest.GetStatus(iafspikes, "events")
                              [ 0 ][ "times" ]))

        # write out
        indexarray = np.vstack((indexarray, np.array([p_rate, i[ 0 ], i[ 1 ],
                i[ 2 ], tau_1, V_range, rate_gif, rate_iaf, cv_gif, cv_iaf])))

    write_excitability_results('excite.csv', indexarray)
    np.savetxt('excitability.txt', indexarray[1:, :], fmt="%12.8G")
    solutionlyarray = np.hstack((indexarray[:, 1:4], indexarray[:, 6:8]))

    # save to text:
    np.savetxt('solutionarray.txt', solutionlyarray[1:, :], fmt="%12.6G")