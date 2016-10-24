#  -*- coding: utf-8 -*-
"""
This is the Blaustein-version of the script that should be run to
obtain F6CD from Richardson et al., 2003.
Usage: can this script as:
srun GIF2val_executaion.py <arraynumber> <maxarraynumber>

TODO: for now, the script only works with current input. To study synaptic
drive, enable the parameter extraction from the dictionary for poisson drive.

# Recent changes:
v2:
Implmented compute_gain2, which corrects for the phase delay in the driving
generators and uses a phase difference to a fit to the driving signal instead
of just the phase extracted from the sine fitted to rates.

v3:
lanning to add the option to give a plot with 25 subplots from single data
point fits.
"""
import sys, getopt
import numpy as np
execfile('GIF2val_functions_blau.py')

import nest
try: nest.set_verbosity('M_ERROR')
except: print('Changing the nest verbosity did not succeed')
try: nest.Install("gif2_module")
except: print ("The GIF2 model had already been installed")
dt = 0.01  # simulation timestep
freqindex = int(sys.argv[ 1 ])
maxindex = int(sys.argv[ 2 ])
nest.SetKernelStatus({"resolution": dt,
                      "print_time": False,
                      "overwrite_files": False})

# setting up the frequency array and determining the frequency from the array
frequencies = np.hstack((np.array([ 0 ]),
                         np.logspace(-1., 2., num=maxindex )))
f = frequencies[ freqindex ]

# The script will iterate over all conditions in this list.
# It's strings should be recognised by get_stim_params()!
noise_conditions = ['R', 'r']


for condition in noise_conditions:
    nest.SetKernelStatus({"resolution": dt,
                          "print_time": False,
                          "overwrite_files": False})

    simparameterdict = import_params_as_dict(filename='jobdict.txt')
    # contains N, binwidth, current/poisson, t_rec, t_recstart,
    # simindex, synweight
    simparameterdict[ 'simindex' ] = int(simparameterdict[ 'simindex' ] + 1)
    # with open('jobdict.txt', 'a') as fs:
    #     fs.write("\n{")
    #     [ fs.write("'{0}': '{1}', ".format(key, value)) for key, value in simparameterdict.items()[ :-1 ] ]
    #     fs.write("'{0}': '{1}'".format(simparameterdict.items()[ -1 ][0], simparameterdict.items()[ -1 ][1]))
    #     fs.write("}")
    #     fs.close()

    neuronparamdict = import_params_as_dict(filename='neurondict.txt')
    # contains tau_1, C_m, g_rr=g_1, g, V_reset, E_L, V_th

    I_stimdict, r_stimdict, xi_stimdict = \
        get_stim_params(simparameterdict, f, condition, dt)

    # -------------------------------------------------------------------------
    # Starting simulation part here
    # -------------------------------------------------------------------------
    # CALCULATE SECOND-ORDER PARAMETERS
    # alpha = g * tau_1 / C_m  # regime determining constant 1
    # beta = g_1 * tau_1 / C_m  # regime determining constant 2
    # factor = np.sqrt((alpha + 1.) * (alpha + 1.) + 1.) - (1. + alpha)
    # t_sim = t_rec + t_recstart  # length of data acquisition time
    # exp_f_r = 1. / (2. * np.pi * tau_1) * np.sqrt(np.sqrt(
    #     (alpha + beta + 1.) * (alpha + beta + 1.) - (alpha + 1.) * (
    #     alpha + 1.)) - 1.)
    # # formula A6, p.2552

    # CREATE DEVICES
    neuron = nest.Create('gif2_psc_exp',
                         n=int(simparameterdict[ 'N' ]),
                         params=neuronparamdict)

    stim_I = nest.Create('ac_generator', params=I_stimdict)
    stim_xi = nest.Create('noise_generator', params=xi_stimdict)
    spikedetector = nest.Create('spike_detector')
    nest.SetStatus(spikedetector, {"withgid": True,
                                   "withtime": True,
                                   "start": simparameterdict[ 't_recstart' ],
                                   "to_file": False})


    # BUILD NETWORK
    nest.Connect(stim_I, neuron)
    nest.Connect(stim_xi, neuron)
    nest.Connect(neuron, spikedetector)

    # SIMULATE
    nest.Simulate(simparameterdict[ 't_rec' ] + simparameterdict[ 't_recstart' ])


    # RECORDING
    spike_senders = nest.GetStatus(spikedetector, "events")[ 0 ][ "senders" ]
    spike_times = nest.GetStatus(spikedetector, "events")[ 0 ][ "times" ]

    # EVALUATE
    # Create the istogram to fit with
    hist_bins, hist_heights, hist_binwidth = compute_histogram(
        spike_times, simparameterdict)


    # Fit and compute gain
    gain, exp_r_0 = compute_gain(hist_bins, hist_heights,
                                 simparameterdict[ 't_rec' ],
                                 simparameterdict[ 't_recstart' ],
                                 I_stimdict, f, dt)

    resultdict = dict(freqindex=freqindex,
                      gain=gain,
                      exp_r_0=exp_r_0,
                      neuronparamdict=neuronparamdict,
                      simparameterdict=simparameterdict,
                      condition=condition)

    write_results(resultdict)
    nest.ResetKernel()


