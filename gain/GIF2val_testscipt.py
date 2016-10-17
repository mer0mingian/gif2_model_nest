# coding=utf-8
"""
This file is intendend to be run in ipython to test for coding errors
independent of a bash script.
"""

import sys, getopt
import numpy as np
execfile('GIF2val_functions_blau.py')

import nest
try: nest.set_verbosity('M_ERROR')
except: print('Changing the nest verbosity did not succeed')
try: nest.Install("gif2_module")
except: print ("The GIF2 model had already been installed")
dt = 0.1  # simulation timestep
nest.SetKernelStatus({"resolution": dt,
                      "print_time": True,
                      "overwrite_files": False})

condition = 'R'
f = 5.0

simparameterdict = import_params_as_dict(filename='jobdict.txt')
# contains N, binwidth, current/poisson, t_rec, t_recstart,
# simindex, synweight
simparameterdict['t_recstart'] = 0.0

neuronparamdict = import_params_as_dict(filename='neurondict.txt')
# contains tau_1, C_m, g_rr=g_1, g, V_m, V_reset, E_L, V_th

I_stimdict, r_stimdict, xi_stimdict = \
    get_stim_params(simparameterdict, f, condition, dt)

# -------------------------------------------------------------------------
# Starting simulation part here
# -------------------------------------------------------------------------
# CALCULATE SECOND-ORDER PARAMETERS
# alpha = g * tau_1 / C_m  # regime determining constant 1
# beta = g_1 * tau_1 / C_m  # regime determining constant 2
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
nest.SetStatus(spikedetector, {"withgid":               True,
                               "withtime":              True,
                               "start": simparameterdict[ 't_recstart' ],
                               "to_file":               False})

# BUILD NETWORK
nest.Connect(stim_I, neuron)
nest.Connect(stim_xi, neuron)
nest.Connect(neuron, spikedetector)

# SIMULATE
nest.Simulate(1500.0)

# RECORDING
spike_senders = nest.GetStatus(spikedetector, "events")[ 0 ][ "senders" ]
spike_times = nest.GetStatus(spikedetector, "events")[ 0 ][ "times" ]

# EVALUATE
# Create the istogram to fit with
hist_bins, hist_heights, hist_binwidth = compute_histogram(
    spike_times, simparameterdict)

# Fit and compute gain
gain, exp_r_0 = compute_gain(hist_bins, hist_heights, hist_binwidth,
                             I_stimdict, f, dt, condition)

resultdict = dict(freqindex=0,
                  gain=gain,
                  exp_r_0=exp_r_0,
                  neuronparamdict=neuronparamdict,
                  simparameterdict=simparameterdict,
                  condition=condition)

