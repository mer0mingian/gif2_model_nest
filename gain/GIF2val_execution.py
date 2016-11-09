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
with sys.arg[3] give the option for multiple outputs, each showing the fitted
 function to the histogram.
"""
import getopt
import sys
import numpy as np
execfile('GIF2val_functions_blau.py')

import time
import nest
try: nest.set_verbosity('M_ERROR')
except: print('Changing the nest verbosity did not succeed')
try: nest.Install("gif2_module")
except: print ("The GIF2 model had already been installed")
dt = 0.01  # simulation timestep
freqindex = int(sys.argv[ 1 ])
maxindex = int(sys.argv[ 2 ])
show_fits = bool(sys.argv[ 3 ])
queueid = int(sys.argv[ 4 ])

# setting up the frequency array and determining the frequency from the array
frequencies = np.hstack((np.array([ 0 ]),
                         np.logspace(-1., 2., num=maxindex )))
f = frequencies[ freqindex ]

# The script will iterate over all conditions in this list.
# It's strings should be recognised by get_stim_params()!
noise_conditions = ['R', 'r']

for condition in noise_conditions:
    starttime = time.time()
    nest.ResetKernel()
    nest.SetKernelStatus({#"local_num_threads": 16,
                          "print_time": False,
                          "overwrite_files": False,
                          "resolution": dt})

    simparameterdict = import_params_as_dict(filename='jobdict.txt')
    # contains N, binwidth, current/poisson, t_rec, t_recstart,
    # simindex, synweight

    neuronparamdict = import_params_as_dict(filename='neurondict.txt')
    # contains tau_1, C_m, g_rr=g_1, g, V_reset, E_L, V_th

    I_stimdict, r_stimdict, xi_stimdict = \
        get_stim_params(simparameterdict, f, condition, dt)

    # -------------------------------------------------------------------------
    # Starting simulation part here
    # -------------------------------------------------------------------------

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
    nest.Simulate(simparameterdict[ 't_rec' ] +
                  simparameterdict[ 't_recstart' ])


    # RECORDING
    spike_senders = nest.GetStatus(spikedetector, "events")[ 0 ][ "senders" ]
    spike_times = nest.GetStatus(spikedetector, "events")[ 0 ][ "times" ]

    # EVALUATE
    # Create the istogram to fit with
    hist_bins, hist_heights, hist_binwidth = compute_histogram(
        spike_times, simparameterdict)


    # Fit and compute gain
    gain, exp_r_0 = compute_gain(hist_bins, hist_heights,
                                 simparameterdict,
                                 I_stimdict, f, dt)

    resultdict = dict(freqindex=freqindex,
                      gain=gain,
                      exp_r_0=exp_r_0,
                      neuronparamdict=neuronparamdict,
                      simparameterdict=simparameterdict,
                      condition=condition)

    if show_fits:
        multiplot(condition, simparameterdict[ 'simindex' ], freqindex, hist_bins,
                  hist_heights, f, gain[ 0 ] * I_stimdict[ 'amplitude' ],
                  gain[ 1 ], exp_r_0)

    write_results(resultdict)
    #nest.ResetKernel()
    endtime = time.time()
    print('condition {0} with frequency {1} took {2}ms.'.format(
        condition, f, endtime - starttime))

