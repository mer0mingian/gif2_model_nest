# -*- coding: utf-8 -*-

# -------------------- MAKE STUFF LOOK NICE -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('mathtext', default='regular')

from nested_dict import nested_dict
import export_summary
from export_summary import myprint
# ---------------------- WE WANT NEST, NEST, NEST -----------------------
import nest
import nest.raster_plot
import nest.voltage_trace

nest.ResetKernel()
try:
    nest.Install("gif2_module")
except:
    print ("The GIF2 model had already been installed")

# ---------- FOR FITTING THE HISTOGRAM AND COMPUTING THE GAIN -----------
import scipy.optimize as opti
import scipy.fftpack as fftpack


def mysine(t, f2, amp, off):
    return amp * np.sin(2. * np.pi * f2 * t) + off


"""
HOW TO USE THIS SCRIPT:
- first set the input flag for either direct current stimulation or synaptic poisson drive
- select the neuron parameters (presets exist for the original Richardson configuration and microcircuit-similar one)
- choose to display the trace for the second variable or not
- Note: r_0 and r_1 denote the poisson input parameters, not the firing rate, as in the diagram
"""
flag_stim_type = 'current'  # options: poisson, current
flag_show_w = False
flag_annotate = True
flag_injection = 'manual'
plotmaxnr = 1

r_0_manual = 8050.  # 7500.0 * 40.0 / 87.8
r_1_manual = 220.  # 150.0
I_0_manual = 720.  # 1800.
I_1_manual = 37.  # 250.
I_N_manual = 255.  # 150.0  # Amplitude of the Gaussian White Noise in pA, 550.0
synweight = 40.0  # synaptic weight, 87.8

# SIMULATION PARAMETERS
t_run = 400.0  # total length of simulationb
t_recstart = 500.0  # time to begin recording, usually 350.0
f = 10.0  # The frequency of the drive, Hz
dt = 1.0  # Recording timestep
N = 2500  # Number of trials/parallel unconnected neurons
nbins = 100.0  # number of bins for histogram
nest.SetKernelStatus(
    {"resolution": dt, "print_time": True, "overwrite_files": True})

# NEURON PARAMETERS
# NEURON PARAMETERS
tau_1 = 100.  # 100.0  # time constant for 2nd variable, ms
C_m = 250.0  # membrane capacity, pF
g_1 = 25.0  # 89.466  # conductance 2nd variable, nS, 30.0
g = 10.  # conductance 1st variable, nS, 6.5
t_ref = 2.0  # length refractory period, ms
E_L = 0.0  # voltage leak term, mV
V_th = 20.0  # Spiking threshold, mV
V_reset = 14.0  # reset current, mV

# STIMULATOIN PROTOCOL
if flag_stim_type == 'current':
    #    I_0 = 1500.
    #    I_1 = 0.025 * I_0
    if flag_injection == 'Richardson Fig1 w=87.8':
        I_N = 250.0  # Amplitude of the Gaussian White Noise in pA, 550.0
        I_0 = 1628.57  # stationary component of the input current, standard: 780.0
        I_1 = 22.6  # oscillatory component of the input current, 59.0
        C_m = 250.0
        r_0 = g * I_0
        r_1 = g * I_1
    elif flag_injection == 'Richardson Fig1 w=40.0':
        I_N = 300.0  # Amplitude of the Gaussian White Noise in pA, 550.0
        I_0 = 1628.57  # stationary component of the input current, standard: 780.0
        I_1 = 22.6  # oscillatory component of the input current, 59.0
        C_m = 250.0
        r_0 = g * I_0
        r_1 = g * I_1
    elif flag_injection == 'manual':
        I_0 = I_0_manual  # Poissonian input, stationary component, spikes/s
        I_1 = I_1_manual  # Sinusoidal poissonian input amplitude, spikes/s
        I_N = I_N_manual  # Amplitude of the Gaussian White Noise in pA, 550.0
        r_0 = g / synweight * I_0
        r_1 = g / synweight * I_1
        C_m = 500.
    elif flag_injection == 'Richardson Fig6 high':
        # correction = np.sqrt(2.)*3. #  should be np.sqrt(2.) to correct for C = 250 instead of 500
        I_N = 550.0  # Amplitude of the Gaussian White Noise in pA, 550.0
        I_0 = 780.0  # stationary component of the input current, standard: 780.0
        I_1 = 59.0  # oscillatory component of the input current, 59.0
        C_m = 500.0  # membrane capacity
        g = 25.0  # in nS, NOT µS
        g_1 = 25.0
elif flag_stim_type == 'poisson':
    if flag_injection == 'Richardson Fig1 w=87.8':
        C_m = 250.0
        r_0 = 3500.3
        r_1 = 150.8
        I_0 = r_0 / g
        I_1 = r_1 / g
        I_N = I_N_manual
        synweight = 87.8
    elif flag_injection == 'Richardson Fig1 w=40.0':
        C_m = 250.0
        r_0 = 8025.0
        r_1 = 255.8
        I_0 = r_0 / g
        I_1 = r_1 / g
        I_N = I_N_manual
        synweight = 40.0
    elif flag_injection == 'manual':
        r_0 = r_0_manual  # Poissonian input, stationary component, spikes/s
        r_1 = r_1_manual  # Sinusoidal poissonian input amplitude, spikes/s
        I_0 = r_0 / g
        I_1 = r_1 / g
        I_N = I_N_manual

# SECOND-ORDER PARAMETERS
alpha = g * tau_1 / C_m  # regime determining constant 1
beta = g_1 * tau_1 / C_m  # regime determining constant 2
# factor = (alpha - 1) ** 2 / 4  # regime total factor. factor > beta for resonance
factor = np.sqrt((alpha + 1.) * (alpha + 1.) + 1.) - (1. + alpha)
t_sim = t_run + t_recstart  # length of data acquisition time
exp_f_r = 1. / (2. * np.pi * tau_1) * np.sqrt(np.sqrt(
    (alpha + beta + 1.) * (alpha + beta + 1.) - (alpha + 1.) * (
    alpha + 1.)) - 1.)
# formula A6, p.2552

# DEVICES
# see parameters from stimulation protocol in Fig. 6, p2546 in Richardson et al. (2003)
neuron = nest.Create("gif2_psc_exp", n=N, params={
    "tau_1":   tau_1,
    "C_m":     C_m,
    "g_rr":    g_1,
    "g":       g,
    "V_m":     0.0,
    "V_reset": V_reset,
    "E_L":     E_L,
    "V_th":    V_th})
if flag_stim_type == 'poisson':
    stimparams_r = [ {'rate':      r_0,
                      'amplitude': r_1,
                      'frequency': f,
                      'phase':     0.0} ]
    stim_r = nest.Create('sinusoidal_poisson_generator', params=stimparams_r)
elif flag_stim_type == 'current':
    stimparams_I = [ {'offset':    I_0,
                      'amplitude': I_1,
                      'frequency': f,
                      'phase':     0.0} ]
    stim_I = nest.Create('ac_generator', params=stimparams_I)
stimparams_xi = [ {'mean':      0.0,
                   'std':       I_N,
                   'dt':        dt,
                   'frequency': 0.0,
                   'phase':     0.0,
                   'std_mod':   0.0} ]
stim_xi = nest.Create('noise_generator', params=stimparams_xi)
voltmeter = nest.Create('voltmeter')
nest.SetStatus(voltmeter, {"withgid":  True,
                           "withtime": True,
                           "interval": dt,
                           "start":    t_recstart})
spikedetector = nest.Create('spike_detector')
nest.SetStatus(spikedetector, {"withgid":  True,
                               "withtime": True,
                               "start":    t_recstart})
multimeter = nest.Create("multimeter",
                         params={"interval":    dt,
                                 "record_from": [ "V_m", "w" ],
                                 "withgid":     True})

# BUILD NETWORK
if flag_stim_type == 'poisson':
    nest.Connect(stim_r, neuron, syn_spec={'weight': synweight})
elif flag_stim_type == 'current':
    nest.Connect(stim_I, neuron)
    nest.Connect(stim_xi, neuron)
nest.Connect(voltmeter, neuron)
nest.Connect(multimeter, neuron)
nest.Connect(neuron, spikedetector)

# SIMULATE
nest.Simulate(t_sim)
# -----------------------------------------------------------------------------------------------------
# RECORDING
multi_senders = nest.GetStatus(multimeter, "events")[ 0 ][
    "senders" ]  # extract the senders for each event in V_m's
multi_voltage = nest.GetStatus(multimeter, "events")[ 0 ][
    "V_m" ]  # extract the voltages
multi_times = nest.GetStatus(multimeter, "events")[ 0 ][ "times" ]
multi_w = nest.GetStatus(multimeter, "events")[ 0 ][ "w" ]
v_panel1 = multi_voltage[
    multi_senders == 1 ]  # voltages recorded from neuron 1
w_panel1 = multi_w[ multi_senders == 1 ]  # 2nd variable trace from neuron 1
times_panel1 = multi_times[
    multi_senders == 1 ]  # times at which these were recorded

spikes = nest.GetStatus(spikedetector, "n_events")[ 0 ]
spike_senders = nest.GetStatus(spikedetector, "events")[ 0 ][ "senders" ]
spike_times = nest.GetStatus(spikedetector, "events")[ 0 ][ "times" ]
spike_times_panel1 = spike_times[ spike_senders == 1 ]
spike_senders_panel2 = spike_senders[ spike_senders <= 51 ]
spike_times_panel2 = spike_times[ spike_senders <= 51 ]

# My self-designed plotting stuff!!! Here goes the fun!
plt.clf()
# ---------------------------------------------- PANEL 1 ------------------------------------------------
if flag_show_w:
    plt.subplot(411)
else:
    plt.subplot(311)
# plt.title("Variable traces", size=14)
plt.xlim([ t_recstart, t_sim ])
plt.ylim([ -70.0, 40.0 ])
plt.ylabel(r"$V_m$ [mV]", size=14)
plt.plot(times_panel1, v_panel1 - 70. + E_L)
y_bar_lims = np.array([ V_th - 70.0 + E_L, 10.75 ])
for i in spike_times_panel1:
    plt.vlines(x=times_panel1[ times_panel1 == i ] - dt,
               ymin=y_bar_lims[ 0 ] - 0.5,
               # v_panel1[ times_panel1[ times_panel1 == i ][0] ] - 70. + E_L + V_th, # y_bar_lims[0],
               ymax=y_bar_lims[ 1 ],
               colors='blue')
if flag_show_w:
    plt.subplot(412)
    plt.xlim([ t_recstart, t_sim ])
    plt.ylabel(r"$w$ [mV]", size=14)
    plt.plot(times_panel1, w_panel1)

ax2 = plt.twinx()
if flag_stim_type == 'current':
    currentdrive = np.ones_like(times_panel1) * I_0 + np.sin(
        1. / 500. * np.pi * f * times_panel1) * I_1
    ax2.plot(times_panel1, currentdrive, color='black')
    ax2.set_xlim([ t_recstart, t_sim ])
    ax2.set_ylim([ I_0 - 12.5 * I_1, I_0 + 2.5 * I_1 ])
    ax2.set_ylabel(r"$I_E$ [nA]", size=14)
    if flag_annotate:
        # I_0 line
        mean_I_0s = times_panel1[ abs(currentdrive - I_0) <= 1e-3 ]
        mean_I_0s = mean_I_0s[ mean_I_0s >= t_recstart ]

        plt.plot((mean_I_0s[ plotmaxnr ], mean_I_0s[ plotmaxnr + 2 ]),
                 (I_0, I_0), ls='-', color='black')
        # I_0 arrow
        plt.annotate(r"$I_0$",
                     xy=((mean_I_0s[ plotmaxnr ] + mean_I_0s[
                         plotmaxnr + 1 ]) / 2.0, I_0), xycoords='data',
                     xytext=(0.12, 0.93), textcoords='axes fraction',
                     arrowprops=dict(facecolor='black', shrink=0.02, width=0.4,
                                     headwidth=6.5))
        # I_1 elongated
        max_I_1s = times_panel1[ currentdrive == max(currentdrive) ]
        max_I_1s = max_I_1s[ max_I_1s >= mean_I_0s[ plotmaxnr ] ]
        plt.plot((max_I_1s[ 0 ], max_I_1s[ 0 ]), (I_0 + I_1, I_0), ls='-',
                 color='red', linewidth='2')
        # I_1 arrow
        plt.annotate(r"$I_1$",
                     xy=(max_I_1s[ 0 ], I_0 + I_1 / 2.), xycoords='data',
                     xytext=(0.25, 0.93), textcoords='axes fraction',
                     arrowprops=dict(facecolor='black', shrink=0.02, width=0.4,
                                     headwidth=6.5))


elif flag_stim_type == 'poisson':
    poissondrive = np.ones_like(times_panel1) * r_0 + np.sin(
        1. / 500. * np.pi * f * times_panel1) * r_1 / 1000.
    ax2.plot(times_panel1, poissondrive, color='black')
    ax2.set_xlim([ t_recstart, t_sim ])
    ax2.set_ylim([ r_0 - 12.5 * r_1 / 1000., r_0 + 2.5 * r_1 / 1000. ])
    ax2.set_ylabel(r"$I_E$ [pA]", size=14)

# if flag_annotate:
# TODO: r_0 arrow
# TODO: r_1 elongated
# TODO: r_1 arrow

# ---------------------------------------------- PANEL 2 ------------------------------------------------
# The subplot-compatible version of the raster plot
if flag_show_w:
    plt.subplot(413)
else:
    plt.subplot(312)
hist_binwidth = t_run / nbins  # !> 8.0 for the instantaneous rate
plt.xlim([ t_recstart, t_sim ])
plt.ylabel('Neuron ID', size=14)
plt.plot(spike_times_panel2, spike_senders_panel2,
         ".k")  # <----- THIS IS THE RASTER PLOT
plt.ylim([ 0, 52 ])
# ---------------------------------------------- PANEL 3 ------------------------------------------------
if flag_show_w:
    plt.subplot(414)
else:
    plt.subplot(313)
t_bins = np.arange(np.amin(spike_times), np.amax(spike_times),
                   float(hist_binwidth))
n, bins = nest.raster_plot._histogram(spike_times,
                                      bins=t_bins)  # uses the NEST histogram function. see below!
heights = 1000. * n / (hist_binwidth * N * dt)  # num_neurons)  # population firing rate
plt.bar(t_bins, heights, width=hist_binwidth, color="blue", edgecolor="black")
plt.yticks(
    [ int(x) for x in np.linspace(0.0, int(max(heights) * 1.1) + 5, 4) ])
plt.ylabel("rate [Hz]", size=14)
plt.xlabel("time [ms]", size=14)
# plt.xlim([t_recstart+0.5*hist_binwidth, t_sim+0.5*hist_binwidth])
plt.xlim([ t_recstart, t_sim ])

# mean firing rate
r_0_reconstructed = np.mean(heights)
plt.plot(np.ones(t_sim + 1) * r_0_reconstructed, c="black", ls="-")

# the instantaneous firing rate sinusoidal fit to the histogram
f2 = f / 1000.
amp_guess = (max(heights) + abs(min(heights))) / 2. - r_0_reconstructed
popt, pcov = opti.curve_fit(mysine, bins, heights,
                            p0=(f2, amp_guess, r_0_reconstructed))
sinecurve = mysine(bins - hist_binwidth, f2, popt[ 1 ], popt[ 2 ])
plt.plot(bins, sinecurve, color='black')

# -----------------------------------------------------------------------------------------------------
# calculate the signal gain:
r_1_reconstructed = max(sinecurve - popt[ 2 ])
# if flag_stim_type == 'current':
gain = abs(r_1_reconstructed / I_1)
# elif  flag_stim_type == 'poisson':
# K = 1. # indegree
# I_0_reconstructed =
# I_1_reconstructed =
# gain = abs(r_1_reconstructed / I_1_reconstructed)
#  TODO: equivalent of current input for poisson drive --> compute gain
# -----------------------------------------------------------------------------------------------------

# annotations
if flag_annotate:
    # r_0 arrow
    plt.annotate(r"$r_0$",
                 xy=(t_recstart + 75., r_0_reconstructed), xycoords='data',
                 xytext=(0.15, 0.93), textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', shrink=0.02, width=0.4,
                                 headwidth=6.5))
    # the peak of the sinus elongated to the right
    max_r_1s = bins[ sinecurve >= r_0_reconstructed + 0.9 * r_1_reconstructed ]
    max_r_1s = max_r_1s[ max_r_1s >= t_recstart ]
    plt.plot((max_r_1s[ plotmaxnr + 2 ], max_r_1s[ plotmaxnr + 4 ]),
             (r_0_reconstructed + r_1_reconstructed,
              r_0_reconstructed + r_1_reconstructed),
             ls='-', color='black')
    # r_1 line
    plt.plot(((max_r_1s[ plotmaxnr + 2 ] + max_r_1s[ plotmaxnr + 4 ]) / 2.0,
              (max_r_1s[ plotmaxnr + 2 ] + max_r_1s[ plotmaxnr + 4 ]) / 2.0),
             (r_0_reconstructed, r_0_reconstructed + r_1_reconstructed),
             ls='-', color='red', linewidth='2')
    # r_1 arrow
    plt.annotate(r"$r_1$",
                 xy=(
                 (max_r_1s[ plotmaxnr + 2 ] + max_r_1s[ plotmaxnr + 4 ]) / 2.0,
                 r_0_reconstructed + r_1_reconstructed / 2.0),
                 xycoords='data', xytext=(0.4, 0.93),
                 textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', shrink=0.02, width=0.4,
                                 headwidth=6.5))

# -----------------------------------------------------------------------------------------------------
# PRINTOUT
print("    Input type: {0}, Stimulation sequence: {1}".format(flag_stim_type,
                                                              flag_injection))
print("    alpha = {0}, beta = {1}, beta >? {2:2}".format(alpha, beta, factor))
# print("    Number of resonator spikes in network: {0}".format(nest.GetStatus(spikedetector, "n_events")[ 0 ]))
print("    Number of spikes from example neuron: {0}".format(
    len(spike_times_panel1)))
if flag_stim_type == 'poisson':
    print("    fraction of oscillatory drive: {0:3}".format(r_1 / r_0))
elif flag_stim_type == 'current':
    print("    fraction of oscillatory drive: {0:3}".format(I_1 / I_0))
    print("    gaussian white noise drive: {0:3}".format(I_N))
print("    Signal gain: {0:3}".format(gain))
print("    reconstructed r_0: {0:3}".format(r_0_reconstructed))
print("    reconstructed r_1: {0:3}".format(r_1_reconstructed))
if beta > factor:
    print("    expected subthreshold resonance frequency: {0:3} Hz".format(
        exp_f_r * 1000))
# print("    staionary input rate vs. current: {0:3} Sp/s | {1:3} nA".format(r_0, I_0))
# print("    oscillatory input rate vs. current: {0:3} Sp/s | {1:3} nA".format(r_1, I_1))
plt.ion()
plt.show()

# plt.plotaxis(0, rangex, 0, rangey) --> apply
# TODO: plot labels relative to data
# TODO: add annotations for poisson case
# TODO: Print relevant information into plots
# TODO: Generate vistools from plotting section

# -------------------------------------- SAVE SUMMARY TO FILE ----------------------------------------
# dd = nested_dict()
# numofsim = 0
# dd['simulation']['num'] = numofsim
# # dd['simulation']['timestamp'] =
# dd['input']['used'] = flag_stim_type
# if flag_stim_type == 'poisson':
#     dd['input']['poisson']['rate'] = r_0
#     dd['input']['poisson']['amplitude'] = r_1
#     dd['input']['poisson']['frequency'] = f
#     #dd['input']['poisson']['phase'] = 0.0
#     #dd['input']['current']['mean'] = 0.0
#     dd['input']['current']['std'] = 150.0 # I_N
#     #dd['input']['current']['phase'] = 0.0
#     dd['input']['current']['frequency'] = f
#     dd['input']['current']['amplitude'] = r_1 / g # I_1
#     dd['input']['current']['offset'] = r_0 / g # I_0
# else:
#     dd['input']['poisson']['rate'] = I_0 * g # current input does not use synweight
#     dd['input']['poisson']['amplitude'] = I_1 * g
#     dd['input']['poisson']['frequency'] = f
#     #dd['input']['poisson']['phase'] = 0.0
#     #dd['input']['current']['mean'] = 0.0
#     dd['input']['current']['std'] = I_N
#     #dd['input']['current']['phase'] = 0.0
#     dd['input']['current']['frequency'] = f
#     dd['input']['current']['amplitude'] = I_1
#     dd['input']['current']['offset'] = I_0
# dd['input']['stimlabel']= 'Richardson Fig6 low' #options: 'Richardson Fig6 high/low (equivalent)', 'manual', 'realistic'
# dd['parameters']['neuron']['g'] = g
# dd['parameters']['neuron']['g_1'] = g_1
# dd['parameters']['neuron']['tau_1'] = tau_1
# dd['parameters']['neuron']['C'] = C_m
# dd['parameters']['model']['V_th'] = V_th
# dd['parameters']['model']['V_reset'] = V_reset
# dd['parameters']['model']['E_L'] = E_L
# dd['parameters']['model']['t_ref'] = t_ref
# dd['parameters']['synapse']['synweight'] = synweight
# dd['parameters']['simulation']['N'] = N
# dd['parameters']['simulation']['dt'] = dt
# dd['parameters']['simulation']['nbins'] = nbins
# dd['parameters']['simulation']['t_sim'] = t_sim
# dd['parameters']['simulation']['t_run'] = t_run
# dd['predictions']['alpha'] = alpha
# dd['predictions']['beta'] = beta
# dd['predictions']['STR resonance occurs?'] = False # beta >? ...
# dd['predictions']['f_R'] = exp_f_r * 1000.
# dd['output']['r_0'] = r_0_reconstructed
# dd['output']['r_1'] = r_1_reconstructed
# dd['output']['gain'] = gain
# dd1 = dd.iteritems_flat()
# dd2 = dd.to_dict()
