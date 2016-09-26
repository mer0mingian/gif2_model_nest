# -*- coding: utf-8 -*-

import nest
import nest.raster_plot
import nest.voltage_trace
import numpy as np
import matplotlib.pyplot as plt
nest.ResetKernel()
try:
	nest.Install("gif2_module")
except:
	print ("The GIF2 model had already been installed")
import scipy.optimize as opti
import scipy.fftpack as fftpack
def mysine(x, a1, a2, a3):
    return a1 * np.sin(a2 * x + a3)
nest.set_verbosity('M_WARNING')



# Stimulation Protocol
t_run = 3000.0       # total length of simulation
t_recstart = 500.0  # time to begin recording
dt = 1.0            # Recording timestep
N = 3500             # Number of trials/parallel unconnected neurons
t_sim = t_run + t_recstart   # length of data acquisition time
nbins = 500.0

# Setup kernel
nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
intervals = 3
t_stim = np.ones(intervals) * 1000.

# NEURON PARAMETERS
tau_1 = 100.0 # time constant for 2nd variable
C_m = 500.0 # membrane capacity
g_1 = 25.0 # conductance 2nd variable
g = 25.0 # conductance 1st variable
t_ref = 2.0 # length refractory period
synweight = 87.8 # synaptic weight, 87.8
alpha = g * tau_1 / C_m     # regime determining constant 1
beta = g_1 * tau_1 / C_m    # regime determining constant 2
factor = (alpha - 1)**2 / 4 # regime total factor. factor > beta for resonance
K = 1.
E_L = 0.0
V_reset = 14.0
V_th = 20.0
r_rates = np.ones(intervals)
r_amplitudes = np.ones(intervals)

# STIMULATION PROTOCOL
"""
Low-noise case: I_0 = 0.95, I_1 = 0.0024 => I1/I0 = 0.0253, regular spiking, high modulation at 20Hz
High-noise case: I_0 = 0.78, I=1 = 0.059 => I1/I0 = 0.0756, irregular spiking, high modulation at 5Hz
Regarding the constant 20Hz rate I_0 in the paper: J_orig = 0.11 nA, nu_orig = 20 Hz
"""
# flag_stim_type = 'current'  # options: poisson, current
flag_injection = 'Richardson high' # options: Richardson, realistic
if ('Richardson' in flag_injection) | ('realistic' in flag_injection):
    flag_stim_type = 'current'
elif 'poisson' in flag_injection:
    flag_stim_type = 'poisson'
else:
    flag_stim_type = 'current'

if flag_injection == 'Richardson high':
    I_N = 550.0  # Amplitude of the Gaussian White Noise in pA, 550.0
    I_0 = 780.0  # stationary component of the input current, standard: 780.0
    I_1 = 59.0  # oscillatory component of the input current, 59.0
    C_m = 500.0
elif flag_injection == 'Richardson low':
    I_N = 110.0  # Amplitude of the Gaussian White Noise in pA, 110.0
    I_0 = 950.0  # stationary component of the input current, standard: 950.0
    I_1 = 24.0  # oscillatory component of the input current, 24.0
    C_m = 500.0
elif flag_injection == 'realistic low':
    I_N = 82.5  # Amplitude of the Gaussian White Noise in pA, 110.0
    I_0 = 945.0  # stationary component of the input current, standard: 950.0
    I_1 = 24.0   # oscillatory component of the input current, 24.0
    C_m = 500.0
#elif flag_injection == 'realistic high':
#    I_N = 550.0  # Amplitude of the Gaussian White Noise in pA, 550.0
#    I_0 = 780.0  # stationary component of the input current, standard: 780.0
#    I_1 = 30.0  # oscillatory component of the input current, 59.0
#    C_m = 500.0
elif flag_injection == 'poisson low':
    I_N = 690.0  # Amplitude of the Gaussian White Noise in pA, 550.0
    I_0 = 600.0  # stationary component of the input current, standard: 780.0
    I_1 = 59.0   # oscillatory component of the input current, 59.0
    r_rates = np.ones(intervals) * 5000.
    r_amplitudes = r_rates * 0.155
elif flag_injection == 'poisson high':
    I_N = 690.0  # Amplitude of the Gaussian White Noise in pA, 550.0
    I_0 = 600.0  # stationary component of the input current, standard: 780.0
    I_1 = 59.0   # oscillatory component of the input current, 59.0
    r_rates = np.ones(intervals) * 5000.
    r_amplitudes = r_rates * 0.02

r_frequencies = np.array([0., 5., 20.])
r_stim = {'rates': r_rates,
          'amplitudes': r_amplitudes,
          'frequencies': r_frequencies}


# Neuron spiking characteristics
J = synweight / g
I0_calc = K *  J * r_stim["rates"]
I1_calc =  K *  J * r_stim["amplitudes"]

# DEFINE DEVICES
"""  see parameters from stimulation protocol in Fig. 6, p2546 in Richardson et al. (2003)
This configuration shows the STR with I_stim = 1500.0 over t_sim_stim = 12.0 + 350.0:
neuron = nest.Create("gif2_psc_exp", params = {
    "tau_1": tau_1, "C_m": C_m, "g_rr": g_1, "g": g, "V_m": 0.0, "V_reset": 14.0, "E_L": 0.0, "V_th": 20.0}) """
neuron = nest.Create("gif2_psc_exp", n=N, params = {
    "tau_1": tau_1,
    "C_m": C_m,
    "g_rr": g_1,
    "g": g,
#    "V_m": 0.0,
    "V_reset": V_reset,
    "E_L": E_L,
    "V_th": V_th
})

# stimulation devices
#stim = nest.Create('dc_generator')
stimparams_r={'rate': r_stim['rates'][0],
              'amplitude': r_stim['amplitudes'][0],
              'frequency': r_stim['frequencies'][0]}
stimparams_I = [ {'offset': I_0,
                  'amplitude': I_1,
                  'frequency': r_stim['frequencies'][0],
                  'phase': 0.0} ]
stimparams_xi = [ {'mean': 0.0,
                     'std': I_N,
                     'dt': dt,
                     'frequency': 0.0,
                     'phase': 0.0,
                     'std_mod': 0.0} ]

if flag_stim_type == 'poisson':
    stim2 = nest.Create('sinusoidal_poisson_generator', params=stimparams_r)
elif flag_stim_type == 'current':
    stim_I = nest.Create('ac_generator', params=stimparams_I)
    stim_xi = nest.Create('noise_generator', params=stimparams_xi)

# recording devices
voltmeter = nest.Create('voltmeter')
nest.SetStatus(voltmeter, {"withgid": True, "withtime": True, "interval": dt, "start": t_recstart})
spikedetector = nest.Create('spike_detector')
nest.SetStatus(spikedetector,[{"withtime": True, "withgid": True, "to_file": False, "start": t_recstart}])

# CONNECT
if flag_stim_type == 'poisson':
    nest.Connect(stim2, neuron, syn_spec={'weight': synweight})
elif flag_stim_type == 'current':
    nest.Connect(stim_I, neuron)
    nest.Connect(stim_xi, neuron)
nest.Connect(voltmeter, neuron)
nest.Connect(neuron, spikedetector)

# SIMULATE
nest.Simulate(t_recstart)
if flag_stim_type == 'poisson':
    for i in range(0, intervals):
        print('stimulating sequence ' + str(i + 1))
    #    nest.SetStatus(stim, {'amplitude': I_stim[i]})
        nest.SetStatus(stim2, {'rate' : r_stim['rates'][i],
                            'amplitude' : r_stim['amplitudes'][i],
                            'frequency' : r_stim['frequencies'][i]})
        nest.Simulate(t_stim[i])
elif flag_stim_type == 'current':
    for i in range(0, intervals):
        print('stimulating sequence ' + str(i + 1))
        nest.SetStatus(stim_I, {'frequency' : r_stim['frequencies'][i]})
        nest.Simulate(t_stim[i])


# RECORD
senders = nest.GetStatus(voltmeter, "events")[0]["senders"]
potentials_all = nest.GetStatus(voltmeter, "events")[0]["V_m"]
print('    finding the voltage trace of the example neuron')
potentials = potentials_all[senders == 1]
times_all = nest.GetStatus(voltmeter, "events")[0]["times"]
print('    finding all spikes of the example neuron')
times = times_all[senders == 1]
times_panel2 = nest.GetStatus(spikedetector, "events")[0]["times"]
gids_panel2 = nest.GetStatus(spikedetector, "events")[0]["senders"]
spikes_panel1 = times_panel2 [ gids_panel2 == 1 ]
hist_binwidth = (t_sim - t_recstart) / nbins # !> 8.0 for the instantaneous rate
t_bins = np.arange(
	np.amin(times_panel2), np.amax(times_panel2),
	float(hist_binwidth)
)


fig = plt.figure("Figure 6 A/B")
plt.clf()
# nest.voltage_trace.from_device(voltmeter)
print('    creating the voltage trace')
ax = fig.add_subplot(211)
ax.set_xlim([ t_recstart, t_sim ])
ax.set_ylim([ -70.0, 40.0 ])
ax.set_ylabel('V_m [mV]', size=12)
ax.plot(times, potentials - 70. + E_L)
y_bar_lims = np.array([V_th -70.0 + E_L, 10.75])
for i in spikes_panel1:
    ax.vlines(x = times[ times == i ] - dt,
               ymin = y_bar_lims[0],
               ymax = y_bar_lims[1],
               colors = 'blue')
ax2 = plt.twinx()
if flag_stim_type == 'current':
    t_stimcum = t_stim.cumsum()
    t_scr = np.array(t_stimcum, dtype=int)
    talt = np.array(np.roll(t_stimcum, 1), dtype=int)
    # talt = talt.sint()
    currentdrive = np.ones_like(times) * I_0
    for i in range(1, intervals):
        addosc = mysine(times[t_scr[i-1]-1 :  t_scr[i]-1]/1000., # np.arange(0,t_stim[i]),
                        r_stim['amplitudes'][i]*I_1,
                        np.pi*2*r_stim['frequencies'][i],
                        0.)
        # currentdrive[ int(talt[ i ]):int(talt[ i ]) + len(addosc) +1] *= r_stim[ 'rates' ][ i ]
        # currentdrive[int(talt[i]):int(talt[i])+len(addosc)] += addosc
        # the above line does not work yet
        currentdrive[ talt[i]-1 : talt[i]+len(addosc)-1] += addosc
    ax2.plot(times, currentdrive, color = 'black')
    ax2.set_xlim([ t_recstart, t_sim ])
    ax2.set_ylim([ I_0 - 12.5 * I_1, I_0 + 2.5 * I_1])
    ax2.set_ylabel('I_E [nA]', size=12)

ax3= fig.add_subplot(212)
print('    creating the histogram')
n, bins = nest.raster_plot._histogram(times_panel2, bins=t_bins) # uses the NEST histogram function. see below!
num_neurons = len(np.unique(gids_panel2))
heights = N * n / (hist_binwidth * num_neurons) # population firing rate
if 'high' in flag_injection:
    col = 'blue'
    fig.suptitle('High noise', fontsize=20)
elif 'low' in flag_injection:
    col = 'red'
    fig.suptitle('Low noise', fontsize=20)
ax3.bar(t_bins, heights, width=hist_binwidth, color=col, edgecolor=col)
ax3.set_yticks([int(x) for x in np.linspace(0.0, int(max(heights) * 1.1) + 5, 4)])
ax3.set_ylabel("rate [Hz]", size=12)
ax3.set_xlabel("time [ms]", size=12)
ax3.set_xlim([t_recstart, t_sim])

print "    alpha = ", alpha, "  , beta = ", beta,  " , beta >? ", factor
print("    Number of spikes: {0}".format(nest.GetStatus(spikedetector, "n_events")[0]))
print("    rate1: {0}".format(heights[0:int(len(heights)/3)].mean()))
print("    rate2: {0}".format(heights[int(len(heights)/3):int(2*len(heights)/3)].mean()))
print("    rate3: {0}".format(heights[int(2*len(heights)/3):len(heights)].mean()))
plt.ion()
