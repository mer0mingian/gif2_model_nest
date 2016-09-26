# -*- coding: utf-8 -*-

# -------------------- MAKE STUFF LOOK NICE -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')

from mingtools.figcplots import figcplots
# from nested_dict import nested_dict
# import export_summary
# from export_summary import myprint
# ---------------------- WE WANT NEST, NEST, NEST -----------------------
import nest
import nest.raster_plot
import nest.voltage_trace
try:
    nest.Install("gif2_module")
except:
    print ("The GIf_reconstructed model had already been installed")
try:
    nest.set_verbosity('M_WARNING')
except:
    print('Changing the nest verbosity did not succeed')

#try:
#    gaindict
#except NameError:
#    gaindict = nested_dict()

# ---------- FOR FITTING THE HISTOGRAM AND COMPUTING THE GAIN -----------
import scipy.optimize as opti
import scipy.fftpack as fftpack

def mysine(t, f2, amp, off):
    return amp * np.sin(2. * np.pi * f2 * t) + off

def mysine2(t, f_reconstructed, amp, off, phi):
    return amp * np.sin(2. * np.pi * f_reconstructed * t + phi) + off

def mysine3(t, f_reconstructed, amp, off, phi):
    return amp * np.sin(2. * np.pi * f_reconstructed * t + phi) + off


def get_stim_params(flag_stim_type='current', flag_injection='Richardson_high',
                    C_m=250., g=25., g_1=25.,
                    I_0_manual=2000., I_1_manual=50., I_N_manual=100.,
                    synweight=40.):
    if flag_stim_type == 'current':
        #    I_0 = 1500.
        #    I_1 = 0.025 * I_0
        if flag_injection == 'Richardson_high':
            # correction = np.sqrt(2.)*3. #  should be np.sqrt(2.) to correct for C = 250 instead of 500
            I_N = 550.0  # Amplitude of the Gaussian White Noise in pA, 550.0
            I_0 = 780.0  # stationary component of the input current, standard: 780.0
            I_1 = 59.0  # oscillatory component of the input current, 59.0
            C_m = 500.0  # membrane capacity
            g = 25.0  # in nS, NOT µS
            g_1 = 25.0
            r_0 = g / synweight * I_0
            r_1 = g / synweight * I_1
        elif flag_injection == 'Richardson_low':
            I_N = 82.5 # 110.0  # Amplitude of the Gaussian White Noise in pA, 550.0
            I_0 = 950.0  # stationary component of the input current, standard: 780.0
            I_1 = 24.0  # oscillatory component of the input current, 59.0
            C_m = 500.0  # membrane capacity
            g = 25.0  # in nS, NOT µS
            g_1 = 25.0
            r_0 = g / synweight * I_0
            r_1 = g / synweight * I_1
        elif flag_injection == 'realistic':
            I_N = 300.0  # Amplitude of the Gaussian White Noise in pA, 550.0
            I_0 = 640.0  # stationary component of the input current, standard: 780.0
            I_1 = 25.0  # oscillatory component of the input current, 59.0
            C_m = 250.0
            r_0 = g / synweight * I_0
            r_1 = g / synweight * I_1
        elif flag_injection == 'manual':
            I_0 = I_0_manual  # Poissonian input, stationary component, spikes/s
            I_1 = I_1_manual  # Sinusoidal poissonian input amplitude, spikes/s
            I_N = I_N_manual  # Amplitude of the Gaussian White Noise in pA, 550.0
            r_0 = g / synweight * I_0
            r_1 = g / synweight * I_1
    elif flag_stim_type == 'poisson':
        if flag_injection == 'Richardson_high':
            # correction = np.sqrt(2.)*3. #  should be np.sqrt(2.) to correct for C = 250 instead of 500
            I_N = 550.0  # Amplitude of the Gaussian White Noise in pA, 550.0
            I_0 = 780.0  # stationary component of the input current, standard: 780.0
            I_1 = 59.0  # oscillatory component of the input current, 59.0
            C_m = 500.0  # membrane capacity
            g = 25.0  # in nS, NOT µS
            g_1 = 25.0
            r_0 = g / synweight * I_0
            r_1 = g / synweight * I_1
        elif flag_injection == 'Richardson_low':
            I_N = 110.0  # Amplitude of the Gaussian White Noise in pA, 550.0
            I_0 = 950.0  # stationary component of the input current, standard: 780.0
            I_1 = 24.0  # oscillatory component of the input current, 59.0
            C_m = 500.0  # membrane capacity
            g = 25.0  # in nS, NOT µS
            g_1 = 25.0
            r_0 = g / synweight * I_0
            r_1 = g / synweight * I_1
        elif flag_injection == 'realistic':
            I_N = 300.0  # Amplitude of the Gaussian White Noise in pA, 550.0
            I_0 = 640.0  # stationary component of the input current, standard: 780.0
            I_1 = 25.0  # oscillatory component of the input current, 59.0
            C_m = 250.0
            r_0 = g / synweight * I_0
            r_1 = g / synweight * I_1
        elif flag_injection == 'manual':
            r_0 = r_0_manual  # Poissonian input, stationary component, spikes/s
            r_1 = r_1_manual  # Sinusoidal poissonian input amplitude, spikes/s
            I_0 = r_0 / g
            I_1 = r_1 / g
            I_N = 300.
    return I_0, I_1, I_N, r_0, r_1, C_m, g, g_1


print('So it begins...')

# -------------------------------------- HERE STARTS THE SCRIPT CONTENT ----------------------------------------------


nsims = 2
flag_injections = ['Richardson_high', 'Richardson_low']
normaliser = np.zeros(len(flag_injections))

# NEURON PARAMETERS
tau_1 = 100.  # 100.0  # time constant for 2nd variable, ms
C_m = 250.0  # membrane capacity, pF
g_1 = 89.466  # conductance 2nd variable, nS, 30.0
g = 10.  # conductance 1st variable, nS, 6.5
t_ref = 2.0  # length refractory period, ms
E_L = 0.0  # voltage leak term, mV
V_th = 20.0  # Spiking threshold, mV
V_reset = 14.0  # reset current, mV

# STIMULATION PARAMETERS
wfactor = 1.
r_0_manual = 7000.  # 7500.0 * 40.0 / 87.8
r_1_manual = 120.  # 150.0
I_0_manual = 1800.
I_1_manual = 26.
I_N_manual = 100.0  # 150.0  # Amplitude of the Gaussian White Noise in pA, 550.0

# SIMULATION PARAMETERS
t_run = 3000.0  # total length of simulationb
t_recstart = 500.0  # time to begin recording, usually 350.0
dt = 1.0  # Recording timestep
N = 3500  # Number of trials/parallel unconnected neurons
nbins = 100.0  # number of bins for histogram
synweight = 40.0  # synaptic weight, 87.8

flag_annotate = False

# -------------------------------------- HERE STARTS THE LOOPING OVER PROTOCOLS --------------------------------------


for j in flag_injections: #loop over stimulation protocols
    print('Starting protocol {0}'.format(j))
    gains = np.zeros((nsims, 2))
    frequencies = np.hstack( ( np.array( [ 0 ] ), np.logspace( -1., 2., num=nsims-1) ) )

    for l in np.arange(0,nsims): # loop for data points
        f = frequencies[ l ]  # The frequency of the drive, Hz
        nest.ResetKernel()
        nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
        try:
            nest.set_verbosity('M_WARNING')
        except:
            print('Changing the nest verbosity did not succeed')
        print('\n setting up simulation {0}: f = {1}'.format(l, frequencies[ l ]))
        flag_stim_type = 'current'  # options: poisson, current
        flag_injection = j # options: Richardson_low, Richardson_high, realistic, manual

        # SET ALL THE PARAMETERS
        I_0, I_1, I_N, r_0, r_1, C_m, g, g_1 = get_stim_params(
            flag_stim_type, flag_injection, C_m, g, g_1,
            I_0_manual, I_1_manual, I_N_manual, synweight)


        # CALCULATE SECOND-ORDER PARAMETERS
        alpha = g * tau_1 / C_m  # regime determining constant 1
        beta = g_1 * tau_1 / C_m  # regime determining constant 2
        factor = np.sqrt((alpha + 1.)*(alpha + 1.) + 1.) - (1. + alpha)
        t_sim = t_run + t_recstart  # length of data acquisition time
        exp_f_r = 1. / (2. * np.pi * tau_1) * np.sqrt(np.sqrt((alpha + beta + 1.)*(alpha + beta + 1.) - (alpha + 1.)*(alpha + 1.) ) - 1.)
        # formula A6, p.2552

        # DEVICES
        # see parameters from stimulation protocol in Fig. 6, p2546 in Richardson et al. (2003)
        neuron = nest.Create("gif2_psc_exp", n=N, params={
            "tau_1": tau_1,
            "C_m": C_m,
            "g_rr": g_1,
            "g": g,
            "V_m": 0.0,
            "V_reset": V_reset,
            "E_L": E_L,
            "V_th": V_th})
        if f != 0:
            if flag_stim_type == 'poisson':
                stimparams_r = [ {'rate': r_0,
                              'amplitude': r_1,
                              'frequency': f,
                              'phase': 0.0} ]
                stim_r = nest.Create('sinusoidal_poisson_generator', params=stimparams_r)
            elif flag_stim_type == 'current':
                stimparams_I = [ {'offset': I_0,
                                  'amplitude': I_1,
                                  'frequency': f,
                                  'phase': 0.0} ]
                stim_I = nest.Create('ac_generator', params=stimparams_I)
        elif f == 0:
            if flag_stim_type == 'poisson':
                stimparams_r = [ {'rate': r_0,
                                  'amplitude': 0.0,
                                  'frequency': 1.,
                                  'phase': 0.0} ]
                stim_r = nest.Create('sinusoidal_poisson_generator', params=stimparams_r)
            elif flag_stim_type == 'current':
                stimparams_I = [ {'offset': I_0,
                                  'amplitude': 0.0,
                                  'frequency': 1.,
                                  'phase': 0.0} ]
                stim_I = nest.Create('ac_generator', params=stimparams_I)
        stimparams_xeta = [ {'mean': 0.0,
                                 'std': I_N,
                                 'dt': dt,
                                 'frequency': 0.0,
                                 'phase': 0.0,
                                 'std_mod': 0.0} ]
        stim_xeta = nest.Create('noise_generator', params=stimparams_xeta)
        spikedetector = nest.Create('spike_detector')
        nest.SetStatus(spikedetector, {"withgid": True,
                                       "withtime": True,
                                       "start": t_recstart})
        multimeter = nest.Create("multimeter",
                                 params={"interval": dt,
                                         "record_from": [ "V_m", "w" ],
                                         "withgid": True})

        # BUILD NETWORK
        if flag_stim_type == 'poisson':
            nest.Connect(stim_r, neuron, syn_spec={'weight': wfactor * synweight})
        elif flag_stim_type == 'current':
            nest.Connect(stim_I, neuron)
            nest.Connect(stim_xeta, neuron)
        nest.Connect(multimeter, neuron)
        nest.Connect(neuron, spikedetector)


        # SIMULATE
        nest.Simulate(t_sim)
        # -----------------------------------------------------------------------------------------------------
        # RECORDING
        multi_senders = nest.GetStatus(multimeter, "events")[ 0 ][ "senders" ]  # extract the senders for each event in V_m's
        multi_voltage = nest.GetStatus(multimeter, "events")[ 0 ][ "V_m" ]  # extract the voltages
        multi_times = nest.GetStatus(multimeter, "events")[ 0 ][ "times" ]
        spikes = nest.GetStatus(spikedetector, "n_events")[ 0 ]
        spike_senders = nest.GetStatus(spikedetector, "events")[ 0 ]["senders"]
        spike_times = nest.GetStatus(spikedetector, "events")[ 0 ]["times"]
        spikes_panel1 = spike_times[ spike_senders == 1 ]

        # ---------------------------------------------- PANEL 1 ------------------------------------------------
        # Generate the histogram
        if len(spike_times) > 1:
            hist_binwidth = t_run / nbins  # !> 8.0 for the instantaneous rate
            t_bins = np.arange( np.amin(spike_times), np.amax(spike_times), float(hist_binwidth) )
            n, bins = np.histogram(spike_times, bins=t_bins)
            # n, bins = nest.raster_plot._histogram(spike_times, bins=t_bins)  # uses the NEST histogram function. see below!
            # heights = 1000. * binned_spikes / (hist_binwidth * N)
            heights = 1000. * n / (hist_binwidth * N)

            # -----------------------------------------------------------------------------------------------------
            # mean firing rate
            r_0_reconstructed = np.mean(heights)
            # -----------------------------------------------------------------------------------------------------
            # firing rate modulation
            f_reconstructed = f / 1000.
            amp_guess = (max(heights) + abs(min(heights))) / 2. - r_0_reconstructed
            phase = 0.
            print('f_reconstructed = {0:.2f}, amp_guess = {1:.2f}, r0_reconstructed = {2:.2f}'.format(f_reconstructed*1000, amp_guess, r_0_reconstructed))
            try:
                bins = bins[1:]
                popt, pcov = opti.curve_fit(mysine, bins, heights, p0=(f_reconstructed, amp_guess, r_0_reconstructed))
                sinecurve = mysine(bins - hist_binwidth, f_reconstructed, popt[ 1 ], popt[ 2 ])
                if f == 0:
                    exp_r_0 = popt[ 2 ]
                # this is the alternative version including he phase:
                popt2, pcov2 = opti.curve_fit(mysine2, bins, heights, p0 = (f_reconstructed, amp_guess, r_0_reconstructed, phase))
                sinecurve = mysine2( bins - hist_binwidth, f_reconstructed, popt2[1], popt2[2], popt2[3] )
                # popt: freqeuncy, amplitude, offset, phase
                # -----------------------------------------------------------------------------------------------------
                # calculate the signal gain:
                r_1_reconstructed = max(abs(sinecurve - abs(popt[2])))
                gain = abs(r_1_reconstructed / I_1)
            except RuntimeError:
                print('optimal parameters not found')
                gain = 0.
                popt = np.zeros(4)
                popt2[ 3 ] = 0
        else:
            print('not enough spikes detected')
            gain = 0.
            r_1_reconstructed = 0.
            r_0_reconstructed = 0.
            popt = np.zeros(3)
            popt2[ 3 ] = 0
        gains[ l, 0 ] = gain
        gains[ l, 1 ] = popt2[ 3 ]
        print('Computed a gain amplitude: {0:.3f} and phase {1:.3f}'.format(gains[l, 0], gains[l,1]))

        if flag_injection == 'Richardson_low':
            gains_low = np.zeros_like(gains)
            gains_low = gains
        elif flag_injection == 'Richardson_high':
            gains_high = np.zeros_like(gains)
            gains_high = gains

# -----------------------------------------------------------------------------------------------------
#normalise the gains at zer0 frequency
#if gains[0, 0] != 0:
#    gains[:, 0] /= gains[0, 0]
#else:
#    print('Normalising not successful')
#plt.clf()

# plt.subplot(211)
# if 'high' in flag_injection:
#     col = 'blue'
#     plt.title('High noise')
# elif 'low' in flag_injection:
#     col = 'red'
#     plt.title('Low  noise')
# plt.semilogx(frequencies, gains[:,0], color=col, marker='o', ls=' ')
# plt.grid(True)
# plt.text(0.15, 1.3, 'C = {0}, g = {1}, g1 = {2}, t1 = {3}'.format(C_m, g, g_1, tau_1))
# plt.axvline(x=exp_f_r*1000)
# plt.xlabel('Normalised amplitude of signal gain', size=9.)
# plt.ylabel('Frequency [Hz]', size=9.)
# plt.subplot(212)
# plt.grid(True)
# plt.semilogx(frequencies, gains[:,1], color=col, marker='o', ls=' ')
# plt.xlabel('Phase of signal gain', size=9.)
# plt.ylabel('Frequency [Hz]', size=9.)
# plt.axvline(x=exp_f_r*1000)
# plt.ion()

gains_low2 = gains_low
gains_high2 = gains_high
gains_low2[:,0] /= gains_low[1,0]
gains_high2[:,0] /= gains_high[1,0]




# -----------------------------------------------------------------------------------------------------
# Must come at the end for default values to be defined
# Call to plot all conditions together
# def figcplots(gains_low, gains_high, frequencies, exp_f_r, exp_r_0):
#     plt.subplot(211)
#     plt.semilogx(frequencies[1:], gains_high[ 1:, 0 ], color='blue', marker='o', ls=' ')
#     plt.semilogx(frequencies[1:], gains_low[ 1:, 0 ], color='red', marker='o', ls=' ')
#     plt.grid(True)
#     plt.text(0.15, 2.3, 'C = {0}pF, g = {1}nS, g1 = {2}nS, t1 = {3}ms'.format(C_m, g, g_1, tau_1))
#     plt.axvline(x=exp_f_r*1000)
#     plt.axvline(x=exp_r_0)
#     plt.xlabel('Frequency [Hz]', size=9.)
#     plt.ylabel('Normalised amplitude of signal gain', size=9.)
#     plt.ylim([0, 2.5])
#     plt.subplot(212)
#     plt.grid(True)
#     plt.semilogx(frequencies[1:], gains_high[ 1:, 1 ], color='blue', marker='o', ls=' ')
#     plt.semilogx(frequencies[1:], gains_low[ 1:, 1 ], color='red', marker='o', ls=' ')
#     plt.xlabel('Frequency [Hz]', size=9.)
#     plt.ylabel('Phase of signal gain', size=9.)
#     plt.ylim([ -10, 10 ])
#     plt.axvline(x=exp_f_r*1000)
#     plt.axvline(x=exp_r_0)
#     plt.ion()
#     plt.show()


# Write Richardson?
# Is the hierarchy dynamic, for either case of alpha generator
# Use linear integration