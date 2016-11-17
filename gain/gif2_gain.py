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
import numpy as np
import scipy.optimize as opti
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
rc('mathtext', default='regular')


# -----------------------------------------------------------------------------
# Mathematical functions
# -----------------------------------------------------------------------------


class sinus:
    """
	This class allows for initialisation of a fittable sine function with a
	frequency fixed at the stimulation frequency.
	"""

    def __init__(self, freq):
        self.f = freq

    def fit(self, t, amp, off, phi):
        """
		Sine function for fitting amplitude, frequency, offset and phase.
		:return: sine function with given parameters
		"""
        return amp * np.sin(2. * np.pi * self.f * t + phi) + off


# -----------------------------------------------------------------------------
# Parser functions for handling parameters
# -----------------------------------------------------------------------------


def get_stim_params(paramdict, f, condition, dt=0.1):
    """
	This function invokes the stimulation parameters to ensure that all
	 stimulation devices can be called apropriately. It also sets the neuron
	 parameters.
	It only requires the strings for "current/poisson" and
	"Richardson_high/low
	/manual/realistic."
	:return: I_0, I_1, I_N, r_0, r_1, C_m, g, g_1

	flag_stim_type='current', flag_injection='Richardson_high',
					C_m=250., g=25., g_1=25.,
					I_0_manual=2000., I_1_manual=50., I_N_manual=100.,
					synweight=40.
	"""
    phase = 0.0
    ac_phase = -(
        ((float(paramdict[ 't_recstart' ]) - dt) / 1000. * f) % 1) * 360
    if condition is not None:
        if paramdict[ 'stimtype' ] == 'current':
            r_stimdict = dict()
            if condition == 'r':
                xi_stimdict = dict(std=82.5 / np.sqrt(dt),
                                   dt=dt,
                                   phase=phase)
                I_stimdict = dict(offset=950.0,
                                  amplitude=24.0,
                                  frequency=f,
                                  phase=ac_phase)
            elif condition == 'R':
                xi_stimdict = dict(std=550.0 / np.sqrt(dt),
                                   dt=dt,
                                   phase=phase)
                I_stimdict = dict(offset=780.0,
                                  amplitude=59.0,
                                  frequency=f,
                                  phase=ac_phase)
            elif condition == 'm':
                xi_stimdict = dict(std=paramdict[ 'I_N' ] / np.sqrt(dt),
                                   dt=dt,
                                   phase=0.0, )
                I_stimdict = dict(offset=paramdict[ 'I_0' ],
                                  amplitude=paramdict[ 'I_1' ],
                                  frequency=f,
                                  phase=ac_phase)
        elif paramdict[ 'stimtype' ] == 'poisson':
            I_stimdict = dict()
            xi_stimdict = dict()
            r_stimdict = dict()
            print('poisson input not implemented yet')
    else:
        print('no stimulation protocol given!')
        I_stimdict = dict()
        xi_stimdict = dict()
        r_stimdict = dict()
    return I_stimdict, r_stimdict, xi_stimdict


def write_results(resultdict):
    """
	exports the results for a single datapoint to the results file.
	"""
    index = resultdict[ 'simparameterdict' ][ 'simindex' ]
    filename = 'gains_{0}.csv'.format(int(index))
    myCsvRow = str('{0}:{1}:{2}:{3}\n'.format(resultdict[ 'freqindex' ],
                                              resultdict[ 'condition' ],
                                              resultdict[ 'gain' ][ 0 ],
                                              resultdict[ 'gain' ][ 1 ]
                                              )
                   )
    fd = open(filename, 'a')
    fd.write(myCsvRow)
    fd.close()


def save_raw_data(resultdict, spike_times, spike_senders):
    sendersfile = 'raw_data/gains_{0}_raw_times_{1}_{2}.csv'.format(
            resultdict[ 'simparameterdict' ][ 'simindex' ],
            resultdict[ 'freqindex' ],
            resultdict[ 'condition' ])
    timesfile = 'raw_data/gains_{0}_raw_senders_{1}_{2}.csv'.format(
            resultdict[ 'simparameterdict' ][ 'simindex' ],
            resultdict[ 'freqindex' ],
            resultdict[ 'condition' ])
    with open(sendersfile, 'a') as resultfile:
        resultfile.write(str(spike_senders.tolist()))
    with open(timesfile, 'a') as resultfile:
        resultfile.write(str(spike_times.tolist()))


def import_params_as_dict(filename='jobdict.txt'):
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


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------


def import_gain(filename='gains.csv'):
    """
	imports the gain matrix from a given file
	"""
    conditions = [ ]
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line:
                freq, cond, gain_amp, gain_phase = line.strip().split(':')
            else:
                freq, cond, gain_amp, gain_phase = 0., 0., 0., 0.
            if cond not in conditions:
                conditions.append(cond)
                print('condition {0} coded as {1}'.format(cond,
                                                          conditions.index(
                                                                  cond)))
            ind = conditions.index(cond)
            try:
                gain = np.vstack((gain,
                                  np.array([ ind, freq, gain_amp,
                                             gain_phase ],
                                           dtype=float)))
            except NameError:
                gain = np.array([ ind, freq, gain_amp, gain_phase ],
                                dtype=float)
    gain = sort_gain(gain)
    return gain, conditions


def sort_gain(gainmat):
    """
	sorts the matrix with gains such that the first component is sorted by
	noise conditions and the second component by ascending stimulation
	frequencies.
	"""
    conditions = np.unique(gainmat[ :, 0 ])
    gainmat = gainmat[ gainmat[ :, 0 ].argsort() ]
    for cond in conditions:
        submat = gainmat[ gainmat[ :, 0 ] == cond ]
        submat = submat[ submat[ :, 1 ].argsort() ]
        gainmat[ gainmat[ :, 0 ] == cond ] = submat
    return gainmat


def split_gain_conditions(gainmat):
    """
	Separate the conditions for low and high noise in the gain matrix
	"""
    frequencies = np.hstack((np.array([ 0 ]),
                             np.logspace(-1., 2., num=len(
                                     np.unique(gainmat[ :, 1 ])) - 1)))
    freqinddict = dict()
    freqdict = dict()
    for i in np.unique(gainmat[ :, 0 ]):
        freqinddict[ i ] = np.array(gainmat[ gainmat[ :, 0 ] == i ][ :, 1 ],
                                    dtype=int)
        freqdict[ i ] = frequencies[ freqinddict[ i ] ]
    # extract the indices of successful computed gain values
    freqs_high_ind = np.array(gainmat[ gainmat[ :, 0 ] == 0 ][ :, 1 ],
                              dtype=int)
    freqs_low_ind = np.array(gainmat[ gainmat[ :, 0 ] == 1 ][ :, 1 ],
                             dtype=int)

    # find the frequencies for these indices
    freqs_low = frequencies[ freqs_low_ind ]
    freqs_high = frequencies[ freqs_high_ind ]

    # find the gains
    gains_high = gainmat[ gainmat[ :, 0 ] == 0 ][ :, 2: ]
    gains_low = gainmat[ gainmat[ :, 0 ] == 1 ][ :, 2: ]

    plotdict = dict(gains_low=gains_low,
                    gains_high=gains_high,
                    freqs_low=freqs_low,
                    freqs_high=freqs_high)
    return plotdict


def gain_normaliser(gaindict, normalise=True, withone=False, **kwargs):
    gains_low = gaindict[ 'gains_low' ]
    gains_high = gaindict[ 'gains_high' ]

    # Correct the phases for periodicity
    add_360 = kwargs.pop('add_360', 'False')
    if add_360:
        gains_high[ :, 1 ] += 360.0
        gains_low[ :, 1 ] += 360.0
    # while np.sum(gains_high >= 180.0) > 0 or np.sum(gains_high <= -180.0)
    # > 0:
    # 	gains_high[ gains_high >= 180.0 ] -= 360.0
    # 	gains_high[ gains_high <= -180.0 ] += 360.0
    # 	gains_low[ gains_low >= 180.0 ] -= 360.0
    # 	gains_low[ gains_low <= -180.0 ] += 360.0

    # smooth the phases
    smooth_before = kwargs.pop('smooth_before', 'False')
    if smooth_before:
        gains_high = smooth_gain_phase_period(gains_high)
        gains_low = smooth_gain_phase_period(gains_low)

    # normalise?
    if normalise and not withone:
        gains_high[ :, 0 ] /= gains_high[ 0, 0 ]
        gains_high[ :, 1 ] /= gains_high[ 0, 1 ]
        gains_low[ :, 0 ] /= gains_low[ 0, 0 ]
        gains_low[ :, 1 ] /= gains_low[ 0, 1 ]
    elif normalise and withone:
        gains_high[ :, 0 ] /= gains_high[ 1, 0 ]
        gains_high[ :, 1 ] /= gains_high[ 1, 1 ]
        gains_low[ :, 0 ] /= gains_low[ 1, 0 ]
        gains_low[ :, 1 ] /= gains_low[ 1, 1 ]

    smooth_after = kwargs.pop('smooth_after', 'False')
    if smooth_after:
        # smooth the phases again
        gains_high = smooth_gain_phase_period(gains_high)
        gains_low = smooth_gain_phase_period(gains_low)

    gaindict2 = dict(gains_high=gains_high, gains_low=gains_low,
                     freqs_low=gaindict[ 'freqs_low' ],
                     freqs_high=gaindict[ 'freqs_high' ])
    return gaindict2


def smooth_gain_phase_period(gains_sep):
    """
	Smoothing the large angle jumps in the gain phase
	:param gain_sep: array(amplitudes, phases)
	:return: array(amplitudes, smooth phases)
	"""
    gains_sep[ :, 1 ] = np.deg2rad(gains_sep[ :, 1 ])
    gains_sep[ :, 1 ] = np.unwrap(gains_sep[ :, 1 ])
    gains_sep[ :, 1 ] = np.rad2deg(gains_sep[ :, 1 ])
    return gains_sep


# -----------------------------------------------------------------------------
# Visualise the results
# -----------------------------------------------------------------------------

def show_gain(plotdict, save=True):
    """
	creates the fig6 plots from richardson et al., 2003
	"""
    gains_low = plotdict[ 'gains_low' ]
    gains_high = plotdict[ 'gains_high' ]
    freqs_low = plotdict[ 'freqs_low' ]
    freqs_high = plotdict[ 'freqs_high' ]

    # plot the gains
    fig = plt.figure("Noise-dependent gain of the GIF2")
    plt.subplot(211)
    plt.semilogx(freqs_high[ 1:-2 ], gains_high[ 1:-2, 0 ], color='blue',
                 marker='o', ls=' ')
    plt.semilogx(freqs_low[ 1:-2 ], gains_low[ 1:-2, 0 ], color='red',
                 marker='o', ls=' ')
    plt.grid(True)
    # plt.xlabel('Frequency [Hz]', size=9.)
    plt.ylabel('Normalised amplitude of signal gain', size=9.)
    plt.xlim([ -1., 100. ])

    plt.subplot(212)
    plt.grid(True)
    plt.semilogx(freqs_high[ 1:-2 ], gains_high[ 1:-2, 1 ], color='blue',
                 marker='o', ls=' ')
    plt.semilogx(freqs_low[ 1:-2 ], gains_low[ 1:-2, 1 ], color='red',
                 marker='o', ls=' ')
    plt.xlabel('Frequency [Hz]', size=9.)
    plt.ylabel('Normalised phase of signal gain [deg]', size=9.)
    plt.xlim([ -1., 100. ])
    if save == True:
        # exp_f_r, exp_r_0, C_m, g, g_1, tau_1:
        paramdict = import_params_as_dict('jobdict.txt')
        plt.savefig(
            'Richardson_Fig6CD_{0}.png'.format(paramdict[ 'simindex' ]))
    else:
        plt.ion()
        plt.show()
    return fig


def multiplot(condition, queueid, freqindex, bins, bars, f, amp, phase,
              offset):
    """
	create a multiplot, each showing a fit of a single datapoint.
	"""
    plotindex = np.array(np.linspace(0, maxindex, 15), dtype=int)
    mysine = sinus(f / 1000.)
    sineplot = mysine.fit(bins, amp, offset, phase)

    if freqindex in plotindex:
        outputname = 'multiplots_{0}_{1}.pdf'.format(queueid, condition)
        histofit = plt.figure('Gain multiplots')
        ax = plt.subplot2grid((4, 4), (int(freqindex / 4), freqindex % 4))
        ax.hist(bars, bins)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        if int(freqindex / 4) != 3:
            ax.set_yticklabels(( ))
        if (freqindex % 4) != 0:
            ax.set_xticklabels(( ))
        ax.plot(bins, sineplot)
        pp = PdfPages(outputname)
        pp.savefig(histofit, dpi=600, transparent=True)
        pp.close()

        outputname_single = 'singleplots_{0}_{1}_{2}.png'.format(queueid,
                                                                 freqindex,
                                                                 condition)
        histofit2 = plt.figure('Gain singleplots')
        ax2 = histofit2.add_subplot(1, 1, 1)
        ax2.hist(bars, bins)
        ax2.plot(bins, sineplot)
        plt.savefig(outputname_single, format='png', dpi=600, pad_inches=0.1,
                    bbox_inches='tight')


# -----------------------------------------------------------------------------
# Evaluate the results
# -----------------------------------------------------------------------------


def compute_histogram(spike_times, simparameterdict):
    """
	compute a histogram to fit a sinusoidal to the firing rate
	"""
    # Generate the histogram
    if len(spike_times) > 1:
        hist_binwidth = simparameterdict[ 'binwidth' ]
        t_rec = simparameterdict[ 't_rec' ]
        t_recstart = simparameterdict[ 't_recstart' ]
        spike_times -= t_recstart  # shift recstart to 0
        # compute the time indices for the bins:
        t_bins = np.arange(0.0, t_rec, hist_binwidth)
        # compute the bin heights n. bins=t_bins is just an additional output:
        n, t_bins = np.histogram(spike_times, bins=t_bins)
        # correct n for the number of neurons that were averaged:
        heights = 1000. * n / (
        hist_binwidth * simparameterdict[ 'N' ])  # 1000 for spikes/s
    else:
        raise ValueError('no spikes recorded for histogram computation')
    return t_bins, heights, hist_binwidth


def compute_gain(bins, heights, simparamdict, I_stimdict, f, dt):
    """
	This version uses an alternate phase computation to get the difference in
	phase between the fitted sinus and the driving signal.
	:param bins: bin location for histogram
	:param heights: bar height for histogram
	:param simparamdict: ...
	:param I_stimdict: ...
	:param f: frequency
	:param dt: ...
	:return: gain[ amplitude, phase ]
	"""
    I_1 = I_stimdict[ 'amplitude' ]

    # get the stim params to set up start values:
    t_recstart = simparamdict[ 't_recstart' ]

    gain = np.zeros(2)
    r_0_rc = np.mean(heights)  # mean firing rate
    f_rc = f / 1000.  # frequency, 1/ms instead of Hz
    amp_guess_r = (max(heights) - min(heights)) / 2.

    # current injection works with a delay of dt
    # if f != 0:
    # 	phase = -((t_recstart - dt)/f_rc) * 2 * np.pi
    # else:
    phase = 0.0
    bins = bins[ :-1 ]  # for compatibility with np.histogram
    mysine = sinus(f_rc)  # frequency in 1/ms

    # fit a sine to the firing rates to extract sine parameters
    popt, pcov = opti.curve_fit(mysine.fit, bins, heights,
                                p0=(amp_guess_r, r_0_rc, phase))
    # content: popt = [ amplitude, offset, phase ]
    # sine_values = mysine.fit(bins, popt[ 0 ], popt[ 1 ], popt[ 2 ])
    # sine_phase = np.arcsin((sine_values - popt[ 2 ])/popt[ 0 ]) - 2*np.pi*f_rc*bin
    # print(sine_phase)

    print('fitted values: {0} for f={1}'.format(popt, f))
    gain[ 0 ] = abs(popt[ 0 ]) / I_1
    gain[ 1 ] = np.rad2deg(popt[ 2 ])

    # generate the time series for the true current sinsusoidal
    # stim_sinecurve = mysine.fit(bins - hist_binwidth, I_1, I_0, phase)
    # find the zero-crossings in both time series
    # compute the mean difference between zero crossings of both series
    return gain, r_0_rc


def gain_plots(filename='gains.csv', normalise=True, withone=True, **kwargs):
    """
	generates the plots for figure 6 from saved data files.
	"""
    gains, conditions = import_gain(filename)
    gaindict = split_gain_conditions(gains)
    gaindict = gain_normaliser(gaindict, normalise=normalise,
                               withone=withone, **kwargs)
    gaindict[ 'gains_low' ][ :, 1 ] = -360.0 * gaindict[ 'gains_low' ][ :,
                                               1 ] + 360.0
    gaindict[ 'gains_high' ][ :, 1 ] = -360.0 * gaindict[ 'gains_high' ][ :,
                                                1 ] + 360.0
    fig = show_gain(gaindict, save=True)
    return fig


# -----------------------------------------------------------------------------
#                                 __main__
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    import getopt
    import sys
    import time
    import nest
    import numpy as np

    rc('mathtext', default='regular')

    try:
        nest.set_verbosity('M_ERROR')
    except:
        print('Changing the nest verbosity did not succeed')
    try: nest.Install("gif2_module")
    except:
        print ("The GIF2 model had already been installed")
    dt = 0.01  # simulation timestep
    freqindex = int(sys.argv[ 1 ])
    maxindex = int(sys.argv[ 2 ])
    show_fits = bool(sys.argv[ 3 ])

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
        nest.SetKernelStatus({"local_num_threads": 16,
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

        write_results(resultdict)

        endtime = time.time()
        print('condition {0} with frequency {1} took {2}ms.'.format(
                condition, f, endtime - starttime))
        save_raw_data(resultdict, spike_times, spike_senders)
        print('saving successful')
        if show_fits:
            multiplot(condition, simparameterdict[ 'simindex' ], freqindex, hist_bins,
                      hist_heights, f, gain[ 0 ] * I_stimdict[ 'amplitude' ],
                      gain[ 1 ], exp_r_0)