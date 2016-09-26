# -*- coding: utf-8 -*-
"""
I am going to need these functions:
	- set parameters from a string
	- create devices from parameters
	- simulate with fixed f, parameters	
	- create histogram to fit with
	- normalize histogram
	- write histogram to file
	- load histogram
	- fit histogram
	- compute gain from fit
	
	The functions in this file will later be imported into 
	an executable python script. This python script will in
	turn be called by a sbatch script that starts computation
	on Blaustein.

	Suggested workflow: JOB ARRAY
	1.) run this scripit to set functions
	2.) run a bash script that does the following:
		2.1) run a function that write all the simulation parameters into a
			 hash code file.
		2.2) execute an array job with a script.
			2.2.1) update the hash file
		2.3) update
"""
# -----------------------------------------------------------------------------
# Imports and setup of Nest
# -----------------------------------------------------------------------------
import numpy as np
import scipy.optimize as opti
import matplotlib.pyplot as plt
from matplotlib import rc

rc('mathtext', default='regular')


# -----------------------------------------------------------------------------
# Mathematical functions
# -----------------------------------------------------------------------------


def mysine(t, f2, amp, off):
	"""
	Sine function for fitting only the amplitude, frequency and offset.
	:return: sine function with given parameters
	"""
	return amp * np.sin(2. * np.pi * f2 * t) + off


def mysine2(t, f_reconstructed, amp, off, phi):
	"""
	Sine function for fitting amplitude, frequency, offset and phase.
	:return: sine function with given parameters
	"""
	return amp * np.sin(2. * np.pi * f_reconstructed * t + phi) + off


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
	TODO: adjust the phase according to Jannis commentary!!!!!
	"""
	phase = 0.0
	ac_phase = -(
		((float(paramdict[ 't_recstart' ]) - 0.1) / 1000. * f) % 1) * 360
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


def write_results(resultdict, filename='gains.csv'):
	myCsvRow = str('{0}:{1}:{2}:{3}\n'.format(resultdict[ 'freqindex' ],
											  resultdict[ 'condition' ],
											  resultdict[ 'gain' ][ 0 ],
											  resultdict[ 'gain' ][ 1 ]
											  )
				   )
	fd = open(filename, 'a')
	fd.write(myCsvRow)
	fd.close()


def import_params_as_dict(filename='jobdict.txt'):
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
# Evaluate the results
# -----------------------------------------------------------------------------

def compute_histogram(spike_times, simparameterdict):
	# Generate the histogram
	if len(spike_times) > 1:
		# compute the width of one bin from given number of bins:
		hist_binwidth = float(simparameterdict[ 't_rec' ]) / int(
			simparameterdict[ 'nbins' ])
		# compute the time indices for the bins:
		t_bins = np.arange(np.amin(spike_times), np.amax(spike_times),
						   float(hist_binwidth))
		# compute the bin heights n. bins = t_bins is just an additional
		# output:
		n, t_bins = np.histogram(spike_times, bins=t_bins)
		# correct n for the number of neurons that were averaged:
		heights = 1000. * n / (hist_binwidth * simparameterdict[ 'N' ])
	return t_bins, heights, hist_binwidth


def compute_gain(bins, heights, hist_binwidth, I_1, f):
	"""
	Copmutes gain by fitting a sinus to the firing rate histogram.
	popt2 contains: frequency, amplitude, offset, phase
	:param bins: bin location for histogram
	:param heights: bar height for histogram
	:param hist_binwidth auxiliary parameter for shifting the bins
	:param I_1 amplitude of the oscillatory signal for computing the gain
	:param f: frequency
	:return: gain[ amplitude, phase ]
	"""
	# set up auxiliary variables and start values for fitting:
	gain = np.zeros(2)
	r_0_reconstructed = np.mean(heights)  # mean firing rate
	f_reconstructed = f / 1000.  # frequency, 1/ms instead of Hz
	amp_guess = (max(heights) + min(heights)) / 2. - r_0_reconstructed
	phase = 0.1  # current injection works with a delay of 0.1 ms
	bins = bins[ 1: ]  # for compatibility with np.histogram

	try:
		# fit a sine to the firing rates
		popt2, pcov2 = opti.curve_fit(mysine2, bins, heights, p0=(
			f_reconstructed, amp_guess, r_0_reconstructed, phase))

		# create the sine timeseries to calculate with
		sinecurve = mysine2(bins - hist_binwidth, f_reconstructed, popt2[ 1 ],
							popt2[ 2 ], popt2[ 3 ])
		if f == 0:
			exp_r_0 = popt2[ 2 ]
		r_1_reconstructed = max(abs(sinecurve - abs(popt2[ 2 ])))
		gain[ 0 ] = abs(r_1_reconstructed / I_1)
		gain[ 1 ] = popt2[ 3 ]
	except RuntimeError:
		gain = np.array([ 0., 0. ])
	return gain, r_0_reconstructed


# -----------------------------------------------------------------------------
# Visualise the results
# -----------------------------------------------------------------------------
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
				print('condition {0} coded as {1}'.format(cond, conditions.index(cond)))
			ind = conditions.index(cond)
			try:
				gain = np.vstack((gain,
								  np.array([ ind, freq, gain_amp, gain_phase ],
										   dtype=float)))
			except NameError:
				gain = np.array([ ind, freq, gain_amp, gain_phase ],
								dtype=float)
	gain = sort_gain(gain)
	return gain, conditions


def show_gain(gainmat, conditions, save=True):
	"""
	creates the fig6 plots from richardson et al., 2003
	"""
	# exp_f_r, exp_r_0, C_m, g, g_1, tau_1:
	paramdict = import_params_as_dict('jobdict.txt')

	# extract indices of successfully computed gain values for all conditions

	frequencies = np.hstack((np.array([ 0 ]),
							 np.logspace(-1., 2., num=len(
								 np.unique(gainmat[ :, 1 ])) -1 )))
	# TODO: make this more flexible to allow for more conditions!!!

	freqinddict = dict()
	freqdict = dict()
	for i in np.unique(gainmat[ :, 0 ]):
		freqinddict[i] = np.array(gainmat[ gainmat[ :, 0 ] == i ][ :, 1 ], dtype=int)
		freqdict[i] = frequencies[ freqinddict[ i ] ]
	# extract the indices of successful computed gain values
	freqs_high_ind = np.array(gainmat[ gainmat[ :, 0 ] == 0 ][ :, 1 ], dtype=int)
	freqs_low_ind = np.array(gainmat[ gainmat[ :, 0 ] == 1 ][ :, 1 ], dtype=int)
	# print('shape ')

	# find the frequencies for these indices
	freqs_low = frequencies[ freqs_low_ind ]
	freqs_high = frequencies[ freqs_high_ind ]

	# find the gains
	gains_high = gainmat[ gainmat[ :, 0 ] == 0 ][ :, 2: ]
	gains_low = gainmat[ gainmat[ :, 0 ] == 1 ][ :, 2: ]

	# plot the gains
	fig = plt.figure("Noise-dependent gain of the GIF2")
	plt.subplot(211)
	plt.semilogx(freqs_high[ 1: ], gains_high[ 1:, 0 ], color='blue',
				 marker='o', ls=' ')
	plt.semilogx(freqs_low[ 1: ], gains_low[ 1:, 0 ], color='red',
				 marker='o', ls=' ')
	plt.grid(True)
	# plt.text(0.25, 3.3,
	#          'C = {0}pF, g = {1}nS, g1 = {2}nS, t1 = {3}ms'.format(C_m, g,
	#                                                              g_1, tau_1))
	# plt.axvline(x=exp_f_r * 1000, color='green')
	# plt.text(exp_f_r * 1000 - 1., 0.25, r"$f_R$", color='green')
	# plt.axvline(x=exp_r_0, color='green')
	# plt.text(exp_r_0 - 2., 0.25, r"$r_0$", color='green')
	plt.xlabel('Frequency [Hz]', size=9.)
	plt.ylabel('Normalised amplitude of signal gain', size=9.)
	plt.xlim([ -1., 100. ])
	# plt.ylim([ 0, 3.5 ])
	plt.subplot(212)
	plt.grid(True)
	plt.semilogx(freqs_high[ 1: ], gains_high[ 1:, 1 ], color='blue',
				 marker='o', ls=' ')
	plt.semilogx(freqs_low[ 1: ], gains_low[ 1:, 1 ], color='red',
				 marker='o', ls=' ')
	plt.xlabel('Frequency [Hz]', size=9.)
	plt.ylabel('Phase of signal gain', size=9.)
	plt.xlim([ -1., 100. ])
	# plt.ylim([ -10, 10 ])
	# plt.axvline(x=exp_f_r * 1000)
	# plt.axvline(x=exp_r_0)
	if save == True:
		plt.savefig('Richardson_Fig6CD.png')
	else:
		plt.ion()
		plt.show()
	return fig


def gain_plots(normalise=True, withone=False):
	"""
	generates the plots for figure 6 from saved data files.
	"""
	gains, conditions = import_gain(filename='gains.csv')

	if normalise and not withone :
		gains[ :, 2 ] /= gains[ 0, 2 ]
		gains[ :, 3 ] /= gains[ 0, 3 ]
	elif normalise and withone :
		gains[ :, 2 ] /= gains[ 1, 2 ]
		gains[ :, 3 ] /= gains[ 1, 3 ]
	fig = show_gain(gains, conditions, save=True)
	return fig