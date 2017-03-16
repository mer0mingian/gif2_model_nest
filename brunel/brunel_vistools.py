# -*- coding: utf-8 -*-

"""
ana_vistools for Brunel
============
Authors: Maximilian Schmidt, Sacha van Albada
modified by Daniel Mingers for Brunel network and Microcircuit
"""

import os
import re
import numpy as np
import scipy as sp
import matplotlib as mpl
if ('blaustein' in os.environ['HOSTNAME'].split('.')[ 0 ]):
        np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
        cluster = True
        mpl.use('Agg')
        print('recognised cluster')
else: 
    cluster = False
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
    print('working locally')
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.colors import LogNorm

import correlation_toolbox.helper as ch
from itertools import izip, product, repeat
from mingtools1 import *
from brunel_helper import *
try:
    from elephant.statistics import isi, cv
except ImportError:
    print('Warning: elephant functions not available!')

# ****************************************************************************************************************************
# MANUAL VISTOOLS
# ****************************************************************************************************************************


def rasterplot(spikelists, networkdict, targetname, length=None):
    """
    Creates a 3-coloured rasterplo from the outputdict of run_brunel
    """
    plt.clf()
    fraction = networkdict[ 'fraction' ]
    recstart = 2500.0
    spiketrains_ex, spiketrains_in, spiketrains_re = spikelists[ 1:4 ]
    # Raster:
    figu = plt.figure("Rasterplot")
    # figu.set_size_inches(18.5, 10.5)
    offset = 0
    for spiketrain in spiketrains_ex:
        if np.any(spiketrain):
            offset += 1
            figu.add_subplot(1, 1, 1)
            plt.plot(spiketrain - recstart,
                     offset * np.ones_like(spiketrain),
                     'b.',
                     markersize=1)
    for spiketrain in spiketrains_re:
        if np.any(spiketrain):
            offset += 1
            figu.add_subplot(1, 1, 1)
            plt.plot(spiketrain - recstart,
                     offset * np.ones_like(spiketrain),
                     'g.',
                     markersize=1)
    for spiketrain in spiketrains_in:
        if np.any(spiketrain):
            offset += 1
            figu.add_subplot(1, 1, 1)
            plt.plot(spiketrain - recstart,
                     offset * np.ones_like(spiketrain),
                     'r.',
                     markersize=1)
    plt.ylim(0, offset)
    ax = figu.gca()
    ax.set_xlabel('Time [ms]', size=16)
    ax.set_ylabel('Neuron GID', size=16)
    if length is not None:
        plt.xlim(0, length)
    else:
        plt.xlim(0, networkdict[ 'simtime' ])
    plt.savefig('driven_plots/Rasterplot_{0}.png'.format(targetname), dpi=300)



def multi_taper_spectrum(networkdict, spikelists, targetname, hist_binwidth, 
                         NW, xlims=[ 0.0, 150.0 ]):
    """
    create a spectral plot with histogram bin width resolution
    """
    import nitime as nt
    plt.clf()

    def dB(x, out=None):
        if out is None:
            return 10 * np.log10(x)
        else:
            np.log10(x, out)
            np.multiply(out, 10, out)
    ln2db = dB(np.e)


    # First, collect all the spike data
    detectordict = spikelists[ -1 ]

    spikearray_all = np.array([[0.],[0.]])
    for key in detectordict.keys():
        detectordict[ key ][ 'spikearray' ] = np.vstack((
                       detectordict[ key ][ u'senders' ],
                       detectordict[ key ][ u'times' ]))
        spikearray_all = np.hstack((spikearray_all, 
                       detectordict[ key ][ 'spikearray' ]))
    # cut off the initialisation point:
    spikearray_all = spikearray_all[ :, 1: ]
    # spikearray_all[ 0, : ] contains the neuron-gids
    # spikearray_all[ 1, : ] contains the spiketimes


    # convert the unsorted spikes in to a time series :
    t_min = 2500.0 # recstart
    t_max = 2500.0 + networkdict[ 'simtime' ] # t_min + simtime
    # resolution = resolution # histogram bin width in ms
    num_neur = 5915 # num neurons in L5

    # rate, times = np.histogram(spikearray_all[ 1, : ], bins=int((t_max - t_min) / (resolution)),
    #                            range=(t_min + resolution / 2., t_max + resolution / 2.))
    # rate = rate / (num_neur * resolution / 1000.0)

    # spectrum
    # Fs is the signal sampling rate. since we bin in resolution * 1e-3, the
    # inverse of that is our sampling frequency
    # Fs = np.power(resolution * 1e-3, -1.)

    rectime = networkdict[ 'simtime' ] # recording time in ms
    n_hist_bins = rectime / hist_binwidth
    Fs = n_hist_bins / (rectime / 1000.0)  # histogram sampling freq [Hz]

    rate, times = np.histogram(spikearray_all[ 1, : ], 
                bins=int(n_hist_bins),
                range=(t_min + hist_binwidth / 2., 
                       t_max + hist_binwidth / 2.))
    rate = rate / (num_neur * hist_binwidth / 1000.0)


    freqs, psd_est, nu = nt.algorithms.spectral.multi_taper_psd(
                        rate, Fs=Fs, NW=NW, adaptive=True, jackknife=False)

    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    show_variability = False

    if not show_variability:
        ax.set_title('Power spectrum Layer 5', size=20)
        ax.plot(freqs, psd_est, color='k', linewidth=1., label='PSD')
        ax.set_xlabel('Frequency [Sp/s]', size=16)
        ax.set_ylabel('Power', size=16)
        try:
            ax.set_yscale("Log")
        except UserWarning:
            pass
        ax.set_xlim(xlims)
        ax.set_ylim([ psd_est.min(), psd_est.max() + 1. ])
        plt.savefig('driven_plots/spectrum_{0}.png'.format(targetname))
    else:
        import scipy.stats.distributions as dist

        print('WARNING: NOT CORRECTLY IMPLEMENTED INTERVALS')
        p975 = dist.chi2.ppf(.975, nu)
        p025 = dist.chi2.ppf(.025, nu)
        l1 = nu / p975
        l2 = nu / p025
        hyp_limits = np.log(( l1, l2 ) + psd_est)

        # temporarily change to notation from nitime function
        ax.set_title('MT with adaptive weighting and 95% interval, BW=1Hz', size=14)
        ax.plot(freqs, psd_est, color=c, linewidth=1., label='PSD')
        ax.fill_between(freqs, hyp_limits[0], y2=hyp_limits[1], color=(1, 0, 0, .3), alpha=0.5)
        # ax_limits = (psd_est.min() - 2*np.abs(psd_est.min()),
        #              psd_est.max() + 1.25*np.abs(psd_est.max()))
        # ax.set_ylim(ax_limits)
        ax.legend()
        ax.set_yscale("Log")
        ax.set_xlabel('Frequency [Sp/s]', size=16)
        ax.set_ylabel('Power', size=16)
        plt.savefig('driven_plots/spectrum_{0}.png'.format(targetname))


# ****************************************************************************************************************************
# VISTOOLS, MOSTLY FROM THE MAM
# ****************************************************************************************************************************

def single_power_display(area, pop=None, t_min=None,
                         t_max=None, resolution=1., kernel='binned', Df=None, **keywords):
    """
    Plot power spectrum for a single area.
    Directly computes the values via function 'spectrum' using
    rate time series stored in dictionary pop_rate_time_series.

    Parameters
    ----------
    area : string {area}
        Area to be plotted.
    pop : string, optional
        If given, the rate of a specific population in area is plotted.
        Defaults to None, then the area-averaged rate is plotted.
    t_min : float, optional
        Minimal time in ms of spikes to be shown.
        Defaults to minimal time of underlying rate time series.
    t_max : float, optional
        Minimal time in ms of spikes to be shown.
        Defaults to maximal time of underlying rate time series.
    kernel : {'gauss_time_window', 'alpha_time_window', 'rect_time_window'}, optional
        Specifies the kernel to be convolved with the spike histogram.
        Defaults to 'binned', which corresponds to no convolution.
    resolution: float, optional
        Width of the convolution kernel. Specifically it correponds to:
        - 'binned' : bin width of the histogram
        - 'gauss_time_window' : sigma
        - 'alpha_time_window' : time constant of the alpha function
        - 'rect_time_window' : width of the moving rectangular function
    Df : float, optional
        Window width of sliding rectangular filter (smoothing) of the spectrum.
        The default value is None and leads to no smoothing.
    output : {'pdf', 'png', 'eps'}, optional
        If given, the function stores the plot to a file of the given format.
    """
    if pop is None:
        data = self.spike_data[area][self.structure[area][0]]
        num_neur = self.num_spike_neurons[area][self.structure[area][0]]
        for population in self.structure[area][1:]:
            data = np.vstack((data, self.spike_data[area][population]))
            num_neur += self.num_spike_neurons[area][self.structure[area][0]]
    else:
        data = self.spike_data[area][pop]
        num_neur = self.num_spike_neurons[area][pop]

    if t_max is None:
        t_max = self.T
    if t_min is None:
        t_min = 0.

    power, freq = ah.spectrum(data, num_neur, t_min, t_max,
                              resolution=resolution, kernel=kernel, Df=Df)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(freq, power, color='k', markersize=3)
    if pop:
        ax.set_title(area + ' ' + pop + ' ' + kernel)
    else:
        ax.set_title(area + ' ' + kernel)
    ax.set_xlabel('Frequency [Hz]', size=16)
    ax.set_ylabel('Power', size=16)
    ax.set_xlim(0.0, 500.0)
    ax.set_yscale("Log")

    if 'output' in keywords:
        if pop:
            plt.savefig(os.path.join(self.output_dir, self.label + '_power_spectrum_' + area +
                                     '_' + pop + '.' + keywords['output']))

        else:
            plt.savefig(os.path.join(self.output_dir, self.label +
                                     '_power_spectrum_' + area + '.' + keywords['output']))
    else:
        fig.show()



import numpy as np

def pop_rate_time_series(data_array, num_neur, t_min, t_max,
                         resolution=10., kernel='binned'):
    """
    BY MAX SCHMIDT, MULTI-AREA MODEL: ana_vistool_helpers.py
    Computes time series of the population-averaged rates of a group of neurons.

    Parameters
    ----------
    data_array : numpy.ndarray
        Array with spike data.
        column 0: neuron_ids, column 1: spike times
    tmin : float
        Minimal time for the calculation.
    tmax : float
        Maximal time for the calculation.
    num_neur: int
        Number of recorded neurons. Needs to provided explicitly
        to avoid corruption of results by silent neurons not
        present in the given data.
    kernel : {'gauss_time_window', 'alpha_time_window', 'rect_time_window'}, optional
        Specifies the kernel to be convolved with the spike histogram.
        Defaults to 'binned', which corresponds to no convolution.
    resolution: float, optional
        Width of the convolution kernel. Specifically it correponds to:
        - 'binned' : bin width of the histogram
        - 'gauss_time_window' : sigma
        - 'alpha_time_window' : time constant of the alpha function
        - 'rect_time_window' : width of the moving rectangular function
        Defaults to 1 ms.

    Returns
    -------
    time_series : numpy.ndarray
        Time series of the population rate
    """
    if kernel == 'binned':
        rate, times = np.histogram(data_array[:, 1], bins=int((t_max - t_min) / (resolution)),
                                   range=(t_min + resolution / 2., t_max + resolution / 2.))
        rate = rate / (num_neur * resolution / 1000.0)
        rates = np.array([])
        last_time_step = times[0]

        for ii in range(1, times.size):
            rates = np.append(
                rates, rate[ii - 1] * np.ones_like(np.arange(last_time_step, times[ii], 1.0)))
            last_time_step = times[ii]

        time_series = rates
    else:
        spikes = data_array[:, 1][data_array[:, 1] > t_min]
        spikes = spikes[spikes < t_max]
        binned_spikes = np.histogram(spikes, bins=int(
            (t_max - t_min)), range=(t_min, t_max))[0]
        if kernel == 'rect_time_window':
            kernel = np.ones(int(resolution)) / resolution
        if kernel == 'gauss_time_window':
            sigma = resolution
            time_range = np.arange(-0.5 * (t_max - t_min),
                                   0.5 * (t_max - t_min), 1.0)
            kernel = 1 / (np.sqrt(2.0 * np.pi) * sigma) * \
                np.exp(-(time_range ** 2 / (2 * sigma ** 2)))
        if kernel == 'alpha_time_window':
            alpha = 1 / resolution
            time_range = np.arange(-0.5 * (t_max - t_min),
                                   0.5 * (t_max - t_min), 1.0)
            time_range[time_range < 0] = 0.0
            kernel = alpha * time_range * np.exp(-alpha * time_range)

        rate = np.convolve(kernel, binned_spikes, mode='same')
        rate = rate / (num_neur / 1000.0)
        time_series = rate

    return time_series



def pop_rate_distribution(data_array, t_min, t_max, num_neur):
    """
    Calculates firing rate distribution over neurons in a given array of spikes.
    Rates are calculated in spikes/s. Assumes spikes are sorted according to time.
    First calculates rates for individual neurons and then averages over neurons.

    Parameters
    ----------
    data_array : numpy.ndarray
        Array with spike data.
        column 0: neuron_ids, column 1: spike times
    tmin : float
        Minimal time stamp to be considered in ms.
    tmax : float
        Maximal time stamp to be considered in ms.
    num_neur: int
        Number of recorded neurons. Needs to provided explicitly
        to avoid corruption of results by silent neurons not
        present in the given data.

    Returns
    -------
    bins : numpy.ndarray
        Left edges of the distribution bins
    vals : numpy.ndarray
        Values of the distribution
    mean : float
        Arithmetic mean of the distribution
    std : float
        Standard deviation of the distribution
    """
    indices = np.where(np.logical_and(data_array[:, 1] > t_min,
                                      data_array[:, 1] < t_max))
    neurons = data_array[:, 0][indices]
    neurons = np.sort(neurons)
    n = neurons[0]
    rates = np.zeros(num_neur)
    ii = 0
    for i in xrange(neurons.size):
        if neurons[i] == n:
            rates[ii] += 1
        else:
            n = neurons[i]
            ii += 1
    rates /= (t_max - t_min) / 1000.
    vals, bins = np.histogram(rates, bins=100)
    vals = vals / float(np.sum(vals))
    if num_neur > 0. and t_max != t_min and len(data_array) > 0 and len(indices) > 0:
        return bins[0:-1], vals, np.mean(rates), np.std(rates)
    else:
        return np.arange(0, 20., 20. / 100.), np.zeros(100), 0.0, 0.0


# Synchrony measures
def synchrony(data_array, num_neur, t_min, t_max, resolution=1.0):
    """
    Compute the synchrony of an array of spikes as the coefficient
    of variation of the population rate.
    Uses pop_rate_time_series().


    Parameters
    ----------
    data_array : numpy.ndarray
        Array with spike data.
        column 0: neuron_ids, column 1: spike times
    tmin : float
        Minimal time for the calculation of the histogram in ms.
    tmax : float
        Maximal time for the calculation of the histogram in ms.
    resolution : float, optional
        Bin width of the histogram. Defaults to 1 ms.

    Returns
    -------
    synchrony : float
        Synchrony of the population.
    """
    spike_count_histogramm = pop_rate_time_series(
        data_array, num_neur, t_min, t_max, resolution=resolution)
    mean = np.mean(spike_count_histogramm)
    variance = np.var(spike_count_histogramm)
    synchrony = variance / mean
    return synchrony


def spike_synchrony(spike_data, t_min, t_max):
    """
    Compute the synchrony of a population of neurons as the ratio of
    the temporal variance of the population-averaged spike rate
    and the population-averaged temporal variance of single cell spike
    rates. The spike rates are computed as spike histograms with bin width 1 ms.
    See http://www.scholarpedia.org/article/Synchrony_measures, Eg. (4).

    Parameters
    ----------
    data_array : numpy.ndarray
        Array with spike data.
        column 0: neuron_ids, column 1: spike times
    t_min : float
        Minimal time point to be considered in ms.
    t_max : float
        Maximal time point to be considered in ms.

    Returns
    -------
    synchrony : float
        Synchrony of the population.
    """
    total_rate = np.zeros((t_max - t_min))
    variances = []

    neurons = np.unique(spike_data[:, 0])
    for n in neurons:
        dat = spike_data[np.where(spike_data[:, 0] == n)]
        rate = np.histogram(dat, bins=int((t_max - t_min)),
                            range=(t_min, t_max))[0]
        rate = np.array(rate, dtype=np.float)
        rate /= 1000.
        total_rate += rate
        variances.append(np.var(rate))
    total_rate /= neurons.size
    synchrony = np.sqrt(np.var(total_rate) / np.mean(variances))
    return synchrony


def spectrum(data_array, num_neur, t_min, t_max, resolution=1., kernel='binned', Df=None):
    """
    Compute compound power spectrum of a population of neurons.
    Uses the powerspec function of the correlation toolbox.

    Parameters
    ----------
    data_array : numpy.ndarray
        Array with spike data.
        column 0: neuron_ids, column 1: spike times
    t_min : float
        Minimal time for the calculation of the histogram in ms.
    t_max : float
        Maximal time for the calculation of the histogram in ms.
    num_neur: int
        Number of recorded neurons. Needs to provided explicitly
        to avoid corruption of results by silent neurons not
        present in the given data.
    kernel : {'gauss_time_window', 'alpha_time_window', 'rect_time_window'}, optional
        Specifies the kernel to be convolved with the spike histogram.
        Defaults to 'binned', which corresponds to no convolution.
    resolution: float, optional
        Width of the convolution kernel. Specifically it correponds to:
        - 'binned' : bin width of the histogram
        - 'gauss_time_window' : sigma
        - 'alpha_time_window' : time constant of the alpha function
        - 'rect_time_window' : width of the moving rectangular function
        Defaults to 1 ms.
    Df : float, optional
        Window width of sliding rectangular filter (smoothing) of the spectrum.
        The default value is None and leads to no smoothing.

    Returns
    -------
    power : numpy.ndarray
        Values of the power spectrum.
    freq : numpy.ndarray
        Discrete frequency values
    """
    rate = pop_rate_time_series(
        data_array, num_neur, t_min, t_max, kernel=kernel, resolution=resolution)
    rate = ch.centralize(rate, units=True)
    freq, power = corr.powerspec([rate * 1e-3], 1., Df=resolution)
    return power[0][freq > 0], freq[freq > 0]


def cross_correlation(time_series1, time_series2):
    """
    Compute cross-correlation coefficient between two time series.
    Uses numpy.correlate.

    Parameters
    ----------
    time_series1, time_series2 : numpy.ndarray
        The two time series to correlate with each other.

    Returns
    -------
    t : numpy.ndarray
        Discrete time lag values
    corr : numpy.ndarray
        Cross-correlation values
    """
    rates = [time_series1, time_series2]

    dat = [ch.centralize(rates[0], units=True),
           ch.centralize(rates[1], units=True)]
    freq, crossspec = corr.crossspec(dat, 1.)
    t, cross = corr.crosscorrfunc(freq, crossspec)
    return t, cross[0][1]


def cross_coherence(data_array1, data_array2, num_neur1, num_neur2,
                    t_min, t_max, resolution=1., kernel='binned'):
    """
    Computes compound cross coherence between 2 populations of neurons

    Parameters
    ----------
    data_array1, data_array2 : numpy.ndarray
        Arrays with spike data.
        column 0: neuron_ids, column 1: spike times
    num_neur1, num_neur2: int
        Number of recorded neurons. Needs to provided explicitly
        to avoid corruption of results by silent neurons not
        present in the given data.
    t_min : float
        Minimal time for the calculation.
    t_max : float
        Maximal time for the calculation.
    kernel : {'gauss_time_window', 'alpha_time_window', 'rect_time_window'}, optional
        Specifies the kernel to be convolved with the spike histogram.
        Defaults to 'binned', which corresponds to no convolution.
    resolution: float, optional
        Width of the convolution kernel. Specifically it correponds to:
        - 'binned' : bin width of the histogram
        - 'gauss_time_window' : sigma
        - 'alpha_time_window' : time constant of the alpha function
        - 'rect_time_window' : width of the moving rectangular function
        Defaults to 1 ms.

    Returns
    -------
    cr : numpy.ndarray
        Cross-coherence values
    freq : numpy.ndarray
        Discrete frequency values
    """
    rate_time_series1 = pop_rate_time_series(
        data_array1, num_neur1, t_min, t_max, resolution=resolution, kernel=kernel)
    rate_time_series2 = pop_rate_time_series(
        data_array2, num_neur2, t_min, t_max, resolution=resolution, kernel=kernel)

    d = [rate_time_series1, rate_time_series2]
    freq, cr = corr.crossspec(d, 1.0, Df=resolution)
    return cr[0][1][0:freq.size / 2], freq[0:freq.size / 2]


def corrcoeff(data_array, t_min, t_max, resolution=1., subsample=None):
    """
    Computes the correlation coefficient averaged across
    single cells of the given data_array.
    Uses correlation_toolbox.helper to prepare the spike data.
    Parameters
    ----------
    data_array : numpy.ndarray
        Array with spike data.
        column 0: neuron_ids, column 1: spike times
    t_min : float
        Minimal time for the calculation of the histogram in ms.
    t_max : float
        Maximal time for the calculation of the histogram in ms.
    resolution : float, optional
        Bin width of the histogram. Defaults to 1 ms.
    subsample : int, optional
        Number of neurons to consider in the calculation.
        Default value of None corresponds to taking the entire population into account.

    Returns
    -------
    corrcoeff : float
        Mean correlation coefficient of the population
    """

    min_gid = np.min(data_array[:, 0])
    if subsample:
        data_array = data_array[
            np.where(data_array[:, 0] < min_gid + subsample)]
    spikes = ch.sort_gdf_by_id(data_array)
    spikes = ch.strip_binned_spiketrains(spikes[1])[:subsample]
    bins, hist = ch.instantaneous_spike_count(
        spikes, resolution, tmin=t_min, tmax=t_max)
    cc = np.corrcoef(hist)
    cc = np.extract(1 - np.eye(cc[0].size), cc)
    cc[np.where(np.isnan(cc))] = 0.
    corrcoeff = np.mean(cc)
    return corrcoeff
