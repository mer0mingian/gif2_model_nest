# coding=utf-8
"""
Collection of tools pasted together from various locations to analyse the Brunel network
"""
import os
import copy
import numpy as np
import time
import sys
import shutil
import contextlib
import matplotlib as mpl
if ('blaustein' in os.environ['HOSTNAME'].split('.')[ 0 ]):
        cluster = True
        mpl.use('Agg')
        print('recognised cluster')
else: 
    cluster = False
    print('working locally')
import matplotlib.pyplot as plt
from elephant.statistics import isi, cv
from itertools import product
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

import correlation_toolbox.correlation_analysis as corr
import correlation_toolbox.helper as ch
from mingtools1 import *

# ****************************************************************************************************************************
# FORMER BETTER CONFIGURATION GENERATOR
# ****************************************************************************************************************************
def looptester():
    """
    This is a default script that should be adapted to the respective purpose.

    It is intended to compute the array length required to adequately scan
    parameter space for a suitable configuration of a GIF2 neuron in a Brunel
    network.

    Insert k as the array length in one of the shell scripts in this folder!
    """

    from mingtools1 import predict_str_freq

    dV1_range = np.arange(3.0, 9.5, 0.05)
    dV2_range = np.arange(4.0, 9.5, 0.05)
    k = 0
    # conn_probs_new_L5
    for i in dV1_range:
        for j in dV2_range:
            k += 1
    print(k)


def configuration_extractor(resultfile, criterion='new'):
    data = np.loadtxt('brunel_results/brunel_array_results_{0}.csv'.format(
                            resultfile))
    if criterion == 'old':
        """
        Detects three test arrays with rows containing parameter configs.
        Use function convert_to_paramdict to obtain dict for run_brunel.
        """
        exin_rates_lower_bound = 5.0
        exin_rates_upper_bound = 7.5
        distance_r_to_exin_bound = 2.0

        # find all rows with acceptable E/I rates:
        mean_exin = np.mean((data[:,8:9]),axis=1)
        mean_exin_accept = np.all((
            mean_exin <= exin_rates_upper_bound, 
            mean_exin >=exin_rates_lower_bound), axis=0)

        # is R rate also acceptable?
        dist_r_exin = np.abs(data[:,10] - mean_exin)
        dist_r_exin_accept = np.abs(dist_r_exin) <= distance_r_to_exin_bound

    elif criterion == 'new':
        # 'L5' : {'E':8.142, 'I':9.078},
        ex_lower_bound = 8.142 - 0.25
        ex_upper_bound = 8.142 + 0.25
        in_lower_bound = 9.078 - 0.25
        in_upper_bound = 9.078 + 0.25

        mean_ex = data[ :, 8 ]
        mean_in = data[ :, 9 ]
        mean_re = data[ :, 10 ]

        mean_in_accept = np.all((
            mean_in <= in_upper_bound, mean_in >= in_lower_bound), axis=0)
        mean_re_accept = np.all((
            (mean_ex + mean_re) / 2.0 <= ex_upper_bound,
            (mean_ex + mean_re) / 2.0 >= ex_lower_bound), axis=0)

    # where do both criteria hold?
    rates_accept = np.all((mean_in_accept, mean_re_accept), axis=0)
    print('We have {0} results with acceptable rates.'.format(
        rates_accept.sum()))

    cv_lower_bound = 0.65
    cv_upper_bound = 1.25
    distance_penalty = 6.0

    # if the rates fit, what are the resulting CVs?
    cvs = data[:, 14]
    cvs_accept = np.all((
        cvs <= cv_upper_bound, cvs >= cv_lower_bound), axis=0)

    all_accept = np.all((rates_accept, cvs_accept), axis=0)
    # also acceptable rates?
    print('{0} among these have CVs between {1} and {2}'.format(
        all_accept.sum(), cv_lower_bound, cv_upper_bound))

    if criterion == 'old':
        # of the remaining configurations, which has...
        # ... the largest dV2?
        testindices1 = data[ :, 6 ] == np.amax(data[ all_accept, 6 ])
        testconfigs1 = data[ testindices1, : ]
        # ... the largest total dV?
        testvalue2 =  np.amax(data[ all_accept, 6 ] + data[ all_accept, 5 ])
        testindices2 = (data[ :, 5 ] + data[ :, 6 ]) == testvalue2
        testconfigs2 = data[ testindices2, : ]
        # ... the lowest RMSE of rate difference and dV total?
        # ... not yet implemented
        # ... the lower RMSE of rate difference and dV2?
        testvalue3 = np.sqrt((distance_penalty * np.ones_like(data[ 
            all_accept, 6 ]) - data[ all_accept, 6 ])**2 + dist_r_exin[
            all_accept ]**2)
        testindices3 = np.sqrt((distance_penalty * np.ones_like(data[
         :, 6 ]) - data[ :, 6 ])**2 + dist_r_exin**2) == np.amin(testvalue3)
        testconfigs3 = data[ testindices3, : ]
    elif criterion == 'new':
        # of the reamaining configurations, which has...
        # ... the closest mean rates?
        mean_reex = np.abs(np.mean(
                            [ data[ :, 8 ], data[ :, 10 ] ], axis=0) - 8.142)
        testvalue1 = np.amin(mean_reex[ all_accept ])
        testindeices1 = (mean_reex == testvalue1)
        testconfigs1 = data[ testindeices1, : ]
        # the closest rates in E and R to previous E?
        testconfigs2 = testconfigs1
        # the smallest RMSE between difference in E and mean difference in E/R?
        dist_ex = np.abs(mean_ex - 8.142)
        total_dist = dist_ex**2 + mean_reex**2
        testvalue3 = np.amin(total_dist[ all_accept ])
        testindices3 = (total_dist == testvalue3)
        testconfigs3 = data[ testindices3, : ]

    return testconfigs1, testconfigs2, testconfigs3

# ****************************************************************************************************************************
# MICROCIRCUIT CONNECTIVITY TRANSFORMER FUNCTIONS
# ****************************************************************************************************************************

def find_Nsyn(Cold, Npre, Npost):
    """
    compute h3 = syn_nums = actual number of synapses = K_in * Npost
    """
    assert Npost != 0
    assert Npre != 0
    h1 = np.log(1. - Cold)
    h2 = (Npre * Npost - 1.) / (Npre * Npost)
    h3 = h1 / np.log(h2)
    return h3


def find_new_C(K, Npre, Npost):
    """
    find new connection probability
    """
    assert Npost != 0
    assert Npre != 0
    h1 = (Npre * Npost - 1.) / float(Npre * Npost)
    h2 = np.power(h1, K)
    h3 = 1. - h2
    return h3


def compute_new_connectivity(conn_probs_old, neuron_nums_new, neuron_nums_old, layers, 
                             pops_old, pops_new, structure_new, structure_old):
    """
    the core function.
    """
    n_new_pops_per_layer = len(pops_new)
    n_pops_per_layer = len(pops_old)
    n_layers = len(layers)
    n_pops_new = n_layers * n_new_pops_per_layer
    n_pops_old = n_layers * n_pops_per_layer
    conn_probs_new = np.zeros((n_pops_new, n_pops_new))
    syn_nums_new = np.zeros((n_pops_new, n_pops_new))
    syn_nums_old = np.zeros((n_pops_old, n_pops_old))

    for tl, tl_dict in neuron_nums_new.iteritems():
        for tp, tp_size in tl_dict.iteritems():
            bindex_t = structure_new[ tl ][ tp ]

            for sl, sl_dict in neuron_nums_new.iteritems():
                for sp, sp_size in sl_dict.iteritems():
                    bindex_s = structure_new[ sl ][ sp ]
                    if (sp_size == 0 or tp_size == 0):                         # target or source have size 0?
                        conn_probs_new[ bindex_t, bindex_s ] = 0
                        syn_nums_new[ bindex_t, bindex_s ] = 0
                    else:
                        # collect the old connectivity:
                        if 'R' in sp:                                          # take source connectivity from E for R?
                            source_index = structure_old[ sl ][ 'E' ]          # get the source index for the old connectivity list
                            Npre_old = neuron_nums_old[ sl ][ 'E' ]
                        else:
                            source_index = structure_old[ sl ][ sp ]           # get the source index for the old connectivity list
                            Npre_old = neuron_nums_old[ sl ][ sp ]
                        if 'R' in tp:                                          # take target connectivity from E for R?
                            target_index = structure_old[ tl ][ 'E' ]          # get the target index for the old connectivity list
                            Npost_old = neuron_nums_old[ tl ][ 'E' ]
                        else:                                                  # just E and I populations connecting
                            target_index = structure_old[ tl ][ tp ]           # get the target index for the old connectivity list
                            Npost_old = neuron_nums_old[ tl ][ tp ]
                        Cold = conn_probs_old[ target_index ][ source_index ]  # get the 'old' connectivity list entry

                        # compute new connectivity:
                        if Cold == 0:                                          # was connectivity 0 anyway?
                            conn_probs_new[ bindex_t, bindex_s ] = 0
                        else:                                                  # compute new connectivity
                            n_syn = find_Nsyn(Cold, Npre_old, Npost_old)       # replaces K with Nsyn, find_C with find_Nsyn
                            rel = float(sp_size) / float(Npre_old) * float(tp_size) / float(Npost_old)  # number of synapses with same relation as between old population sizes
                            # put synapse number into matrix
                            syn_nums_new[ bindex_t, bindex_s ] = n_syn * rel
                            syn_nums_old[ target_index, source_index ] = n_syn

                            conn_probs_new_L5 = find_new_C(
                                float(syn_nums_new[ bindex_t, bindex_s ]), #* rel, 
                                neuron_nums_new[ sl ][ sp ], 
                                neuron_nums_new[ tl ][ tp ])
                            conn_probs_new[ bindex_t, bindex_s ] = conn_probs_new_L5
    return conn_probs_new, syn_nums_new, syn_nums_old


# ****************************************************************************************************************************
# VISTOOLS, MOSTLY FROM THE MAM
# ****************************************************************************************************************************

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
