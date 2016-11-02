# coding=utf-8
"""
part two of the excitability project. do a parameterscan to fit the firing
rates of gif2 to lif between 0 and 80 Hz.
"""
import numpy as np
import matplotlib.pyplot as plt
import nest
from excitability1 import read_solution
import sys


def fit_spikerates(filename, weighted=False):
    """
    fit the spike rates depending on input frequency
    """
    a, b = read_solution(filename)
    c = a[ b, : ]
    print('Acquired {0} solutions from file {1}.'.format(c.shape[ 0 ],
                                                         filename))
    assert (c.shape[ 0 ] > 10)
    if c.shape[ 1 ] > 1000:
        print('Found {0} possible solutions. This might take a while.'.format(
            c.shape[ 1 ]))
    c = np.hstack((c, np.zeros((c.shape[ 0 ], 1))))
    # rows of c are:
    #      0       1   2    3     4       5          6          7        8
    #    9          10
    # input rate | C | g | g1 | tau1 | V_range | rate_gif | rate_iaf | cv_gif
    # | cv_iaf | rmse diff

    for i in np.arange(0, c.shape[ 0 ]):
        # sys.stdout.write('\r')
        # sys.stdout.write('case {0}/{1}'.format(i, c.shape[ 0 ]))
        # sys.stdout.flush()
        results = run_sims(c[ i, 1 ], c[ i, 2 ], c[ i, 3 ])
        if not weighted:
            results = compute_rmse(results)
        else:
            results = compute_rmse(results, weighted=True)
        c[ i, -1 ] = results[ 3 ]
        if weighted:
            with open('ratefit_weighted_{0}.csv'.format(sys.argv[ 1 ]), 'a') as output:
                print('weighted')
                np.savetxt(output, results, fmt="%12.6G", newline=' ')
                output.write(' \n')
        else:
            with open('ratefit_{0}.csv'.format(sys.argv[ 1 ]), 'a') as output:
                np.savetxt(output, results, fmt="%12.6G", newline=' ')
                output.write(' \n')
    if weighted:
        with open('fitted_excitability_weighted_{0}.csv'.format(sys.argv[ 1 ]), 'w') as output:
            np.savetxt(output, c, fmt="%12.6G")  # , newline=' ')
            print('weighted')
    else:
        with open('fitted_excitability_{0}.csv'.format(sys.argv[ 1 ]), 'w') as output:
            np.savetxt(output, c, fmt="%12.6G")  # , newline=' ')
    return c


def run_sims(C, g, g1):
    """
    runs simulations for the rmse for a combination of parameters
    """
    runs = 51
    resultarray = np.zeros(5 + 2 * runs)
    resultarray[ 0 ] = C
    resultarray[ 1 ] = g
    resultarray[ 2 ] = g1
    # resultarray 3 will contain the rmse
    # resultarray 4 will contain the comparable number of rates

    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    dt = 0.1  # the resolution in ms
    nest.SetKernelStatus(
        {"resolution": dt, "print_time": False, "overwrite_files": True})
    try:
        nest.Install("gif2_module")
    except:
        pass
    recstart = 500.0
    simtime = 2000.0
    tauSyn = 0.5
    V_theta = 15.0
    tau_1 = 100.0
    V_range = 6.0
    synweight = 87.8
    J_ex = 0.125
    gif_params = {"tau_1":      tau_1,
                  "C_m":        C,  # C_m2,
                  "tau_syn_ex": tauSyn,
                  "tau_syn_in": tauSyn,
                  "g_rr":       g1,  # g_1,
                  "g":          g,  # g_m,
                  "V_m":        0.0,
                  "V_reset":    V_theta - V_range,
                  "E_L":        0.0,
                  "V_th":       V_theta}

    iaf_params = {"C_m":        250.0,
                  "tau_m":      10.0,
                  "tau_syn_ex": tauSyn,
                  "tau_syn_in": tauSyn,
                  "t_ref":      2.0,
                  "E_L":        0.0,
                  "V_reset":    0.0,
                  "V_m":        0.0,
                  "V_th":       V_theta}

    det_params = {"withtime": True,
                  "withgid":  False,
                  "to_file":  False,
                  "start":    recstart}

    stm_params = {"rate":      0.0,
                  "amplitude": 0.025 * 0.0,
                  "frequency": 10.0,
                  "phase":     0.0}

    nest.CopyModel("static_synapse", "excitatory",
                   {"weight": J_ex * synweight, "delay": 1.0})

    drive = nest.Create("sinusoidal_poisson_generator", params=stm_params)
    iafspikes = nest.Create("spike_detector", params=det_params)
    gifspikes = nest.Create("spike_detector", params=det_params)
    gif = nest.Create("gif2_psc_exp")
    iaf = nest.Create("iaf_psc_exp")
    nest.SetStatus(gif, gif_params)
    nest.SetStatus(iaf, iaf_params)
    nest.Connect(drive, gif, syn_spec={"model":  "static_synapse",
                                       "weight": J_ex * synweight,
                                       "delay":  0.5})
    nest.Connect(drive, iaf, syn_spec={"model":  "static_synapse",
                                       "weight": J_ex * synweight,
                                       "delay":  0.5})
    nest.Connect(gif, gifspikes)
    nest.Connect(iaf, iafspikes)
    for p, q in enumerate(np.linspace(50000.0, 125000.0, runs)):
        nest.ResetNetwork()
        nest.SetStatus(gif, gif_params)
        nest.SetStatus(iaf, iaf_params)
        stm_params = {"rate":      q,
                      "amplitude": 0.025 * q,
                      "frequency": 10.0,
                      "phase":     0.0}
        nest.SetStatus(drive, stm_params)
        nest.Simulate(recstart + simtime)
        rate_iaf = nest.GetStatus(iafspikes, "n_events")[ 0 ] / simtime * 1000.
        rate_gif = nest.GetStatus(gifspikes, "n_events")[ 0 ] / simtime * 1000.
        resultarray[ 5 + p ] = rate_gif
        resultarray[ 5 + runs + p ] = rate_iaf
    return resultarray


def squared_weighting(points):
    """
    weights along an array of points, s.th. minimum in center = 0.25 and max
     at the edges of 80 Hz = 4.0 with parabolic slope.
    """
    ratio = 3.75 / 1600
    return ratio * ( points - 40.0 )**2 + 0.25


def compute_rmse(simdata, weighted=False):
    """
    this computes the rmse for a given array of simulated data
    """
    runs = 51
    npoints = 0
    rmse = 0.0
    # do the weighting:
    if weighted:
        weights = squared_weighting(np.linspace(0, 80.0, runs))
    else:
        weights = np.ones(runs)
    # loop over the timeseries and compute:
    for j in np.arange(0, runs):
        if simdata[ 5 + j ] <= 80.0 or simdata[ 5 + runs + j ] <= 80.0:
            npoints += 1
            rmse += weights[ j ] * (simdata[ 5 + j ] - simdata[ 5 + runs + j ])**2
    simdata[ 3 ] = np.sqrt(rmse)
    simdata[ 4 ] = npoints
    return simdata


def return_n_smallest_rmses(n, solarray):
    """
    for a given array of solutions with the rmse in the last column extract the
    lines with the n smallest rmses
    """
    lastcol = solarray[ :, -1 ]
    ind = np.argpartition(lastcol, -n)[ -n: ]
    return solarray[ ind ]


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import nest
    from excitability1 import read_solution
    import sys

    filestring = 'excitability_{0}.csv'.format(sys.argv[ 1 ])
    fit_spikerates(filestring, weighted=bool(sys.argv[ 2 ]))
    plt.ion()
