# coding=utf-8
"""
part two of the excitability project. do a parameterscan to fit the firing
rates of gif2 to lif between 0 and 80 Hz.
"""
import numpy as np
import matplotlib.pyplot as plt
import nest
from excitability1 import read_solution


def fit_spikerates(filename):
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
    c = np.hstack((c, np.zeros_like(c[ :, -1 ])))
    # rows of c are:
    #      0       1   2    3     4       5          6          7        8
    #    9          10
    # input rate | C | g | g1 | tau1 | V_range | rate_gif | rate_iaf | cv_gif
    # | cv_iaf | rmse diff

    for i in np.arange(0, c.shape[ 1 ]):
        results = run_sims(c[ i, 1 ], c[ i, 2 ], c[ i, 3 ])
        results[ 3 ] = compute_rmse(results)
        with open('ratefit_{0}.csv'.format(sys.argv[ 1 ]), 'a') as output:
            np.savetxt(output, results, fmt="%12.6G", newline=' ')
            output.write(' \n')
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
    simtime = 1500.0
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
    for p in np.linspace(50000.0, 125000.0, runs):  # TODO: change indexing!
        nest.ResetNetwork()
        nest.SetStatus(gif, gif_params)
        nest.SetStatus(iaf, iaf_params)
        stm_params = {"rate":      p,
                      "amplitude": 0.025 * p,
                      "frequency": 10.0,
                      "phase":     0.0}
        nest.SetStatus(drive, stm_params)
        nest.Simulate(recstart + simtime)
        rate_iaf = nest.GetStatus(iafspikes, "n_events")[ 0 ] / simtime * 1000.
        rate_gif = nest.GetStatus(gifspikes, "n_events")[ 0 ] / simtime * 1000.
        resultarray[ 5 + p ] = rate_gif
        resultarray[ 5 + runs + p ] = rate_iaf
    return resultarray


def compute_rmse(simdata):
    """
    this computes the rmse for a given array of simulated data
    """
    runs = 51
    npoints = 0
    rmse = 0.0
    for j in np.arange(0, 51):
        if resultarray[ 5 + j ] <= 80.0 and resultarray[ 5 + runs + j ] <= 80.0:
            npoints += 1
            rmse += (resultarray[ 5 + j ] - resultarray[ 5 + runs + j ])**2
    simdata[ 3 ] = rmse
    simdata[ 4 ] = npoints
    return simdata