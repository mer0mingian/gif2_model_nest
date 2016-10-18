# -*- coding: utf-8 -*-
"""
This project intends to find parameter section, where the firing rate of the
generalised integrate-and-fire model with 2 variables is comparable to that of
the integrate-and-fire neuron.
(see Richardson, Brunel and Hakim, 2003, "From subthreshold resonance to firing
rate resonance." and Brunel et al., 2003.)

Parameters indices are deliverd via sys.argv to be run on a queing system.
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import sys
import elephant.statistics as es
from itertools import product


def compute_str_freq(C, t1, g, g1, verbose=False):
    a = g * t1 / C
    b = g1 * t1 / C
    h1 = (a + b + 1.0)**2
    h2 = (a + 1.0)**2
    h3 = np.sqrt(h1 - h2)
    fR = np.sqrt(h3 - 1.0) / t1 / np.pi * 500.0
    if verbose:
        # stability:
        if a > -1 and a + b > 0:
            print('Membrane potential stable')
        else:
            print('Membrane potential unstable')
        # Subthreshold resonance?
        if b > np.sqrt((a + 1.0)**2 + 1.0) - 1.0 - a:
            print('Subthreshold resonance will occur.')
        else:
            print('Subthreshold resonance will not occur.')
        # Phase lag:
        if b > 1:
            print('Zero phase-lag.')
        elif a < 0:
            print('Phase-lag > 90 deg.')
        # TODO: add behaviour for square pulse currents
    return fR



def read_solution(filename, rt=0.0, at=15.0):
    compmat = np.loadtxt(filename)
    # restore columns
    compmat = compmat.reshape((-1, 10))
    # replace nans by 0.0
    compmat = np.nan_to_num(compmat)
    # compare via spike rates:
    closevec = np.isclose(compmat[ :, 6 ], compmat[ :, 7 ], rtol=rt, atol=at)
    # find non-zero rates among these:
    nonzeroclose = np.all((closevec, compmat[ :, 6 ] != 0.0), axis=0)
    print('There are {0} simulated cases of equal rates!'.format(np.sum(nonzeroclose)))
    # compute the subthreshold resonance frequency:
    f_R = compute_str_freq(compmat[ :, 1 ], 100.0 * np.ones_like(compmat[ :, 1 ]), compmat[ :, 2 ], compmat[ :, 3 ], verbose=False)
    resonating = np.isclose(f_R, 10.0 * np.ones_like(f_R), atol=5.0)
    output = np.all((nonzeroclose, resonating), axis=0)
    print('Among these, {0} are resonating near 10Hz'.format(np.sum(output)))
    # equalcond = np.all((nonzeroclose, np.isclose(compmat[ :, 2 ], compmat[ :, 3 ], rtol=0.0, atol=15.0)), axis=0)
    # print('{0} of these have similar conductances!'.format(np.sum(equalcond)))
    # nonzerocv1 = np.all((equalcond, compmat[ :, 7 ] != 0.0), axis=0)
    # nonzerocv2 = np.all((nonzerocv1, compmat[ :, 9 ] != 0.0), axis=0)
    # print('Filtered solutions with CV = 0...')
    # output = np.all((nonzerocv2, compmat[ :, 2 ] <= 10e3), axis=0)
    # print('{0} of these non.zero CVs!'.format(np.sum(nonzerocv2)))
    # print('Maximal values: {0}'.format(np.max(compmat[ output, : ], axis=0)))
    # print('Minimal values: {0}'.format(np.min(compmat[ output, : ], axis=0)))
    return compmat, output


def read_final_solution(filename):
    """
    read acceptable parametersets and return as matrix
    """
    compmat = np.loadtxt(filename)
    # restore columns
    compmat = compmat.reshape((-1, 10))
    print('Found {0} cases'.format(compmat.shape[0]))
    return compmat


def find_unique_paramsets(compmat):
    """
    filter solutions that differ only in rate
    """
    mat = compmat[ :, 1:4 ]
    uniquelines = np.array()
    unique_c = np.unique(mat[ :, 0 ])
    for i in unique_c:
        unique_c_mat = mat[ mat[ :, 0 ] == i, : ]
        for j in np.unique(unique_c_mat[ :, 1 ]):
            unique_cg_mat = unique_c_mat[ unique_c_mat[ :, 1 ] == j, : ]
            for k in np.unique(unique_cg_mat[ :, 2 ]):
                if k != j:
                    uniquelines = np.vstack(uniquelines, np.array([i, j, k]))
    return uniquelines


def scatter_solutions(matrix):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(matrix[ :, 0 ], matrix[ :, 1 ], matrix[ :, 2 ], c='r', marker='o')
    ax.set_xlabel('rate')
    ax.set_ylabel('C')
    ax.set_zlabel('g')
    #ax.set_xlim(50000.0, 125000.0)
    #ax.set_ylim(150.0, 600.0)
    #ax.set_zlim(5.0, 125.0)
    return fig    


def run_specific_comparison(p_rate, C, g, g1):
    # Configure Nest
    import nest

    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    dt = 0.1  # the resolution in ms
    nest.SetKernelStatus(
        {"resolution": dt, "print_time": False, "overwrite_files": True})
    try:
        nest.Install("gif2_module")
    except:
        pass

    # Simulation setup
    recstart = 1500.0
    simtime = 5000.0

    # Shared/fixed neuron parameters
    tauSyn = 0.5
    V_theta = 15.0
    tau_1 = 100.0
    V_range = 6.0
    synweight = 87.8
    J_ex = 0.125
    J_re = 0.125

    # setup dictionaries
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

    stm_params = {"rate":      p_rate,
                  "amplitude": 0.025 * p_rate,
                  "frequency": 10.0,
                  "phase":     0.0}

    nest.CopyModel("static_synapse", "excitatory",
                   {"weight": J_ex * synweight, "delay": 1.0})

    # Create devices and neurons
    drive = nest.Create("sinusoidal_poisson_generator",
                        params=stm_params)
    iafspikes = nest.Create("spike_detector", params=det_params)
    gifspikes = nest.Create("spike_detector", params=det_params)
    gif = nest.Create("gif2_psc_exp")
    iaf = nest.Create("iaf_psc_exp")
    nest.SetStatus(gif, gif_params)
    nest.SetStatus(iaf, iaf_params)

    # Connect everything
    nest.Connect(drive, gif, syn_spec={"model":  "static_synapse",
                                       "weight": J_ex * synweight,
                                       "delay":  0.5})
    nest.Connect(drive, iaf, syn_spec={"model":  "static_synapse",
                                       "weight": J_ex * synweight,
                                       "delay":  0.5})
    nest.Connect(gif, gifspikes)
    nest.Connect(iaf, iafspikes)

    # Simulate
    nest.Simulate(recstart + simtime)

    # Evaluate
    rate_iaf = nest.GetStatus(iafspikes, "n_events")[
                   0 ] / simtime * 1000.
    rate_gif = nest.GetStatus(gifspikes, "n_events")[
                   0 ] / simtime * 1000.
    cv_gif = es.cv(es.isi(nest.GetStatus(gifspikes, "events")
                          [ 0 ][ "times" ]))
    cv_iaf = es.cv(es.isi(nest.GetStatus(iafspikes, "events")
                          [ 0 ][ "times" ]))

    # write out
    indexarray = np.array([ rate_gif, rate_iaf, cv_gif, cv_iaf ])
    return indexarray, nest.GetStatus(gifspikes, "events")[ 0 ][ "times" ]


def show_solution_rates(matrix):
    """
    plot spike courses of all gifs in solutions
    :param matrix: output from read_solution
    """
    import matplotlib.pyplot as plt
    fig = plt.figure('spikes by gifs')
    ax = fig.add_axes([0, 0, 1, 1])
    for i in np.arange(0, matrix.shape[0]):
        sys.stdout.write('\r')
        sys.stdout.write('case {0}/{1}'.format(i, matrix.shape[ 0 ]))
        sys.stdout.flush()
        index, spikes = run_specific_comparison(matrix[ i, 0 ], matrix[ i, 1 ],
                                                matrix[ i, 2 ], matrix[ i, 3 ])
        ax.scatter(spikes, i * np.ones(len(spikes)))
    return fig


if __name__ == '__main__':

    cluster = bool(sys.argv[ 2 ])

    # Configure Nest
    import nest

    nest.set_verbosity('M_WARNING')
    nest.ResetKernel()
    dt = 0.1  # the resolution in ms
    nest.SetKernelStatus(
        {"resolution": dt, "print_time": False, "overwrite_files": True})
    if cluster:
        nest.SetKernelStatus({"local_num_threads": 16})
    try:
        nest.Install("gif2_module")
    except:
        pass

    # Simulation setup
    recstart = 1000.0
    simtime = 2500.0

    # Index variables
    p_rate = float(sys.argv[ 1 ])
    C_m2 = 100.0
    indexarray = np.zeros(10)
    p_range = np.arange(50000.0, 125000.0, 5000.0)
    C_range = np.arange(100.0, 400.0, 10.0)  # np.arange(100.0, 600.0, 25.0)
    g_range = np.arange(5.0, 45.0, 2.5)  # np.arange(5.0, 90.0, 5.0)
    cases = len(C_range) * len(g_range) * len(g_range)
    g_m = 5.0
    g_1 = 5.0

    # Shared/fixed neuron parameters
    tauSyn = 0.5
    V_theta = 15.0
    tau_1 = 100.0
    V_range = 6.0
    synweight = 87.8
    J_ex = 0.125
    J_re = 0.125

    j = 0
    headerstr = 'input rate, C, g1, g, tau1, V_range, rate_gif, rate_iaf, cv_gif, cv_iaf'
    with open('excitability_{0}.txt'.format(sys.argv[ 1 ]), 'a') as output:
        np.savetxt(output, np.array([]), header=headerstr)
        for i in product(C_range, g_range, g_range, p_range):
            j += 1
            nest.ResetKernel()
            p_rate = i[ 3 ]

            # adopt neuron parameters
            gif_params = {"tau_1":      tau_1,
                          "C_m":        i[ 0 ],  # C_m2,
                          "tau_syn_ex": tauSyn,
                          "tau_syn_in": tauSyn,
                          "g_rr":       i[ 2 ],  # g_1,
                          "g":          i[ 1 ],  # g_m,
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

            stm_params = {"rate":      p_rate,
                          "amplitude": 0.025 * p_rate,
                          "frequency": 10.0,
                          "phase":     0.0}

            nest.CopyModel("static_synapse", "excitatory",
                           {"weight": J_ex * synweight, "delay": 1.0})

            # Create devices and neurons
            drive = nest.Create("sinusoidal_poisson_generator",
                                params=stm_params)
            iafspikes = nest.Create("spike_detector", params=det_params)
            gifspikes = nest.Create("spike_detector", params=det_params)
            gif = nest.Create("gif2_psc_exp")
            iaf = nest.Create("iaf_psc_exp")
            nest.SetStatus(gif, gif_params)
            nest.SetStatus(iaf, iaf_params)

            # Connect everything
            nest.Connect(drive, gif, syn_spec={"model":  "static_synapse",
                                               "weight": J_ex * synweight,
                                               "delay":  0.5})
            nest.Connect(drive, iaf, syn_spec={"model":  "static_synapse",
                                               "weight": J_ex * synweight,
                                               "delay":  0.5})
            nest.Connect(gif, gifspikes)
            nest.Connect(iaf, iafspikes)

            # Simulate
            if not cluster:
                sys.stdout.write('\r')
                sys.stdout.write('case {0}/{1}'.format(j, cases))
                sys.stdout.flush()
            nest.Simulate(recstart + simtime)

            # Evaluate
            rate_iaf = nest.GetStatus(iafspikes, "n_events")[
                           0 ] / simtime * 1000.
            rate_gif = nest.GetStatus(gifspikes, "n_events")[
                           0 ] / simtime * 1000.
            cv_gif = es.cv(es.isi(nest.GetStatus(gifspikes, "events")
                                  [ 0 ][ "times" ]))
            cv_iaf = es.cv(es.isi(nest.GetStatus(iafspikes, "events")
                                  [ 0 ][ "times" ]))

            # write out
            indexarray = np.array(
                [ p_rate, i[ 0 ], i[ 1 ], i[ 2 ], tau_1, V_range, rate_gif,
                  rate_iaf, cv_gif, cv_iaf ])
            np.savetxt(output, indexarray, fmt="%12.6G")
    footerstr = 'tauSyn={0}, V_theta={1}, V_range={2}, synweight={3}, ' \
                'J_ex={4}'.format(tauSyn, V_theta, V_range, synweight, J_ex)
    print(footerstr)
	# np.savetxt(output, np.array([ ]), footer=footerstr)
    output.close()


# TODO: put p_rate into seperate loop. after each iteration on it, write
    # results to file. Should limit writeout time.