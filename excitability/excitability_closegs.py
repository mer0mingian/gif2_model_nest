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
import matplotlib.pyplot as plt


def compute_str_freq(C, t1, g, g1, verbose=False):
    a = g * t1 / C
    b = g1 * t1 / C
    h1 = (a + b + 1.0)**2
    h2 = (a + 1.0)**2
    try:
        h3 = np.sqrt(h1 - h2)
        fR = np.sqrt(h3 - 1.0) / t1 / np.pi * 500.0
    except:
        return 0
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
    """
    the first thing to do
    """
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
    resonating = np.isclose(f_R, 10.0 * np.ones_like(f_R), atol=3.5)
    output = np.all((nonzeroclose, resonating), axis=0)
    print('Among these, {0} are resonating near 10Hz'.format(np.sum(output)))
    equalcond = np.isclose(compmat[ :, 2 ], compmat[ :, 3 ], rtol=0.0, atol=15.0)
    output = np.all((output, equalcond), axis=0)
    print('{0} of these have similar conductances!'.format(np.sum(output)))
    nonzerocv1 = ~np.isclose(compmat[ :, 8 ], np.zeros_like(compmat[ :, 8 ]), rtol=0.0, atol=1e-3)
    nonzerocv2 = ~np.isclose(compmat[ :, 9 ], np.zeros_like(compmat[ :, 9 ]), rtol=0.0, atol=1e-3)
    output = np.all((output, nonzerocv1, nonzerocv2), axis=0)
    print('{0} of these have a CV different from 0.0.'.format(np.sum(output)))
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
    uniquelines = np.zeros(3)
    unique_c = np.unique(mat[ :, 0 ])
    for i in unique_c:
        unique_c_mat = mat[ mat[ :, 0 ] == i, : ]
        for j in np.unique(unique_c_mat[ :, 1 ]):
            unique_cg_mat = unique_c_mat[ unique_c_mat[ :, 1 ] == j, : ]
            for k in np.unique(unique_cg_mat[ :, 2 ]):
                if k != j:
                    uniquelines = np.vstack((uniquelines, np.array([i, j, k])))
    return uniquelines[ 1:, : ]


def scatter_solutions(matrix):
    """
    make a scatterplot of similar behaviours for parameters C, g, g1 of gif2
    """
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


def run_specific_comparison(C, g, g1, tau_1=100.0, V_dist=6.0, runs=200):
    """
    compare rates and CV of preconfigured iaf and gif with input params at runs
    different input rates.
    :return: matrix with input rates | output gif | output iaf | CV1 | CV2
    """
    import nest
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    dt = 0.1  # the resolution in ms
    nest.SetKernelStatus(
        {"resolution": dt, "print_time": False, "overwrite_files": True})
    try: nest.Install("gif2_module")
    except: pass
    recstart = 1500.0
    simtime = 5000.0
    resultarray = np.zeros(5)
    tauSyn = 0.5
    V_theta = 15.0
    synweight = 87.8
    J_ex = 0.125
    gif_params = {"tau_1":      tau_1,
                  "C_m":        C,  # C_m2,
                  "tau_syn_ex": tauSyn,
                  "tau_syn_in": tauSyn,
                  "g_rr":       g1,  # g_1,
                  "g":          g,  # g_m,
                  "V_m":        0.0,
                  "V_reset":    V_theta - V_dist,
                  "E_L":        0.0,
                  "V_th":       V_theta,
                  "t_ref":      2.0}

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
                                       "delay":  1.0})
    nest.Connect(drive, iaf, syn_spec={"model":  "static_synapse",
                                       "weight": J_ex * synweight,
                                       "delay":  1.0})
    nest.Connect(gif, gifspikes)
    nest.Connect(iaf, iafspikes)
    for p in np.linspace(50000.0, 75000.0, runs):  # 125000.0, runs):
        nest.ResetNetwork()
        nest.SetStatus(gif, gif_params)
        nest.SetStatus(iaf, iaf_params)
        stm_params = {"rate":      p,
                      "amplitude": 0.025 * 0.0,
                      "frequency": 10.0,
                      "phase":     0.0}
        nest.SetStatus(drive, stm_params)
        nest.Simulate(recstart + simtime)
        rate_iaf = nest.GetStatus(iafspikes, "n_events")[ 0 ] / simtime * 1000.
        rate_gif = nest.GetStatus(gifspikes, "n_events")[ 0 ] / simtime * 1000.
        cv_gif = es.cv(es.isi(nest.GetStatus(gifspikes, "events")[ 0 ][ "times" ]))
        cv_iaf = es.cv(es.isi(nest.GetStatus(iafspikes, "events")[ 0 ][ "times" ]))
        indexarray = np.array([ p, rate_gif, rate_iaf, cv_gif, cv_iaf ])
        resultarray = np.vstack((resultarray, indexarray))
    return resultarray[1:, :]


def plot_comparison_resultarray(results, color='k', limity=True, lif=True, add_label=None):
    """
    plots I/F-curve
    :param results: stim rates | gif rates | iaf rates
    :return: figure
    """
    fig = plt.figure('IF-curves')
    plt.plot(results[ :, 0 ], results[ :, 1 ],
             color=color, label='gif2'+add_label)
    if lif:
        plt.plot(results[ :, 0 ], results[ :, 2 ], color='b', label='iaf')
    plt.legend(loc='upper left')
    plt.xlim([50000.0, 75000.0])
    if limity:
        plt.ylim([ 0.0, 100.0 ])
    return fig


def show_solution_rates(matrix):
    """
    plot spike courses of all gifs in solutions
    :param matrix: output from read_solution
    """
    fig = plt.figure('spikes by gifs')
    ax = fig.add_axes([0, 0, 1, 1])
    for i in np.arange(0, matrix.shape[0]):
        sys.stdout.write('\r')
        sys.stdout.write('case {0}/{1}'.format(i, matrix.shape[ 0 ]))
        sys.stdout.flush()
        spikes = run_specific_comparison(matrix[ i, 1 ], matrix[ i, 2 ], matrix[ i, 3 ])
        ax.scatter(spikes, i * np.ones(len(spikes)))
    return fig


def maketenplotsfromsolutions(filename, numgraphs=10):
    """
    work in progress
    """
    a, b = read_solution(filename)
    c = a[ b, : ]
    print('Acquired {0} solutions from file {1}.'.format(c.shape[0], filename))
    assert(c.shape[0] > 10)
    displayindices = np.array(np.ceil(np.linspace(0, c.shape[0], numgraphs, endpoint=False)), dtype=int)
    solution = dict()
    solution[ 'all_results' ] = a
    solution[ 'resultfiltervector' ] = b
    solution[ 'comparison' ] = dict()
    for (i, j) in enumerate(displayindices):
        # counter output
        sys.stdout.write('\r')
        sys.stdout.write('simulating case {0}/{1}'.format(i, numgraphs))
        sys.stdout.flush()
        #actual simulation:
        solution[ 'comparison' ][ i ] = dict()
        solution[ 'comparison' ][ i ][ 'results' ] = \
            run_specific_comparison(c[ j, 1 ], c[ j, 2 ], c[ j, 3 ], runs=21)
        solution[ 'comparison' ][ i ][ 'index' ] = j
    # printing stuff
    sn = solution[ 'comparison' ][ 0 ][ 'results' ]
    plt.clf()
    fig1 = plt.figure('Different gif2 rates')
    for (k, val) in solution[ 'comparison' ].iteritems():
        # counter output
        sys.stdout.write('\r')
        sys.stdout.write('plotting case {0}/{1}'.format(k, numgraphs))
        sys.stdout.flush()
        # actual plotting:
        plt.plot(val['results'][ :, 0 ]/1000., val['results'][ :, 1 ], label='C={0}pF, g={1}nS, g1={2}nS'.format(c[val[ 'index' ], 1], c[val[ 'index' ], 2], c[val[ 'index' ], 3]))
    plt.plot(sn[ :, 0 ]/1000., sn[ :, 2 ], label='iaf neuron: C=250.0pF, g=25.0nS', linestyle="--")
    plt.xlabel('Stimulation rate [spikes/sec]')
    plt.ylabel('Spike response [spikes/sec]')
    plt.legend(loc='upper left')
    plt.xlim([50.0, 100.0])
    return solution, fig1

def num_close_frs(Cs, gs, g1s, t1s, tol=0.1):
    i = 0
    cases = len(Cs) * len(gs) * len(g1s) * len(t1s)
    for j in product(Cs, gs, g1s, t1s):
        sys.stdout.write('\r')
        sys.stdout.write('case {0}/{1}'.format(j, cases))
        sys.stdout.flush()
        fr = compute_str_freq(j[ 0 ], j[ 3 ], j[ 1 ], j[ 2 ])
        if np.isclose(fr, 10.0, atol=tol, rtol=0.0):
            i += 1
    return i


if __name__ == '__main__':

    cluster = bool(sys.argv[ 2 ])
    config = sys.argv[ 1 ]

    # Configure Nest
    import nest

    nest.set_verbosity('M_ERROR')
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
    recstart = 2500.0
    simtime = 2500.0

    # Index variables
    indexarray = np.zeros(10)
    if 'test' in config:
        p_range = np.arange(59000.0, 80000.0, 1000.0)
        C_range = np.array([ 250.0 ])
        g_range = np.array([ 8.0 ])
        V_range = np.array([ 4.5 ])
    else:
        p_range = np.arange(59000.0, 80000.0, 1000.0)
        C_range = np.arange(200.0, 510.0, 10.0)
        g_range = np.arange(12.5, 85.0, 2.5)
        V_range = np.arange(5.0, 12.0, 0.5)

    cases = len(C_range) * len(g_range) * len(V_range)

    # Shared/fixed neuron parameters
    tauSyn = 0.5
    V_theta = -50.0  # 15.0
    synweight = 87.8
    J_ex = 0.125
    J_re = 0.125

    delay = 1.0
    j = 0
    for i in product(np.array([ 0.0 ]), g_range, V_range):
        for g_1 in np.arange(i[ 1 ] - 10.0, i[ 1 ] + 10.0, 2.5):
            V_dist = i[ 2 ]
            C_m = C_range[ int(sys.argv[ 3 ]) ]
            for tau_1 in np.arange( min(5.0 * C_m / i[ 1 ], 80.0), 120.0, 10.0 ):
                for Vdist2 in np.arange( np.abs(-70.0 - V_theta + V_dist), np.abs(-70.0 - V_theta), 5.0):
                    fr = compute_str_freq(C_m, tau_1, i[ 1 ], g_1)
                    if np.isclose(fr, 10.0, atol=0.05, rtol=0.0):
                        j += 1
                        # check for resonance frequency here
                        nest.ResetKernel()
    
                        # adopt neuron parameters
                        gif_params = {"tau_1":      tau_1,
                                      "C_m":        C_m,  # C_m2,
                                      "tau_syn_ex": tauSyn,
                                      "tau_syn_in": tauSyn,
                                      "g_rr":       g_1,  # g_1,
                                      "g":          i[ 1 ],  # g_m,
                                      "V_m":        V_theta - V_dist,
                                      "V_reset":    V_theta - V_dist,
                                      "E_L":        -70.0 + Vdist2,
                                      "V_th":       V_theta}
    
                        iaf_params = {"C_m":        250.0,
                                      "tau_m":      10.0,
                                      "tau_syn_ex": tauSyn,
                                      "tau_syn_in": tauSyn,
                                      "t_ref":      2.0,
                                      "E_L":        -65.0,
                                      "V_reset":    -65.0,
                                      "V_m":        -65.0,
                                      "V_th":       V_theta}
    
                        det_params = {"withtime": True,
                                      "withgid":  False,
                                      "to_file":  False,
                                      "start":    recstart}
    
                        stm_params = {"rate":      0.0}
    
        #                nest.CopyModel("static_synapse", "excitatory",
        #                               {"weight": J_ex * synweight, "delay": 1.0})
    
                        # Create devices and neurons
                        drive = nest.Create("poisson_generator",
                                            params=stm_params)
                        iafspikes = nest.Create("spike_detector", params=det_params)
                        gifspikes = nest.Create("spike_detector", params=det_params)
                        gif = nest.Create("gif2_psc_exp", n=50)
                        iaf = nest.Create("iaf_psc_exp", n=50)
                        nest.SetStatus(gif, gif_params)
                        nest.SetStatus(iaf, iaf_params)
    
                        # Connect everything
                        nest.Connect(drive, gif, syn_spec={"model":  "static_synapse", "weight": J_ex * synweight, "delay":  delay})
                        nest.Connect(drive, iaf, syn_spec={"model":  "static_synapse", "weight": J_ex * synweight, "delay":  delay})
                        nest.Connect(gif, gifspikes)
                        nest.Connect(iaf, iafspikes)
    
                        # Simulate
                        if not cluster:
                            sys.stdout.write('\r')
                            sys.stdout.write('case {0}/{1}'.format(j, cases))
                            sys.stdout.flush()
    
                        rate_iaf = np.zeros_like(p_range)
                        rate_gif = np.zeros_like(p_range)
                        cv_gif = np.zeros_like(p_range)
                        cv_iaf = np.zeros_like(p_range)
    
                        for index, rate in enumerate(p_range):
                            nest.ResetNetwork()
                            stm_params[ 'rate' ] = rate
                            nest.SetStatus(drive, params=stm_params)
                            nest.Simulate(recstart + simtime)
                            # Evaluate
                            rate_iaf[ index ] = nest.GetStatus(iafspikes, "n_events")[ 0 ] / simtime * 1000. / 50.0
                            rate_gif[ index ] = nest.GetStatus(gifspikes, "n_events")[ 0 ] / simtime * 1000. / 50.0
                            cv_gif[ index ] = es.cv(es.isi(nest.GetStatus(gifspikes, "events")[ 0 ][ "times" ]))
                            cv_iaf[ index ] = es.cv(es.isi(nest.GetStatus(iafspikes, "events")[ 0 ][ "times" ]))
    
                        # compute the rmse
                        diff = rate_iaf - rate_gif
                        se = np.dot(diff.T, diff)
                        rmse = np.sqrt(np.mean(se))
                        # compute the rmse for iaf rate between 0.5 and 80.0 Sp/s
                        gtzero = rate_iaf > 0.5
                        lteighty = rate_iaf >= 80.0
                        roi2 = np.any((gtzero, lteighty), axis=0) - np.all((gtzero, lteighty), axis=0)
                        lroi = np.sum(roi2)
                        diff2 = rate_iaf[ roi2 ] - rate_gif[ roi2 ]
                        se2 = np.dot(diff2.T, diff2)
                        rmse2 = np.sqrt(np.mean(se2))
                        # compute the rmse around 80.0 pm 5 Sp/s
                        gt75 = rate_iaf > 75.0
                        lt85 = rate_iaf >= 85.0
                        roi3 = np.any((gt75, lt85), axis=0) - np.all((gt75, lt85), axis=0)
                        lroi3 = np.sum(roi3)
                        diff3 = rate_iaf[ roi3 ] - rate_gif[ roi3 ]
                        se3 = np.dot(diff3.T, diff3)
                        rmse3 = np.sqrt(np.mean(se3))
    
    
                        #rwiteout conditions:
                        if (rmse != 0.0) and (lroi > 0):
                            # write out
                            indexarray = np.array(
                                [ fr, C_m, i[ 1 ], g_1, tau_1, i[ 2 ], Vdist2, rmse, rmse2,
                                    lroi, rmse3, lroi3])
                            with open('closegs/AdvExcitability_{0}_C{1}_del{2}.csv'.format(sys.argv[ 1 ], int(C_m), delay), 'a') as output:
                                np.savetxt(output, indexarray, fmt="%12.6G", newline=' ')
                                output.write(' \n')
                                output.close()
    

# TODO: put p_rate into seperate loop. after each iteration on it, write
    # results to file. Should limit writeout time.