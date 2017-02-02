# coding=utf-8

import numpy as np
from itertools import product
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})


def find_Nsyn(Cold, Npre, Npost):
    """
    compute indegree * Npost = actual number of synapses
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
    h1 = (Npre * Npost - 1.) / (Npre * Npost)
    h2 = np.power(h1, K)
    h3 = 1. - h2
    return h3


def compute_new_connectivity(conn_probs, N_full_new, N_full_old, layers, pops, newpopdict, new_structure, K_bg=None):
    """
    the core function.
    """
    n_new_pops_per_layer = len(newpopdict)
    n_pops_per_layer = len(pops)
    n_layers = len(layers)
    connmat = np.zeros((n_layers * n_new_pops_per_layer, n_layers * n_new_pops_per_layer))
    if K_bg is not None:
        K_bg = np.array(K_bg, dtype=float)
        K_bg_new = np.zeros((n_layers, n_new_pops_per_layer))

    for target_layer, target_layer_dict in N_full_new.iteritems():
        for target_pop, target_pop_size in target_layer_dict.iteritems():
            bindex_t = new_structure[ target_layer ][ target_pop ]

            # here we compute the new background indegree
            if K_bg is not None:
                if target_pop_size == 0:
                    K_bg_new.flat[ bindex_t ] = 0
                else:
                    if 'R' in target_pop:  # take target connectivity from E for R?
                        target_index = n_pops_per_layer * layers[ target_layer ] + pops[ 'E' ]
                        Npost_old = N_full_old[ target_layer ][ 'E' ]
                    else:  # just E and I populations connecting
                        target_index = n_pops_per_layer * layers[ target_layer ] + pops[ target_pop ]
                        Npost_old = N_full_old[ target_layer ][ target_pop ]
                    rel_bg = float(target_pop_size) / float(Npost_old)
                    K_bg_new.flat[ bindex_t ] = K_bg.flat[ target_index ] * rel_bg

            for source_layer, source_layer_dict in N_full_new.iteritems():
                for source_pop, source_pop_size in source_layer_dict.iteritems():
                    bindex_s = new_structure[ source_layer ][ source_pop ]
                    if source_pop_size == 0 or target_pop_size == 0:                                        # target or source have size 0?
                        connmat[ bindex_t, bindex_s ] = 0
                    else:
                        # collect the old connectivity:
                        if 'R' in source_pop:                                                               # take source connectivity from E for R?
                            source_index = n_pops_per_layer * layers[ source_layer ] + pops[ 'E' ]          # get the source index for the old connectivity list
                            Npre_old = N_full_old[ source_layer ][ 'E' ]
                        else:
                            source_index = n_pops_per_layer * layers[ source_layer ] + pops[ source_pop ]   # get the source index for the old connectivity list
                            Npre_old = N_full_old[ source_layer ][ source_pop ]
                        if 'R' in target_pop:                                                               # take target connectivity from E for R?
                            target_index = n_pops_per_layer * layers[ target_layer ] + pops[ 'E' ]          # get the target index for the old connectivity list
                            Npost_old = N_full_old[ target_layer ][ 'E' ]
                        else:                                                                               # just E and I populations connecting
                            target_index = n_pops_per_layer * layers[ target_layer ] + pops[ target_pop ]   # get the target index for the old connectivity list
                            Npost_old = N_full_old[ target_layer ][ target_pop ]
                        Cold = conn_probs[ target_index ][ source_index ]                                   # get the 'old' connectivity list entry

                        # compute new connectivity:
                        if Cold == 0:                                                                       # was connectivity 0 anyway?
                            connmat[ bindex_t, bindex_s ] = 0
                        else:                                                                               # compute new connectivity
                            Nsyn_new = find_Nsyn(Cold, source_pop_size, target_pop_size)                    # replaces K with Nsyn, find_C with find_Nsyn
                            rel = float(source_pop_size) / float(Npre_old) * float(target_pop_size) / float(Npost_old)                  # number of synapses with same relation as between old population sizes

                            C = find_new_C(Nsyn_new * rel, N_full_new[ source_layer ][ source_pop ], N_full_new[ target_layer ][ target_pop ])
                            connmat[ bindex_t, bindex_s ] = C
    if K_bg is None:
        return connmat
    else:
        return connmat, K_bg_new



if __name__ == "__main__":

    microcircuit = True
    if microcircuit:
        #             2/3e      2/3i    4e      4i      5e      5i      6e      6i
        conn_probs = [ [ 0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.,     0.0076, 0.     ],
                       [ 0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.,     0.0042, 0.     ],
                       [ 0.0077, 0.0059, 0.0497, 0.135,  0.0067, 0.0003, 0.0453, 0.     ],
                       [ 0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.,     0.1057, 0.     ],
                       [ 0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.     ],
                       [ 0.0548, 0.0269, 0.0257, 0.0022, 0.06,   0.3158, 0.0086, 0.     ],
                       [ 0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252 ],
                       [ 0.0364,  0.001, 0.0034, 0.0005, 0.0277, 0.008,  0.0658, 0.1443 ] ]

        layers = {'L23': 0, 'L4': 1, 'L5': 2, 'L6': 3}
        pops = {'E': 0, 'I': 1}
        newpopdict = {'E': 0, 'I': 1, 'R': 2}

        new_structure = {'L23': {'E': 0, 'I': 1, 'R': 2},
                         'L4':  {'E': 3, 'I': 4, 'R': 5},
                         'L5':  {'E': 6, 'I': 7, 'R': 8},
                         'L6':  {'E': 9, 'I': 10, 'R': 11}}

        # Numbers of neurons in full-scale model
        N_full_new = {
            'L23': {'E': 20683, 'I': 5834, 'R': 0},
            'L4':  {'E': 21915, 'I': 5479, 'R': 0},
            'L5':  {'E': 2425, 'I': 1065, 'R': 2425},
            'L6':  {'E': 14395, 'I': 2948, 'R': 0}
        }

        N_full_old = {
            'L23': {'E': 20683, 'I': 5834},
            'L4':  {'E': 21915, 'I': 5479},
            'L5':  {'E': 4850, 'I': 1065},
            'L6':  {'E': 14395, 'I': 2948}
        }

        K_bg = [ [ 1600, 1500], [ 2100, 1900], [ 2000, 1900], [ 2900, 2100] ]

        connmat, K_bg_new = compute_new_connectivity(conn_probs, N_full_new, N_full_old, layers, pops, K_bg)
        print('{0} \n \n {1}'.format(connmat, K_bg_new))

    else:
        conn_probs = [ [ 0.0831, 0.3726 ], [ 0.060, 0.3158 ] ]
        layers = {'L5': 0}
        pops = {'E': 0, 'I': 1}
        newpopdict = {'E': 0, 'I': 1, 'R': 2}
        new_structure = {'L5': {'E': 0, 'I': 1, 'R': 2}}
        N_full_new = {'L5': {'E': 2425, 'I': 1065, 'R': 2425}}
        N_full_old = {'L5': {'E': 4050, 'I': 1065}}
        K_bg = [ [ 2000, 1900 ] ]

        connmat = compute_new_connectivity(conn_probs, N_full_new,
                                                     N_full_old, layers, pops)
        # print('{0} \n \n {1}'.format(connmat))



        # let us test, whether the number of synapses has stayed the same:
    # full_scale_num_neurons_new = np.array([ 20683, 5834, 0, 21915, 5479, 0, 2425, 1065, 2425, 14395, 2948, 0 ])
    # full_scale_num_neurons_old = np.array([ 20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948 ])
    # synapses_before_transformation = np.dot(full_scale_num_neurons_old, np.dot(np.array(conn_probs), full_scale_num_neurons_old))
    # synapses_after_transformation = np.dot(full_scale_num_neurons_new, np.dot(connmat, full_scale_num_neurons_new))
    # synapses_before_transformation = np.sum(np.dot(full_scale_num_neurons_old, np.array(conn_probs)))
    # synapses_after_transformation = np.sum(np.dot(full_scale_num_neurons_new, connmat))
    # print('Synapses before transformation: {0}\n Synapses after transformation: {1}'.format(synapses_before_transformation, synapses_after_transformation))
    # print(np.sum(connmat), np.sum(conn_probs))