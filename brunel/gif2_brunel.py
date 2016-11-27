"""
Version 9.5
Option
alternative_connection = False

This version omits the fourth spike detector for the rasterplot and extracts
all data from the three spike detectors.

With p_rate2 = 85000. and h = 1.5, the firing rates of the pops are equal.
With p_rate2 = 70000. and h = 2.5, the firing rates of the pops are equal.
With p_rate2 = 66000. and h = 1.9, the firing rates of the pops are equal.
"""

import numpy as np
import time
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mingtools1 import *
from elephant.statistics import isi, cv


# def run_specific_network(somevec):
import nest
import nest.raster_plot
# Kernel parameters:
dt = 0.1
nest.set_verbosity('M_WARNING')
nest.ResetKernel()
nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
try:
    nest.Install("gif2_module")
except:
    pass
somevec = np.array(sys.argv[ 1:7 ], dtype=float)

p_rate, C_m, gm, g1, tau_1, theta1 = somevec[ 0:6 ]
recstart = 1500.0
simtime = 2000.0  # Simulation time in ms
delay = 1.0  # synaptic delay in ms
delay_ex = 1.5
delay_in = 0.8
g = 5.0  # ratio inhibitory weight/excitatory weight

NE = 2425  #int(5000 / order)
NR = 2425  #int(5000 / order)
NI = 1065  #int(2500 / order)
N_neurons = NE + NI + NR  # number of neurons in total
N_rec = 200  # record from N_rec neurons per population

theta = 15.0  # membrane threshold potential in mVfrom stats
tauSyn = 0.5

neuron_params = {
                "C_m":        250.0,
                "tau_m":      10.0,
                "tau_syn_ex": tauSyn,
                "tau_syn_in": tauSyn,
                "t_ref":      1.5,
                "E_L":        0.0,
                "V_reset":    0.0,
                "V_m":        0.0,
                "V_th":       theta}

neuron_params2 = {
                "tau_1":      tau_1,
                "C_m":        C_m,
                "tau_syn_ex": tauSyn,
                "tau_syn_in": tauSyn,
                "g_rr":       g1,
                "g":          gm,
                "V_m":        0.0,
                "V_reset":    0.0,
                "E_L":        0.0,
                "V_th":       theta1,
                "t_ref":      1.5}

synweight = 87.8
J = 0.125  # postsynaptic amplitude in mV
J_ex = J  # amplitude of excitatory postsynaptic potential
J_in = -g * J

J = np.array([ [ J_ex, J_ex, J_ex ],
               [ J_ex, J_ex, J_ex ],
               [ J_in, J_in, J_in ] ])
C = np.array([ [ 0.083, 0.083, 0.06 ],
               [ 0.083, 0.083, 0.06 ],
               [ 0.373, 0.373, 0.316 ] ])
C[ 0, : ] *= NE
C[ 1, : ] *= NR
C[ 2, : ] *= NI
C = np.array(C, dtype=int)

print("Building network")
startbuild = time.time()
nest.SetDefaults("iaf_psc_exp", neuron_params)
nest.SetDefaults("gif2_psc_exp", neuron_params2)
nodes_ex = nest.Create("iaf_psc_exp", NE)
nodes_re = nest.Create("gif2_psc_exp", NR)
nodes_in = nest.Create("iaf_psc_exp", NI)
noise = nest.Create("sinusoidal_poisson_generator")
espikes = nest.Create("spike_detector")
rspikes = nest.Create("spike_detector")
ispikes = nest.Create("spike_detector")

nest.SetStatus(noise, [
    {"rate":  p_rate, "amplitude": 0.025 * p_rate, "frequency": 10.0,
     "phase": 0.0} ])
nest.SetStatus(espikes, [
    {"label":   "brunel-py-ex", "withtime": True, "withgid": True,
     "to_file": False, 'start': recstart} ])
nest.SetStatus(rspikes, [
    {"label":   "brunel-py-res", "withtime": True, "withgid": True,
     "to_file": False, 'start': recstart} ])
nest.SetStatus(ispikes, [
    {"label":   "brunel-py-in", "withtime": True, "withgid": True,
     "to_file": False, 'start': recstart} ])

print("Connecting devices")
nest.CopyModel("static_synapse", "excitatory",
               {"weight": J_ex * synweight, "delay": delay})

nest.Connect(noise, nodes_ex, syn_spec="excitatory")
nest.Connect(noise, nodes_in, syn_spec="excitatory")
nest.Connect(noise, nodes_re, syn_spec="excitatory")
nest.Connect(nodes_ex[ 0:N_rec - 1 ], espikes, syn_spec="excitatory")
nest.Connect(nodes_re[ 0:N_rec - 1 ], rspikes, syn_spec="excitatory")
nest.Connect(nodes_in[ 0:N_rec - 1 ], ispikes, syn_spec="excitatory")

print("Connecting network")
print("Excitatory connections")
nest.Connect(nodes_ex, nodes_ex,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 0 ]},
             syn_spec={'weight': J[ 0, 0 ], "delay": delay_ex})
nest.Connect(nodes_ex, nodes_re,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 1 ]},
             syn_spec={'weight': J[ 0, 1 ], "delay": delay_ex})
nest.Connect(nodes_ex, nodes_in,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 2 ]},
             syn_spec={'weight': J[ 0, 2 ], "delay": delay_ex})
print("Resonating connections")
nest.Connect(nodes_re, nodes_ex,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 0 ]},
             syn_spec={'weight': J[ 1, 0 ], "delay": delay_ex})
nest.Connect(nodes_re, nodes_re,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 1 ]},
             syn_spec={'weight': J[ 1, 1 ], "delay": delay_ex})
nest.Connect(nodes_re, nodes_in,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 2 ]},
             syn_spec={'weight': J[ 1, 2 ], "delay": delay_ex})
print("Inhibitory connections")
nest.Connect(nodes_in, nodes_ex,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 2, 0 ]},
             syn_spec={'weight': J[ 2, 0 ], "delay": delay_in})
nest.Connect(nodes_in, nodes_re,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 2, 1 ]},
             syn_spec={'weight': J[ 2, 1 ], "delay": delay_in})
nest.Connect(nodes_in, nodes_in,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 2, 2 ]},
             syn_spec={'weight': J[ 2, 2 ], "delay": delay_in})

endbuild = time.time()

print("Simulating")
nest.Simulate(simtime+recstart)
endsimulate = time.time()

print('Computing results')
events_ex = nest.GetStatus(espikes, "events")[ 0 ]
events_re = nest.GetStatus(rspikes, "events")[ 0 ]
events_in = nest.GetStatus(ispikes, "events")[ 0 ]
nevents_ex = nest.GetStatus(espikes, "n_events")[ 0 ]
nevents_re = nest.GetStatus(rspikes, "n_events")[ 0 ]
nevents_in = nest.GetStatus(ispikes, "n_events")[ 0 ]
rate_ex = nevents_ex / simtime * 1000.0 / N_rec
rate_re = nevents_re / simtime * 1000.0 / N_rec
rate_in = nevents_in / simtime * 1000.0 / N_rec
print('Done. \n')

build_time = endbuild - startbuild
sim_time = endsimulate - endbuild

print("Brunel network simulation (Python)")
print("Number of neurons : {0}".format(N_neurons))
print("Excitatory rate   : %.2f Sp/s" % rate_ex)
print("Resonating rate   : %.2f Sp/s" % rate_re)
print("Inhibitory rate   : %.2f Sp/s" % rate_in)
print("Building time     : %.2f s" % build_time)
print("Simulation time   : %.2f s" % sim_time)

# CVs:
spiketrains_ex = list()
spiketrains_in = list()
spiketrains_re = list()
for gid in nodes_ex:
    spiketrains_ex.append(
        events_ex[ 'times' ][ events_ex[ 'senders' ] == gid ])
for gid in nodes_re:
    spiketrains_re.append(
        events_re[ 'times' ][ events_re[ 'senders' ] == gid ])
for gid in nodes_in:
    spiketrains_in.append(
        events_in[ 'times' ][ events_in[ 'senders' ] == gid ])
spiketrains_allex = spiketrains_ex + spiketrains_re
spiketrains_all = spiketrains_allex + spiketrains_in
cv_ex = np.mean([ cv(isi(spiketrain)) for spiketrain in spiketrains_ex ])
cv_re = np.mean([ cv(isi(spiketrain)) for spiketrain in spiketrains_re ])
cv_in = np.mean([ cv(isi(spiketrain)) for spiketrain in spiketrains_in ])
cv_allex = np.mean([ cv(isi(spiketrain)) for spiketrain in spiketrains_allex ])
cv_all = np.mean([ cv(isi(spiketrain)) for spiketrain in spiketrains_all ])
print('mean CV for sim: {0}'.format(cv_all))
print('CVex {0}, CVre {1}, CVin {2}, CVexall{3}'.format(cv_ex, cv_re, cv_in,
                                                        cv_allex))
print('resonating spikes: {0}'.format(nevents_re))
print('expected resonance frequency : {0}'.format(predict_str_freq(tau_1, gm, g1, C_m, remote=True)))
print(    'C = {0}, g = {2}, g_1 = {1}, t_1 = 100.0, p = {3}, '.format(
        C_m, g1, gm, p_rate))

# print('Creating plot')
# plt.clf()
# # Raster:
# figu = plt.figure("Rasterplots")
# for i, spiketrain in enumerate(spiketrains_all):
#     if spiketrain[ 0 ] is not None:
#         t = np.arange(recstart, simtime, dt)
#         plt.plot(t, i * spiketrain, 'k.', markersize=1)
# plt.savefig('specificRasterplot.png')
# return 0

# ####################################################################################################################################################

# if __name__ == '__main__':
# p_rate, C, gm, g1, order
#    somevec = sys.argv[ 1:6 ]
#    run_specific_network(somevec)
