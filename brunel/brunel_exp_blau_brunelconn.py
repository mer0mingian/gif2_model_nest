"""
Version 9.5
Option
alternative_connection = False


With p_rate2 = 85000. and h = 1.5, the firing rates of the pops are equal.
With p_rate2 = 70000. and h = 2.5, the firing rates of the pops are equal.
With p_rate2 = 66000. and h = 1.9, the firing rates of the pops are equal.
"""

import nest
import nest.raster_plot
import numpy as np
import time
import pylab
from numpy import exp
import sys
import matplotlib.pyplot as plt
from mingtools1 import *

# Kernel parameters:
nest.set_verbosity('M_WARNING')
nest.ResetKernel()

frac_range = np.linspace(0., 1., int(sys.argv[ 2 ]))
frac = frac_range[ int(sys.argv[ 1 ]) - 1 ]
dt = 0.1  # the resolution in ms
nest.SetKernelStatus({"resolution": dt, "print_time": False, "overwrite_files": True, "local_num_threads": 8})
startbuild = time.time()
try:
    nest.Install("gif2_module")
except:
    pass

# Connect routine:
alternative_connection = True

# Simulation parameters
recstart = 0.0
simtime = 2000.0    # Simulation time in ms
delay = 1.0         # synaptic delay in ms
g = 5.0             # ratio inhibitory weight/excitatory weight
eta = 2.0           # external rate relative to threshold rate
epsilon = 0.1       # connection probability
print('ext rate: {0}, connection probability: {1}'.format(eta, epsilon))

# Network parameters
order = 2500
NE = int(2 * (1. - frac) * order)   # number of excitatory neurons
NR = int(2 * frac * order)          # number of excitatory neurons
NI = int(1 * order)                 # number of inhibitory neurons
N_neurons = NE + NI + NR            # number of neurons in total
N_rec = 150                         # record from N_rec neurons per population
Nred = np.array([ NE, NR, NI ]) * N_rec / N_neurons

# Since we're connecting by indegree, we can just add all sources again.
CE = int(epsilon * NE)  # number of excitatory synapses per neuron
CR = int(epsilon * NR)  # number of resonating synapses per neuron
CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
C_tot = int(CI + CE + CR)  # total number of synapses per neuron

tauMem = 10.0  # time constant of membrane potential in ms
theta = 15.0  # membrane threshold potential in mV
tauSyn = 0.5
C_m = 250.0
neuron_params = {
                "C_m":        C_m,
                "tau_m":      tauMem,
                "tau_syn_ex": tauSyn,
                "tau_syn_in": tauSyn,
                "t_ref":      2.0,
                "E_L":        0.0,
                "V_reset":    0.0,
                "V_m":        0.0,
                "V_th":       theta}

tau_1 = 100.0
C_m2 = 450.
g_1 = 25.0  # 85.0
g_m = 25.0  # 35.0
neuron_params2 = {
                "tau_1":      tau_1,
                "C_m":        C_m2,
                "tau_syn_ex": tauSyn,
                "tau_syn_in": tauSyn,
                "g_rr":       g_1,
                "g":          g_m,
                "V_m":        0.0,
                "V_reset":    9.0,
                "E_L":        0.0,
                "V_th":       15.0}

alpha = g * tau_1 / C_m     # regime determining effective leak
beta = g_1 * tau_1 / C_m    # regime determining effective coupling
factor = (alpha - 1) ** 2 / 4
exp_f_r = 1. / (2. * np.pi * tau_1) * np.sqrt(np.sqrt( (alpha + beta + 1.) * (alpha + beta + 1.) - (alpha + 1.) * (alpha + 1.)) - 1.)

synweight = 87.8
J = 0.125           # postsynaptic amplitude in mV
J_ex = J            # amplitude of excitatory postsynaptic potential
J_in = -g * J_ex    # amplitude of inhibitory postsynaptic potential
h = 1.9  # 1.9      # ratio resonating weight/excitatory weight
J_re = h * J        # amplitude of resonating postsynaptic potential

# Stimulation rate parameters
nu_th = theta / (J * CE * tauMem * exp(1) * tauSyn)
nu_ex = eta * nu_th
p_rate = 1000.0 * nu_ex * CE * 2.75  # remove C_m
p_rate2 = 66000.0

print('threshold rate: {0}, external rate: {1}, stim rate: {2}'.format(nu_th, nu_ex, p_rate))

# Alternative connectivity routine:
if alternative_connection:
    # synaptic weights matrix: J_ij is the weight for a connection from i to j
    J = np.array([[J_ex, h*J_ex, J_ex], [J_ex, h*J_ex, J_ex], [-g*J_ex, -g*J_ex, -g*J_ex]])
    # connectivity matrix: C_ij is the weight for a connection from i to j
    C = np.array([[0.083, 0.083, 0.06], [0.083, 0.083, 0.06], [0.373, 0.373, 0.316]])
    C[0,:] *= NE
    C[1,:] *= NR
    C[2,:] *= NI
    C = np.array(C, dtype=int)


print("Building network")
nest.SetDefaults("iaf_psc_exp", neuron_params)
nest.SetDefaults("gif2_psc_exp", neuron_params2)
# nest.SetDefaults("poisson_generator", {"rate": p_rate})
nodes_ex = nest.Create("iaf_psc_exp", NE)
nodes_re = nest.Create("gif2_psc_exp", NR)
nodes_in = nest.Create("iaf_psc_exp", NI)
noise = nest.Create("sinusoidal_poisson_generator")
espikes = nest.Create("spike_detector")
rspikes = nest.Create("spike_detector")
ispikes = nest.Create("spike_detector")
spikedetector_raster = nest.Create("spike_detector", params={'start': recstart})
noise2 = nest.Create("sinusoidal_poisson_generator")

multi = nest.Create("multimeter")
nest.SetStatus(multi, params={"interval": dt, "record_from": [ "V_m", "w" ], "withgid": True} )

nest.SetStatus(noise, [ {"rate": p_rate2, "amplitude": 0.025 * p_rate2, "frequency": 10.0, "phase": 0.0 } ] )
nest.SetStatus(noise2, [ {"rate": p_rate2, "amplitude": 0.025 * p_rate2, "frequency": 10.0 } ])
nest.SetStatus(espikes, [ {"label": "brunel-py-ex", "withtime": True, "withgid":  True, "to_file":  False} ])
nest.SetStatus(rspikes, [ {"label": "brunel-py-res", "withtime": True, "withgid":  True, "to_file":  False} ])
nest.SetStatus(ispikes, [ {"label": "brunel-py-in", "withtime": True, "withgid":  True, "to_file":  False} ])

print("Connecting devices")

nest.CopyModel("static_synapse", "excitatory", {"weight": J_ex * synweight, "delay": delay})
nest.CopyModel("static_synapse", "resonating", {"weight": J_re * synweight, "delay": delay})
nest.CopyModel("static_synapse", "inhibitory", {"weight": J_in * synweight, "delay": delay})

nest.Connect(noise, nodes_ex, syn_spec="excitatory")
nest.Connect(noise, nodes_in, syn_spec="excitatory")
if alternative_connection:
    nest.Connect(noise2, nodes_re, syn_spec="resonating")
else:
    nest.Connect(noise, nodes_re, syn_spec="excitatory")

nest.Connect(nodes_ex, espikes, syn_spec="excitatory")
nest.Connect(nodes_re, rspikes, syn_spec="excitatory")
nest.Connect(nodes_in, ispikes, syn_spec="excitatory")
nest.Connect(nodes_ex[ :Nred[ 0 ] ], spikedetector_raster, syn_spec="excitatory")
nest.Connect(nodes_re[ :Nred[ 1 ] ], spikedetector_raster, syn_spec="excitatory")
nest.Connect(nodes_in[ :Nred[ 2 ] ], spikedetector_raster, syn_spec="excitatory")

print("Connecting network")
if not alternative_connection:
    nest.Connect(nodes_ex, nodes_ex + nodes_re + nodes_in, conn_spec={'rule': 'fixed_indegree', 'indegree': CE}, syn_spec="excitatory")
    nest.Connect(nodes_re, nodes_ex + nodes_re + nodes_in, conn_spec={'rule': 'fixed_indegree', 'indegree': CR}, syn_spec="resonating")
    nest.Connect(nodes_in, nodes_ex + nodes_re + nodes_in, conn_spec={'rule': 'fixed_indegree', 'indegree': CI}, syn_spec="inhibitory")

if alternative_connection:
    print("Excitatory connections")
    nest.Connect(nodes_ex, nodes_ex, conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 0 ]}, syn_spec={'weight': J[ 0, 0 ], "delay": delay})
    nest.Connect(nodes_ex, nodes_re, conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 1 ]}, syn_spec={'weight': J[ 0, 1 ], "delay": delay})
    nest.Connect(nodes_ex, nodes_in, conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 2 ]}, syn_spec={'weight': J[ 0, 2 ], "delay": delay})
    print("Resonating connections")
    nest.Connect(nodes_re, nodes_ex, conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 0 ]}, syn_spec={'weight': J[ 1, 0 ], "delay": delay})
    nest.Connect(nodes_re, nodes_re, conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 1 ]}, syn_spec={'weight': J[ 1, 1 ], "delay": delay})
    nest.Connect(nodes_re, nodes_in, conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 2 ]}, syn_spec={'weight': J[ 1, 2 ], "delay": delay})
    print("Inhibitory connections")
    nest.Connect(nodes_in, nodes_ex, conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 2, 0 ]}, syn_spec={'weight': J[ 2, 0 ], "delay": delay})
    nest.Connect(nodes_in, nodes_re, conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 2, 1 ]}, syn_spec={'weight': J[ 2, 1 ], "delay": delay})
    nest.Connect(nodes_in, nodes_in, conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 2, 2 ]}, syn_spec={'weight': J[ 2, 2 ], "delay": delay})

endbuild = time.time()

print("Simulating")
nest.Simulate(simtime)
endsimulate = time.time()

spikesenders_raster = nest.GetStatus(spikedetector_raster, "events")[ 0 ][ 'senders' ]
spiketimes_raster = nest.GetStatus(spikedetector_raster, "events")[ 0 ][ 'times' ]
events_ex = nest.GetStatus(espikes, "n_events")[ 0 ]
events_re = nest.GetStatus(rspikes, "n_events")[ 0 ]
events_in = nest.GetStatus(ispikes, "n_events")[ 0 ]
rate_ex = events_ex / simtime * 1000.0 / N_rec
rate_re = events_re / simtime * 1000.0 / N_rec
rate_in = events_in / simtime * 1000.0 / N_rec
num_synapses = nest.GetDefaults("excitatory")[ "num_connections" ] + nest.GetDefaults("inhibitory")[ "num_connections" ]
build_time = endbuild - startbuild
sim_time = endsimulate - endbuild

print("Brunel network simulation (Python)")
print("Number of neurons : {0}".format(N_neurons))
print("Number of synapses: {0}".format(num_synapses))
print("       Exitatory  : {0}".format(int(CE * N_neurons) + N_neurons))
print("       Resonator  : {0}".format(int(CR * N_neurons) + N_neurons))
print("       Inhibitory : {0}".format(int(CI * N_neurons)))
print("Excitatory rate   : %.2f Hz" % rate_ex)
print("Resonating rate   : %.2f Hz" % rate_re)
print("Inhibitory rate   : %.2f Hz" % rate_in)
print("Building time     : %.2f s" % build_time)
print("Simulation time   : %.2f s" % sim_time)

# a = nest.raster_plot.from_device(espikes, hist=True)
# pylab.savefig('Rasterplot_{0}.png'.format(int(sys.argv[ 1 ])))

figu = plt.figure("Rasterplots")
plt.clf()
spikesenders_raster2 = np.zeros_like(spikesenders_raster)
spikesenders_raster2 = spikesenders_raster
spikesenders_raster2[ spikesenders_raster >= NE + NR ] -= (NE + NR) - Nred[ 1 ] - Nred[ 0 ]
spikesenders_raster2[ spikesenders_raster >= NR ] -= NR - Nred[ 1 ]
ax1 = show_Rasterplot(figu, spiketimes_raster, spikesenders_raster2, recstart, simtime, N_rec, 111, 0)
ax1.text(0.01, 0.99,
        '$C = ${0}, $g = ${3}, $g_1 = ${1}, $t_1 = ${2}, $p = ${4}, $h=${5}, $p_2 =${6}'.format(C_m2, g_1, tau_1, g_m, p_rate, h, p_rate2),
        horizontalalignment='left', verticalalignment='top', fontsize=9, transform=ax1.transAxes)
plt.savefig('Rasterplot_{0}.png'.format(int(sys.argv[ 1 ])))