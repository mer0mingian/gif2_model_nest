# -*- coding: utf-8 -*-
import time
import matplotlib.pyplot as plt
import nest
import nest.raster_plot
import nest.voltage_trace
from numpy import exp
import numpy as np
import pylab
from mingtools.mingtools1 import *
nest.set_verbosity('M_WARNING')
nest.ResetKernel()
dt = 0.1  # the resolution in ms
nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})
try:
	nest.Install("gif2_module")
except:
	print ("The GIF2 model had already been installed")
nest.set_verbosity('M_WARNING')

# plotting routines


flag_plot_voltage = False


rectime = 1500.0
recstart = 1000.0
simtime = rectime + recstart  # Simulation time in ms
delay = 2.0  # synaptic delay in ms
p_rate = 13500.

# neuron counts
ex_times_in_nodes = 4.0
fraction_resonators_ex = 0.5
# fraction_resonators_in = 0.0
order = 500
NE = 2425
NR = 2425
NI = 1065
N_neurons = NE + NI + NR  # number of neurons in total
N_rec = 50  # record from 50 neurons
rasterrows = 50
Nred = np.array([NR, NE, NI]) * rasterrows / N_neurons

# Synapse & Connectivity parameters
g = 5.0  # ratio inhibitory weight/excitatory weight
h = 1. #0.5
l = 1. #0.6
l2 = (l + 1.)/2.
J_ex = 87.8
J_in = -g * J_ex
eta = 2.0  # external rate relative to threshold rate
epsilon = 0.5  # connection probability
# CE should be equal for resonators and integrators for now
CE = int(epsilon * NE / 2)  # number of excitatory synapses per neuron
CR = int(epsilon * NR / 2)
CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
C_tot = int(CI + CE + CR)  # total number of synapses per neuron

startbuild = time.time()
# synaptic weights matrix: J_ij is the weight for a connection from i to j
J = np.array([[J_ex, l*J_ex, J_ex], [h*J_ex, l*h*J_ex, h*J_ex], [-g*l2*J_ex, -l*g*J_ex, -g*J_ex]])
# connectivity matrix: C_ij is the weight for a connection from i to j
C = np.array([[0.083, 0.083, 0.06], [0.083, 0.083, 0.06], [0.373, 0.373, 0.316]])
C[0,:] *= NE
C[1,:] *= NR
C[2,:] *= NI
# C[1,2] *= 1.1
# C[:,1] *= 0.75
C = np.array(C, dtype=int)

# Excitatory integrating neuron parameters
# The synaptic currents are normalized such that the amplitude of the PSP is J.
tauSyn = 0.5  # synaptic time constant in ms
tauMem = 10.0  # time constant of membrane potential in ms
CMem = 250.0  # capacitance of membrane in in pF
theta = 20.0  # membrane threshold potential in mV
neuron_params = {"C_m": CMem,
                 "tau_m": tauMem,
                 "tau_syn_ex": tauSyn,
                 "tau_syn_in": tauSyn,
                 "t_ref": 2.0,
                 "E_L": 0.0,
                 "V_reset": 0.0,
                 "V_m": 0.0,
                 "V_th": theta}



# Excitatory resonating neuron parameters
# mt.show_freqs_dep_t1g1(maxt1=120.0, mint1=1.5, ming1=1.5, maxg1=100.0, C=250., g=25., N=500, only_fitting=True)
# mt.show_freqs_dep_t1g1(maxt1=120.0, mint1=1.5, ming1=1.5, maxg1=100.0, C=250., g=25., N=500, only_fitting=True)
tau_1 = 100.0
C_m = 250.0
g_1 = 85.0
g_m = 15.0
tauSyn2 = 1.5  # synaptic time constant in ms
neuron_params2 = {
    "tau_1": tau_1,
    "C_m": C_m,
    "tau_syn_ex": tauSyn,
    "tau_syn_in": tauSyn,
    "g_rr": g_1,
    "g": g_m,
    "V_m": 0.0,
    "V_reset": 0.0,
    "E_L": 0.0,
    "V_th": 20.0
}

# Threshold rate & poisson generator
#nu_th = (theta * CMem) / (J_ex * CE * exp(1) * tauMem * tauSyn)
#nu_ex = eta * nu_th
#p_rate = p_rate_factor * nu_ex * CE
alpha = g * tau_1 / C_m  # regime determining constant 1
beta = g_1 * tau_1 / C_m  # regime determining constant 2
factor = (alpha - 1)**2 / 4
exp_f_r = 1. / (2. * np.pi * tau_1) * np.sqrt(np.sqrt((alpha + beta + 1.)*(alpha + beta + 1.) - (alpha + 1.)*(alpha + 1.) ) - 1.)

# Setup kernel
nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})

# ------------------------------------ CREATING DEVICES/NEURONS ---------------------------------
# Building network
print("Building network")
nest.SetDefaults("iaf_psc_exp", neuron_params)
nest.SetDefaults("gif2_psc_exp", neuron_params2)

if NR ==0: nodes_r = nest.Create("gif2_psc_exp", 1)
else: nodes_r = nest.Create("gif2_psc_exp", NR)
if NE ==0: nodes_ex = nest.Create("iaf_psc_exp", 1)
else: nodes_ex = nest.Create("iaf_psc_exp", NE)
nodes_in = nest.Create("iaf_psc_exp", NI)

noise = nest.Create("sinusoidal_poisson_generator", params=[{'rate': p_rate,
                                                            'amplitude': 0.0, # p_rate/15.,
                                                            'frequency': 10.0,
                                                            'phase': 0.0}])
# Recording device configuration

multimeter = nest.Create("multimeter", params={"interval": dt,
                                               "record_from": [ "V_m"],
                                               'start': recstart})
if flag_plot_voltage:
    multimeter2 = nest.Create("multimeter", params={"interval": dt,
                                               "record_from": [ "V_m", "w" ],
                                               'start': recstart})
spikedetector_ex = nest.Create("spike_detector", params={'start': recstart})
spikedetector_r = nest.Create("spike_detector", params={'start': recstart})
spikedetector_in = nest.Create("spike_detector", params={'start': recstart})
spikedetector_raster = nest.Create("spike_detector", params={'start': recstart})

# ------------------------------------ SYNAPSES ---------------------------------------
print("Defining Synapses")
nest.CopyModel("static_synapse", "excitatory", {"weight": J_ex, "delay": delay})
nest.CopyModel("static_synapse", "inhibitory", {"weight": J_in, "delay": delay})
# nest.CopyModel("static_synapse", "EE", {"weight": J[0,0], "delay": delay})
# nest.CopyModel("static_synapse", "ER", {"weight": J[0,1], "delay": delay})
# nest.CopyModel("static_synapse", "EI", {"weight": J[0,2], "delay": delay})
# nest.CopyModel("static_synapse", "RE", {"weight": J[1,0], "delay": delay})
# nest.CopyModel("static_synapse", "RR", {"weight": J[1,1], "delay": delay})
# nest.CopyModel("static_synapse", "RI", {"weight": J[1,2], "delay": delay})
# nest.CopyModel("static_synapse", "IE", {"weight": J[2,0], "delay": delay})
# nest.CopyModel("static_synapse", "IR", {"weight": J[2,1], "delay": delay})
# nest.CopyModel("static_synapse", "II", {"weight": J[2,2], "delay": delay})

# ------------------------------------ CONNECTING ---------------------------------------
print("Connecting devices")
nest.Connect(noise, nodes_ex, syn_spec="excitatory")
nest.Connect(noise, nodes_r, syn_spec="excitatory")
nest.Connect(noise, nodes_in, syn_spec="excitatory") #conn_spec={'rule': 'fixed_indegree', 'indegree': 1850},
nest.Connect(nodes_ex[:N_rec], spikedetector_ex, syn_spec="excitatory")
nest.Connect(nodes_r[:N_rec],  spikedetector_r, syn_spec="excitatory")
nest.Connect(nodes_in[:N_rec], spikedetector_in, syn_spec="excitatory")
nest.Connect(nodes_ex[:Nred[1]], spikedetector_raster, syn_spec="excitatory")
nest.Connect(nodes_r[:Nred[0]],  spikedetector_raster, syn_spec="excitatory")
nest.Connect(nodes_in[:Nred[2]], spikedetector_raster, syn_spec="excitatory")

if flag_plot_voltage:
    nest.Connect(multimeter, nodes_ex[ :N_rec ], syn_spec="excitatory")
    nest.Connect(multimeter, nodes_r[ :N_rec ], syn_spec="excitatory")
    nest.Connect(multimeter, nodes_in[ :N_rec ], syn_spec="inhibitory")
    nest.Connect(multimeter2, nodes_r)

#print("Connecting network")
#conn_params_in = {'rule': 'fixed_indegree', 'indegree': CI}
#conn_params_ex = {'rule': 'fixed_indegree', 'indegree': CE}

print("Excitatory connections")
# nest.Connect(nodes_ex, nodes_ex + nodes_in + nodes_r, conn_params_ex, "excitatory")
nest.Connect(nodes_ex, nodes_ex,
             conn_spec= {'rule': 'fixed_indegree', 'indegree': C[ 0, 0 ]},
             syn_spec={'weight': J[ 0, 0 ]})
nest.Connect(nodes_ex, nodes_r,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 1 ]},
             syn_spec={'weight': J[ 0, 1 ]})
nest.Connect(nodes_ex, nodes_in,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 0, 2 ]},
             syn_spec={'weight': J[ 0, 2 ]})

print("Resonating connections")
#nest.Connect(nodes_r, nodes_ex + nodes_in + nodes_r, conn_params_ex, "excitatory")
nest.Connect(nodes_r, nodes_ex,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 0 ]},
             syn_spec={'weight': J[ 1, 0 ]})
nest.Connect(nodes_r, nodes_r,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 1 ]},
             syn_spec={'weight': J[ 1, 1 ]})
nest.Connect(nodes_r, nodes_in,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 1, 2 ]},
             syn_spec={'weight': J[ 1, 2 ]})

print("Inhibitory connections")
#nest.Connect(nodes_in, nodes_r + nodes_ex + nodes_in, conn_params_in, "inhibitory")
nest.Connect(nodes_in, nodes_ex,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 2, 0 ]},
             syn_spec={'weight': J[ 2, 0 ]})
nest.Connect(nodes_in, nodes_r,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 2, 1 ]},
             syn_spec={'weight': J[ 2, 1 ]})
nest.Connect(nodes_in, nodes_in,
             conn_spec={'rule': 'fixed_indegree', 'indegree': C[ 2, 2 ]},
             syn_spec={'weight': J[ 2, 2 ]})

endbuild = time.time()

# ------------------------------------ SIMULATING ---------------------------------------
print("Simulating")
nest.Simulate(simtime)
endsimulate = time.time()
print("Simulation successful")

# ------------------------------------ EVALUATING ---------------------------------------
print("Evaluating recordings")
spikesenders_raster = nest.GetStatus(spikedetector_raster, "events")[0]['senders']
spikesenders_ex = nest.GetStatus(spikedetector_ex, "events")[0]['senders']
spikesenders_r = nest.GetStatus(spikedetector_r, "events")[0]['senders']
spikesenders_in = nest.GetStatus(spikedetector_in, "events")[0]['senders']
spiketimes_raster = nest.GetStatus(spikedetector_raster, "events")[0]['times']
spiketimes_ex = nest.GetStatus(spikedetector_ex, "events")[0]['times']
spiketimes_r = nest.GetStatus(spikedetector_r, "events")[0]['times']
spiketimes_in = nest.GetStatus(spikedetector_in, "events")[0]['times']
if flag_plot_voltage:
    events    = nest.GetStatus(multimeter)[0]["events"]
    t = events["times"];

#voltage = nest.GetStatus(multimeter2, "events")[0]
#woltage

rate_ex = nest.GetStatus(spikedetector_ex,'n_events')[0] / rectime * 1000.0 / N_rec
rate_in = nest.GetStatus(spikedetector_in,'n_events')[0] / rectime * 1000.0 / N_rec
rate_r = nest.GetStatus(spikedetector_r,'n_events')[0] / rectime * 1000.0 / N_rec

# Plot the voltage trace
if flag_plot_voltage:
    voltage = nest.GetStatus(multimeter2, 'events')[0]['V_m']
    woltage = nest.GetStatus(multimeter2, 'events')[0]['w']
    times = nest.GetStatus(multimeter2, 'events')[0]['times']
    senders = nest.GetStatus(multimeter2, 'events')[0]['senders']
    voltage = voltage [ senders ==1 ]
    woltage = woltage [ senders ==1 ]
    times = times [ senders ==1 ]
    fig1 = plt.figure("Traces")
    plt.clf()
    ax1 = fig1.add_subplot(211)
    ax1 = plt.plot(times, voltage)
    ax2 = fig1.add_subplot(212)
    ax2 = plt.plot(times, woltage)


# num_synapses = nest.GetDefaults("EE")["num_connections"] \
#                + nest.GetDefaults("ER")["num_connections"]\
#                + nest.GetDefaults("EI")["num_connections"]\
#                + nest.GetDefaults("RE")["num_connections"]\
#                + nest.GetDefaults("RR")["num_connections"]\
#                + nest.GetDefaults("RI")["num_connections"]\
#                + nest.GetDefaults("IE")["num_connections"]\
#                + nest.GetDefaults("IR")["num_connections"]\
#                + nest.GetDefaults("II")["num_connections"]

build_time = endbuild - startbuild
sim_time = endsimulate - endbuild


# setup for rasterplots
figu = plt.figure("Rasterplots")
plt.clf()
spikesenders_raster2 = np.zeros_like(spikesenders_raster)
spikesenders_raster2 = spikesenders_raster
spikesenders_raster2[ spikesenders_raster >= NE + NR] -= (NE + NR) - Nred[0] - Nred[1]
spikesenders_raster2[ spikesenders_raster >= NR] -= NR - Nred[0]
ax1 = show_Rasterplot(figu, spiketimes_raster, spikesenders_raster2,
                      recstart, simtime, rasterrows, 221, 0)
ax1.set_title("All populations (in fraction of appearance)")
ax2, x1, y1 = show_Rasterplot2(figu, spiketimes_ex, spikesenders_ex,
                                  recstart, simtime, rasterrows, 222, (NE-1))
ax2.set_title("Excitatory\ resonating neurons")
ax4, x2, y2 = show_Rasterplot2(figu, spiketimes_in, spikesenders_in,
                                  recstart, simtime, rasterrows, 224, (NE+NR-1))
ax4.set_title("Inhibitory neurons")
ax3, x3, y3 = show_Rasterplot2(figu, spiketimes_r, spikesenders_r,
                                  recstart, simtime, rasterrows, 223, 0)
ax3.set_title("Resonating neurons")
plt.ion()

print("Brunel network simulation (Python)")
#print("Number of neurons : {0}".format(N_neurons))
#print("Number of synapses: {0}".format(num_synapses))
print("Number of excitatory neurons: {0}".format(NE+NR))
print("Fraction of resonators among these: {0}".format(fraction_resonators_ex))
print("Number of inhibitory neurons: {0}".format(NI))
print "  alpha = ", alpha, "  , beta = ", beta,  " , beta >? ", factor
if beta > factor:
    print("    stable. expected subthreshold resonance frequency: {0:3} Hz".format(exp_f_r * 1000))
print('rates: E: {0}, R: {1}, I: {2}'.format(rate_ex, rate_r, rate_in))
print('This is a Brunel network with population sizes and connectivity from Potjans, reduced synaptic weights for R.')

#pylab.clf()
# TODO: 1.) power spectrum of each population and sum
# TODO: 2.) compute the overall gain and the gain of the resonating population
# TODO: 3.) Autocorrelation and inter-spike-interval
#plt.ion()
#plt.show()
