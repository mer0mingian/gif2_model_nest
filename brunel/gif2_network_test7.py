# -*- coding: utf-8 -*-
#
# brunel_alpha_nest.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU GeNRal Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU GeNRal Public License for more details.
#
# You should have received a copy of the GNU GeNRal Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

# setup environment
import time
import matplotlib.pyplot as plt
import nest
import nest.raster_plot
import nest.voltage_trace
from numpy import exp
import numpy as np
import pylab

nest.ResetKernel()

try:
	nest.Install("gif2_module")
except:
	print ("The GIF2 model had already been installed")
nest.set_verbosity('M_WARNING')

'''
Definition of functions used in this example. First, define the
Lambert W function implemented in SLI. The second function
computes the maximum of the postsynaptic potential for a synaptic
input current of unit amplitude (1 pA) using the Lambert W
function. Thus function will later be used to calibrate the synaptic
weights.
'''


def LambertWm1(x):
	nest.sli_push(x);
	nest.sli_run('LambertWm1');
	y = nest.sli_pop()
	return y


def ComputePSPnorm(tauMem, CMem, tauSyn):
	a = (tauMem / tauSyn)
	b = (1.0 / tauSyn - 1.0 / tauMem)

	# time of maximum
	t_max = 1.0 / b * (-LambertWm1(-exp(-1.0 / a) / a) - 1.0 / a)

	# maximum of PSP for current of unit amplitude
	return exp(1.0) / (tauSyn * CMem * b) * (
		(exp(-t_max / tauMem) - exp(-t_max / tauSyn)) / b - t_max * exp(-t_max / tauSyn))


nest.ResetKernel()
startbuild = time.time()

dt = 1.0  # the resolution in ms
simtime = 750.0  # Simulation time in ms
delay = 1.5  # synaptic delay in ms
p_rate_factor =200.0 # modifies poisson input rate

# neuron counts
ex_times_in_nodes = 4.0
fraction_resonators_ex = 0.5
# fraction_resonators_in = 0.0
order = 500
NE = int(round(order * ex_times_in_nodes * (1 - fraction_resonators_ex)))  # number of excitatory neurons
NR = int(round(order * ex_times_in_nodes * fraction_resonators_ex))  # number of resonating neurons
NI = order  # number of inhibitory neurons
N_neurons = NE + NI + NR  # number of neurons in total
N_rec = 50  # record from 50 neurons

# Synapse & Connectivity parameters
g = 5.0  # ratio inhibitory weight/excitatory weight
eta = 2.0  # external rate relative to threshold rate
epsilon = 0.1  # connection probability
# CE should be equal for resonators and integrators for now
CE = int(epsilon * (NE+NR))  # number of excitatory synapses per neuron
CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
C_tot = int(CI + CE)  # total number of synapses per neuron

# Excitatory integrating neuron parameters
# The synaptic currents are normalized such that the amplitude of the PSP is J.
tauSyn = 0.5  # synaptic time constant in ms
tauMem = 20.0  # time constant of membrane potential in ms
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
J = 0.1  # postsynaptic amplitude in mV
J_unit = ComputePSPnorm(tauMem, CMem, tauSyn)
J_ex = J / J_unit  # amplitude of excitatory postsynaptic current
J_in = -g * J_ex  # amplitude of inhibitory postsynaptic current


# Excitatory resonating neuron parameters
tau_1 = 100.0 / np.sqrt(2)
C_m = 250.0
g_1 = 25.0 / np.sqrt(2)
g_m = 25.0 / np.sqrt(2)
alpha = g_m * tau_1 / C_m
beta = g_1 * tau_1 / C_m
factor = (alpha - 1)**2 / 4
neuron_params2 = {
    "tau_1": tau_1,
    "C_m": C_m,
    "g_rr": g_1,
    "g": g_m,
    "V_m": 0.0,
    "V_reset": 14.0,
    "E_L": 0.0,
    "V_th": 20.0
}

# Threshold rate & poisson generator
nu_th = (theta * CMem) / (J_ex * CE * exp(1) * tauMem * tauSyn)
nu_ex = eta * nu_th
p_rate = p_rate_factor * nu_ex * CE

# Setup kernel
nest.SetKernelStatus({"resolution": dt, "print_time": True, "overwrite_files": True})

# Building network
print("Building network")
nest.SetDefaults("iaf_psc_exp", neuron_params)
nest.SetDefaults("gif2_psc_exp", neuron_params2)

if NR ==0: nodes_r = nest.Create("gif2_psc_exp", 1)
else: nodes_r = nest.Create("gif2_psc_exp", NR)
if NE ==0: nodes_ex = nest.Create("iaf_psc_alpha", 1)
else: nodes_ex = nest.Create("iaf_psc_alpha", NE)
nodes_in = nest.Create("iaf_psc_alpha", NI)

noise = nest.Create("sinusoidal_poisson_generator", params=[{'rate': p_rate,
                                                            'amplitude': p_rate/10.,
                                                            'frequency': 7.0,
                                                            'phase': 0.0}])
# Recording device configuration

multimeter = nest.Create("multimeter", params={"interval": dt, "record_from": [ "V_m"], "withgid": True})
nest.SetStatus(spikes, [{"withtime": True, "withgid": True}])

print("Connecting devices")
nest.CopyModel("static_synapse", "excitatory", {"weight": J_ex, "delay": delay})
nest.CopyModel("static_synapse", "inhibitory", {"weight": J_in, "delay": delay})

nest.Connect(noise, nodes_ex, syn_spec="excitatory")
nest.Connect(noise, nodes_r, syn_spec="excitatory")
nest.Connect(noise, nodes_in, syn_spec="excitatory")

nest.Connect(nodes_ex[:N_rec], spikes, syn_spec="excitatory")
nest.Connect(nodes_r[:N_rec],  spikes, syn_spec="excitatory")
nest.Connect(nodes_in[:N_rec], spikes, syn_spec="excitatory")

nest.Connect(multimeter, nodes_ex[:N_rec], syn_spec="excitatory")
nest.Connect(multimeter, nodes_r[:N_rec], syn_spec="excitatory")
nest.Connect(multimeter, nodes_in[:N_rec], syn_spec="inhibitory")


print("Connecting network")
conn_params_in = {'rule': 'fixed_indegree', 'indegree': CI}
conn_params_ex = {'rule': 'fixed_indegree', 'indegree': CE}

print("Excitatory connections")
nest.Connect(nodes_ex, nodes_ex + nodes_in + nodes_r, conn_params_ex, "excitatory")

print("Resonating connections")
nest.Connect(nodes_r, nodes_ex + nodes_in + nodes_r, conn_params_ex, "excitatory")

print("Inhibitory connections")
nest.Connect(nodes_in, nodes_r + nodes_in + nodes_ex, conn_params_in, "inhibitory")

endbuild = time.time()
print("Simulating")
nest.Simulate(simtime)
endsimulate = time.time()

events_ex = nest.GetStatus(espikes, "n_events")[0]
events_in = nest.GetStatus(ispikes, "n_events")[0]
events_r  = nest.GetStatus(rspikes, "n_events")[0]
events    = nest.GetStatus(multimeter)[0]["events"]
events_n  = nest.GetStatus(sinus, "n_events")[0]
t = events["times"];

rate_ex = events_ex / simtime * 1000.0 / N_rec
rate_in = events_in / simtime * 1000.0 / N_rec
rate_r  = events_r  / simtime * 1000.0 / N_rec
rate_n  = events_n  / simtime * 1000.0 / N_rec

num_synapses = nest.GetDefaults("excitatory")["num_connections"] + nest.GetDefaults("inhibitory")["num_connections"]

build_time = endbuild - startbuild
sim_time = endsimulate - endbuild

print("Brunel network simulation (Python)")
print("Number of neurons : {0}".format(N_neurons))
print("Number of synapses: {0}".format(num_synapses))
print("       Exitatory  : {0}".format(int(CE * N_neurons) + N_neurons))
print("       Inhibitory : {0}".format(int(CI * N_neurons)))
print("Excitatory rate   : %.2f Hz" % rate_ex)
print("Inhibitory rate   : %.2f Hz" % rate_in)
print("Resonating rate   : %.2f Hz" % rate_r)
print("Number of excitatory neurons: {0}".format(NE+NR))
print("Fraction of resonators among these: {0}".format(fraction_resonators_ex))
print("Number of inhibitory neurons: {0}".format(NI))
print "  alpha = ", alpha, "  , beta = ", beta,  " , beta >? ", factor


#pylab.clf()
# TODO: what do i really want to see?
# TODO: 1.) a raster plot and histogram for each population and the sum
# TODO: 2.) average firing rates of each population and sum
# TODO: 3.) power spectrum of each population and sum
# TODO: 4.) compute the overall gain and the gain of the resonating population
# TODO: 5.) Autocorrelation and inter-spike-interval
#plt.ion()
#plt.show()
