import siegert
import nest
import nest.voltage_trace
import numpy as np
import matplotlib.pyplot as plt


# parameters
T = 100000.0
t_start = 1000.0
N = 100
taum = 10.0
tau_syn = 1.0
taur = 0.1
V_th = -50.0
V0 = -70.0
nu = 600.0
C = 250.0
J = 600.0
# scale connection strength by tau_syn such that results for different synaptic time constants are comparable
PSC = J/tau_syn

neurondict = {'V_th':V_th, 'tau_m':10.0, 'E_L':V0, 't_ref':taur, 'V_reset': V0,'C_m':C,'tau_syn_ex':tau_syn}
nest.SetDefaults('iaf_psc_exp', neurondict)
neuron = nest.Create("iaf_psc_exp",N)

# connect poisson input
syn_dict = {'weight': PSC, 'delay': 0.1}
noise = nest.Create("poisson_generator")
nest.SetStatus(noise, [{"rate": nu}])
nest.Connect(noise, neuron, 'all_to_all', syn_spec=syn_dict)

# connect spike detectors
nest.SetDefaults('spike_detector',{'start':t_start})
spikedetector = nest.Create("spike_detector")
nest.Connect(neuron, spikedetector, 'all_to_all')

nest.Simulate(T)

# get firing rate of neurons
rate = nest.GetStatus(spikedetector, "n_events")[0]
print 'frequency neuron', 1000*rate/(T-t_start)*1/float(N)

PSC *= 1e-3
tau_syn *= 1e-3
taum *= 1e-3
taur *= 1e-3
C *= 1e-6
mu = V0 + PSC*tau_syn/C*taum*nu
sigma = PSC*tau_syn/C*np.sqrt(taum*nu)

# standard siegert
rate_th = siegert.nu_0(taum, taur, V_th, V0, mu, sigma)
print 'rate siegert', rate_th
# siegert for filtered synapses
rate_th = siegert.nu0_fb(taum, tau_syn, taur, V_th, V0, mu, sigma)
print 'rate siegert filtered synapse', rate_th
rate_th = siegert.nu0_fb433(taum, tau_syn, taur, V_th, V0, mu, sigma)
print 'rate siegert filtered synapse', rate_th


'''
Results:

tau_syn = 0.001
frequency neuron 17.1885858586
rate siegert 17.9756406223
rate siegert filtered synapse 17.7219741779
rate siegert filtered synapse 17.7213200161

tau_syn = 0.1
frequency neuron 15.9861616162
rate siegert 17.9756406223
rate siegert filtered synapse 15.5011012078
rate siegert filtered synapse 15.4324345605

tau_syn = 1.0
frequency neuron 11.3628282828
rate siegert 17.9756406223
rate siegert filtered synapse 10.6977601049
rate siegert filtered synapse 9.93331690802

'''

