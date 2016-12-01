# # coding=utf-8
# """
# Does the gif2_neuron react to inhibitory input?
# """
# import sys
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from itertools import product
#
# np.set_printoptions(precision=4, suppress=True)
#
# import nest
# dt = 0.1  # the resolution in ms
# nest.SetKernelStatus(dict(resolution=dt, print_time=False,
# 						  overwrite_files=True))
# try:
# 	nest.Install("gif2_module")
# except:
# 	pass
# simtime = 5000.0
# recstart = 0.0
# rate1 = 50000.0
# rate2 = 20000.0
# V_dist = 9.0
#
# gif_params[ 'V_reset' ] = gif_params[ 'V_th' ] - V_dist
# gif_params[ 'E_L' ] = gif_params[ 'V_th' ] - V_dist
# lif_params = dict(V_th=-50.0, tau_m=10.0, E_L=-65.0, t_ref=2.0,
# 					   V_reset=-65.0, C_m=250.0,
# 					   tau_syn_ex=0.5, tau_syn_in=0.5)
# syndict = dict(synweight=87.8, J_ex=0.125, delay=1.5)
# stm_params = dict(rate=rate1)
# det_params = dict(withtime=True, withgid=False, to_file=False, start=recstart)
# inh_params = dict(rate=rate2)
#
# simtime = simtime
# recstart = det_params[ 'start' ]
#
# voltlif = nest.Create('voltmeter')
# voltgif = nest.Create('voltmeter')
# drive = nest.Create("poisson_generator", params=stm_params)
# inhib = nest.Create("poisson_generator", params=inh_params)
# lifspikes = nest.Create("spike_detector", params=det_params)
# gifspikes = nest.Create("spike_detector", params=det_params)
# gif = nest.Create("gif2_psc_exp")
# lif = nest.Create("iaf_psc_exp")
# nest.Connect(drive, gif, syn_spec=dict(
# 		model="static_synapse",
# 		weight=syndict[ 'J_ex' ] * syndict[ 'synweight' ],
# 		delay=syndict[ 'delay' ]))
# nest.Connect(drive, lif, syn_spec=dict(
# 		model="static_synapse",
# 		weight=syndict[ 'J_ex' ] * syndict[ 'synweight' ],
# 		delay=syndict[ 'delay' ]))
#
# nest.Connect(inhib, gif, syn_spec=dict(
# 		model="static_synapse",
# 		weight=-5.0 * syndict[ 'J_ex' ] * syndict[ 'synweight' ],
# 		delay=syndict[ 'delay' ]))
# nest.Connect(inhib, lif, syn_spec=dict(
# 		model="static_synapse",
# 		weight=-5.0 * syndict[ 'J_ex' ] * syndict[ 'synweight' ],
# 		delay=syndict[ 'delay' ]))
#
# nest.Connect(gif, gifspikes)
# nest.Connect(lif, lifspikes)
# nest.Connect(gif, voltgif)
# nest.Connect(lif, voltlif)
#
# nest.Simulate(simtime + recstart)
#


import nest
import numpy as np
import pylab as pl

try:
	nest.Install("gif2_module")
except:
	pass
nest.ResetKernel()

gif_params = dict(tau_1=100.0, C_m=250.0, tau_syn_ex=0.5,
				  tau_syn_in=0.5, g_rr=25.0, g=25.0,
				  V_m=0.0, V_th=15.0, t_ref=1.5, V_reset=9.0, E_L=0.0)

# create neuron and multimeter
n = nest.Create('gif2_psc_exp', params=gif_params)

m = nest.Create('multimeter', params={'withtime':    True, 'interval': 0.1,
									  'record_from': [ 'V_m', 'w']})

# Create spike generators and connect
gex = nest.Create('poisson_generator',
				  params={'rate': 15 * 60000.0})
gin = nest.Create('poisson_generator',
				  params={'rate': 60000.0})

nest.Connect(gex, n, syn_spec={'weight': 0.0125 * 87.8})  # excitatory
nest.Connect(gin, n, syn_spec={'weight': - 0.0125 * 87.8})  # inhibitory
nest.Connect(m, n)

# simulate
nest.Simulate(100)

# obtain and display data
events = nest.GetStatus(m)[ 0 ][ 'events' ]
t = events[ 'times' ];

pl.subplot(211)
pl.plot(t, events[ 'V_m' ])
# pl.axis([ 0, 100, -75, -53 ])
pl.xlim = ([0., 100.])
pl.ylim = ([-80.0, 20.0])
pl.ylabel('Membrane potential [mV]')

pl.subplot(212)
pl.plot(t, events[ 'w' ])
# pl.axis([ 0, 100, 0, 45 ])
pl.xlim = ([ 0., 100. ])
pl.ylim = ([ -30.0, 10.0 ])
pl.xlabel('Time [ms]')
pl.ylabel('Second variable [nS]')

pl.ion()