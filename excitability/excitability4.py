"""
This is a default script that should be adapted to the respective purpose.
"""

import sys
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import nest
import elephant.statistics as es
from itertools import product

np.set_printoptions(precision=4, suppress=True)


class firing_rate_response:
	def __init__(self, recstart=2500.0):
		V_dist = 6.0
		self.gif_params = dict(tau_1=100.0, C_m=250.0, tau_syn_ex=0.5,
		                       tau_syn_in=0.5, g_rr=25.0, g=25.0,
		                       V_m=500.0, V_th=-50)
		self.gif_params[ 'V_reset' ] = self.gif_params[ 'V_th' ] - V_dist
		self.gif_params[ 'E_L' ] = self.gif_params[ 'V_th' ] - V_dist
		self.lif_params = dict(V_th=-50.0, tau_m=10.0, E_L=-65.0, t_ref=2.0,
		                       V_reset=-65.0, C_m=250.0,
		                       tau_syn_ex=0.5, tau_syn_in=0.5)
		self.syndict = dict(synweight=87.8, J_ex=0.125, delay=1.5)
		self.stm_params = dict(rate=0.0, amplitude=0.025 * 0.0,
		                       frequency=10.0, phase=0.0)
		self.det_params = dict(withtime=True, withgid=False, to_file=False,
		                       start=recstart)

	def network_setup(self):
		nest.set_verbosity('M_ERROR')
		nest.ResetKernel()
		self.dt = 0.1  # the resolution in ms
		self.recstart = self.simparamdict[ 'recstart' ]
		self.simtime = self.simparamdict[ 'simtime' ]
		nest.SetKernelStatus(dict(resolution=self.dt, print_time=False,
		                          overwrite_files=True))
		try:
			nest.Install("gif2_module")
		except:
			pass
		self.drive = nest.Create("sinusoidal_poisson_generator",
		                         params=stm_params)
		self.lifspikes = nest.Create("spike_detector", params=det_params)
		self.gifspikes = nest.Create("spike_detector", params=det_params)
		self.gif = nest.Create("gif2_psc_exp")
		self.lif = nest.Create("lif_psc_exp")
		nest.Connect(self.drive, self.gif, syn_spec=dict(
				model="static_synapse",
				weight=self.syndict[ 'J_ex' ] * self.syndict[ 'synweight' ],
				delay=self.syndict[ 'delay' ]))
		nest.Connect(self.drive, self.lif, syn_spec=dict(
				model="static_synapse",
				weight=self.syndict[ 'J_ex' ] * self.syndict[ 'synweight' ],
				delay=self.syndict[ 'delay' ]))
		nest.Connect(self.gif, self.gifspikes)
		nest.Connect(self.lif, self.lifspikes)


	def network_configure(self, gif_params, V_dist=6.0 ,
	                      lif_params=self.lifparams,
	                      stm_params=self.stm_params):
		nest.ResetNetwork()
		self.gif_stimdict = gifdict
		self.gif_params[ 'V_reset' ] = self.gif_params[ 'V_th' ] - V_dist
		self.gif_params[ 'E_L' ] = self.gif_params[ 'V_th' ] - V_dist
		nest.SetStatus(self.drive, stm_params)
		nest.SetStatus(self.gif, gif_params)
		nest.SetStatus(self.lif, lif_params)


	def network_simulate(self):
		nest.Simulate(self.recstart + self.simtime)


	def process_results(self):
		self.rate_lif = nest.GetStatus(self.lifspikes,
		                               "n_events")[ 0 ] / simtime * 1000.
		self.rate_gif = nest.GetStatus(self.gifspikes,
		                               "n_events")[ 0 ] / simtime * 1000.
		self.cv_gif = es.cv(es.isi(nest.GetStatus(self.gifspikes,
		                                          "events")[ 0 ][ "times" ]))
		self.cv_lif = es.cv(es.isi(nest.GetStatus(self.lifspikes,
		                                          "events")[ 0 ][ "times" ]))
		self.indexarray = np.array([ p,
		                             self.rate_gif,
		                             self.rate_lif,
		                             self.cv_gif,
		                             self.cv_lif ])
		self.resultarray = np.vstack((self.resultarray, self.indexarray))

	# def save_results

	# def show_results


	def compute_str_freq(self, verbose=False):
		g = self.gif_params[ 'g' ]
		g1 = self.gif_params[ 'g_rr' ]
		t1 = self.gif_params[ 'tau_1' ]
		C = self.gif_params[ 'C_m' ]
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
		return fR


if __name__ == '__main__':
	gif_params = {"tau_1":      100.0,
				  "C_m":        500.0,  # C_m2,
				  "tau_syn_ex": 0.5,
				  "tau_syn_in": 0.5,
				  "g_rr":       25.0,  # g_1,
				  "g":          25.0,  # g_m,
				  "V_m":        0.0,
				  "V_reset":    3.0,
				  "E_L":        0.0,
				  "V_th":       15.0}

	recstart = 1500.0
	simtime = 5000.0

# TODO: implement different parameter settings. loop over them.