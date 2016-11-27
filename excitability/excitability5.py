"""
This is a default script that should be adapted to the respective purpose.
"""

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import nest
import elephant.statistics as es
from itertools import product

np.set_printoptions(precision=4, suppress=True)


class firing_rate_response:
	def __init__(self, recstart=2500.0, simtime=2000.0):
		V_dist = 6.0
		self.gif_params = dict(tau_1=100.0, C_m=250.0, tau_syn_ex=0.5,
		                       tau_syn_in=0.5, g_rr=25.0, g=25.0,
		                       V_m=500.0, V_th=-50, t_ref=1.5, V_reset=-65.0)
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
		self.simtime = simtime
		self.recstart = self.det_params[ 'start' ]
		self.indexarray = None
		self.resultarray = None
		self.num_of_trials = 0
		self.raw_data = {0: gif_params}

	def network_setup(self):
		nest.set_verbosity('M_ERROR')
		nest.ResetKernel()
		self.dt = 0.1  # the resolution in ms
		nest.SetKernelStatus(dict(resolution=self.dt, print_time=False,
		                          overwrite_files=True))
		try:
			nest.Install("gif2_module")
		except:
			pass
		self.drive = nest.Create("sinusoidal_poisson_generator",
		                         params=stm_params)
		self.lifspikes = nest.Create("spike_detector", params=self.det_params)
		self.gifspikes = nest.Create("spike_detector", params=self.det_params)
		self.gif = nest.Create("gif2_psc_exp", n=100)
		self.lif = nest.Create("iaf_psc_exp", n=100)
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
						  lif_params=None, stm_params=None):
		nest.ResetNetwork()
		if lif_params is not None:
			nest.SetStatus(self.lif, lif_params)
		self.gif_params[ 'V_reset' ] = self.gif_params[ 'V_th' ] - V_dist
		self.gif_params[ 'E_L' ] = self.gif_params[ 'V_th' ] - V_dist
		if stm_params is not None:
			nest.SetStatus(self.drive, stm_params)
		nest.SetStatus(self.gif, gif_params)

	def network_simulate(self, simtime=None):
		for i in np.arange(50000.0, 75000.0, 500.0):
			nest.ResetNetwork()
			self.stm_params[ 'rate' ] = i
			if simtime is not None:
				nest.Simulate(self.recstart + simtime)
			else:
				nest.Simulate(self.recstart + self.simtime)
		self.num_of_trials += 1

	def process_results(self):
		secondary_index = self.num_of_trials
		self.rate_lif = nest.GetStatus(self.lifspikes,
		                               "n_events")[ 0 ] / self.simtime * 10.
		self.rate_gif = nest.GetStatus(self.gifspikes,
		                               "n_events")[ 0 ] / self.simtime * 10.
		self.cv_gif = es.cv(es.isi(nest.GetStatus(self.gifspikes,
		                                          "events")[ 0 ][ "times" ]))
		self.cv_lif = es.cv(es.isi(nest.GetStatus(self.lifspikes,
		                                          "events")[ 0 ][ "times" ]))
		self.indexarray = np.array([ self.stm_params[ 'rate' ],
		                             self.rate_gif,
		                             self.rate_lif,
		                             self.cv_gif,
		                             self.cv_lif ])
		if self.resultarray is None:
			self.resultarray = self.indexarray
		else:
			self.resultarray = np.hstack((self.resultarray, self.indexarray))
		self.raw_data[ self.num_of_trials ] = self.gif_params

	# def save_results

	def show_results(self, targetfilename='IF_curves.png'):
		colorlist = [ 'r', 'g', 'c', 'k', 'y', 'm' ]
		linestyles = [ '-', '--', '-.' ]
		presentations = product(colorlist, linestyles)

		fig1 = plt.figure()
		fig1.suptitle('IF_curves')
		ax1 = fig1.add_subplot(1, 1, 1)

		for index in np.arange(0, min([ len(colorlist),
									    self.num_of_trials - 1 ])):
			c = colorlist[ index ]
			if index == 0:
				drive = self.indexarray.shape[ :, 0 ]
				lif_rate = self.indexarray.shape[ :, 2 ]
				ax1.plot(drive, lif_rate, color='b', label='iaf')
			else:
				gif_rate = self.indexarray.shape[ :, 1 ]
				ax.plot(drive, gif_rate, color=c)
				ax.add_label = ': C={0}, g={1}, g1={2}, t1={3}, dV={4}'.format(
						self.raw_data[ index ][ C_m ],
						self.raw_data[ index ][ g ],
						self.raw_data[ index ][ g_rr ],
						self.raw_data[ index ][ tau_1 ],
						self.raw_data[ index ][ V_th ] -
						self.raw_data[ index ][ V_reset ],)
		plt.legend(loc='upper left')
		plt.ion()
		plt.show()

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
	V_reset = 9.0
	V_dist = 5.395
	tauSyn = 0.5
	lif_params = {
		"C_m":        250.0,
		"tau_m":      10.0,
		"tau_syn_ex": tauSyn,
		"tau_syn_in": tauSyn,
		"t_ref":      1.5,
		"E_L":        0.0,
		"V_reset":    0.0,
		"V_m":        0.0,
		"V_th":       15.0}
	gif_params = {
		 "tau_1":      85.0,
		 "C_m":        200.0,
		 "tau_syn_ex": tauSyn,
		 "tau_syn_in": tauSyn,
		 "g_rr":       40.0,
		 "g":          40.0,
		 "V_m":        0.0,
		 "V_reset":    V_reset,
		 "E_L":        0.0,
		 "V_th":       V_reset + V_dist,
		 "t_ref":		1.5}

	simparamdict = {'recstart': 1500.0, 'simtime': 2500.0}
	stm_params = {'rate': 65500.0}

	usecase = 'IF_curves_1'
	if usecase == 'IF_curves_1':
		print("We're checking some IF-curves.")

		a = firing_rate_response(recstart=1500.0, simtime=2000.0)
		a.network_setup()
			a.network_configure(gif_params=gif_params,
								V_dist=6.0,
								lif_params=lif_params,
								stm_params=stm_params)
			a.network_simulate()
		a.process_results()
