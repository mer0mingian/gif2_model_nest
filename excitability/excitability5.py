"""
This is a default script that should be adapted to the respective purpose.
It is intended to compare the spike response rate of two neuron models for a
list of parameter sets and and an array of driving poisson rates.
v1.0: stable. still need saving procedure.
"""

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import nest
# import elephant.statistics as es
from itertools import product

np.set_printoptions(precision=4, suppress=True)

class firing_rate_response:
	"""
	a = firing_rate_response(simparamdict=simparamdict,
							 diff_configs=diff_configs,
							 V_resets=V_resets, V_dists=V_dists,
							 cluster=False)
	a.network_setup()
	for trial in np.arange(0, len(diff_configs)):
		a.network_configure(trial)
		a.network_simulate()
	"""
	def __init__(self, simparamdict, diff_configs, V_resets, V_dists):
		self.gif_params = dict(tau_1=100.0, C_m=250.0, tau_syn_ex=0.5,
		                       tau_syn_in=0.5, g_rr=25.0, g=25.0,
		                       V_m=V_resets[ 0 ], t_ref=1.5,
							   V_reset=V_resets[ 0 ],
							   V_th=V_resets[ 0 ] + V_dists[ 0 ],
							   E_L=V_resets[ 0 ] + V_dists[ 0 ]
							   )
		self.lif_params = dict(V_th=-50.0, tau_m=10.0, E_L=-65.0, t_ref=1.5,
		                       V_reset=-65.0, C_m=250.0, V_m=-65.0,
		                       tau_syn_ex=0.5, tau_syn_in=0.5)
		self.syndict = dict(synweight=87.8, J_ex=0.125, delay=1.5)

		self.det_params = dict(withtime=True, withgid=False, to_file=False,
		                       start=simparamdict[ 'recstart' ])
		self.stm_params = dict(rate=0.0, amplitude=0.025 * 0.0,
							   frequency=10.0, phase=0.0)

		self.cluster = simparamdict[ 'cluster' ]
		self.simtime = simparamdict[ 'simtime' ]
		self.recstart = simparamdict[ 'recstart' ]
		self.simarray = np.arange(simparamdict[ 'mindrive' ],
									simparamdict[ 'maxdrive' ],
									simparamdict[ 'stepwidth' ])
		self.num_of_trial = 0
		self.create_data_structure(diff_configs, V_resets, V_dists)

	def create_data_structure(self, configs, V_resets, V_dists):
		"""
		put all the relevant data into the raw_data dictionary.
		structure:
		raw_data > trial_num > lif > config   > ...
								   > datadict > drives
											  > rates
							 > gif > config   > ...
								   > datadict > drives
											  > rates
								   > V_dist
								   > V_reset
		"""
		self.data = dict()
		for i in np.arange(len(configs)):
			self.data[ i ] = dict(lif=dict(),
								  gif=dict())
			self.data[ i ][ 'lif' ][ 'params' ] = self.lif_params
			self.data[ i ][ 'lif' ][ 'datadict' ] = dict(
				drives=self.simarray,
				rates=np.zeros_like(self.simarray))
			self.data[ i ][ 'gif' ][ 'params' ] = self.gif_params
			self.data[ i ][ 'gif' ][ 'datadict' ] = dict(
				drives=self.simarray,
				rates=np.zeros_like(self.simarray))
			self.data[ i ][ 'gif' ][ 'params' ][ 'V_reset' ] = V_resets[ i ]
			self.data[ i ][ 'gif' ][ 'params' ][ 'E_L' ] = V_resets[ i ]
			self.data[ i ][ 'gif' ][ 'params' ][ 'V_th' ] = V_resets[ i ] + \
															V_dists[ i ]
			for key, value in configs[ i ].iteritems():
				self.data[ i ][ 'gif' ][ 'params' ][ key ] = value

	def network_setup(self):
		nest.set_verbosity('M_ERROR')
		nest.ResetKernel()
		dt = 0.1  # the resolution in ms
		nest.SetKernelStatus(dict(resolution=dt, print_time=False,
		                          overwrite_files=True))
		try:
			nest.Install("gif2_module")
		except:
			pass
		self.drive = nest.Create("sinusoidal_poisson_generator",
		                         params=self.stm_params)
		self.gifspikes = nest.Create("spike_detector", params=self.det_params)
		self.gif = nest.Create("gif2_psc_exp", n=100)
		nest.Connect(self.drive, self.gif, syn_spec=dict(
				model="static_synapse",
				weight=self.syndict[ 'J_ex' ] * self.syndict[ 'synweight' ],
				delay=self.syndict[ 'delay' ]))
		nest.Connect(self.gif, self.gifspikes)

	def network_configure(self, trial):
		"""
		load the configuration for a specific trial
		"""
		nest.ResetNetwork()
		gif_params = self.data[ trial ][ 'gif' ][ 'params' ]
		nest.SetStatus(self.gif, gif_params)
		print('trial: {0} of {1}'.format(trial + 1, len(self.data)))
		if trial + 1 == len(self.data):
			self.lifspikes = nest.Create("spike_detector", params=self.det_params)
			self.lif = nest.Create("iaf_psc_exp", n=100)
			lif_params = self.data[ trial ][ 'lif' ][ 'params' ]
			nest.SetStatus(self.lif, lif_params)
			nest.Connect(self.drive, self.lif, syn_spec=dict(
					model="static_synapse",
					weight=self.syndict[ 'J_ex' ] * self.syndict[
						'synweight' ],
					delay=self.syndict[ 'delay' ]))
			nest.Connect(self.lif, self.lifspikes)



	def network_simulate(self, trial, simtime=None):
		"""
		simulate
		"""
		for drive_index, drive in enumerate(self.simarray):
			nest.ResetNetwork()
			self.stm_params[ 'rate' ] = drive
			nest.SetStatus(self.drive, self.stm_params)
			if simtime is not None:
				nest.Simulate(self.recstart + simtime)
			else:
				nest.Simulate(self.recstart + self.simtime)
			self.process_results(trial, drive_index, drive)

		self.num_of_trial += 1

	def process_results(self, trial, drive_index, drive):
		"""
		save obtained data into the data dictionary
		"""
		rate_gif = nest.GetStatus(self.gifspikes, "n_events")[ 0 ] / self.simtime * 10.
		self.data[ trial ][ 'gif' ][ 'datadict' ][ 'drives' ][ drive_index ] = drive
		self.data[ trial ][ 'gif' ][ 'datadict' ][ 'rates' ][ drive_index ] = rate_gif

		if trial + 1 == len(self.data):
			rate_lif = nest.GetStatus(self.lifspikes, "n_events")[ 0 ] / self.simtime * 10.
			self.data[ trial ][ 'lif' ][ 'datadict' ][ 'drives' ][ drive_index ] = drive
			self.data[ trial ][ 'lif' ][ 'datadict' ][ 'rates' ][ drive_index ] = rate_lif

	# def save_results

	def show_results(self, targetfilename='IF_curves.png'):
		"""
		old version. needs complete overhaul
		"""
		colorlist = [ 'r', 'g', 'c', 'k', 'y', 'm' ]
		linestyles = [ '-', '--', '-.' ]
		presentations = product(colorlist, linestyles)

		fig1 = plt.figure()
		fig1.suptitle('IF_curves')
		ax1 = fig1.add_subplot(1, 1, 1)

		for index in np.arange(0, min([ len(colorlist),
									    self.num_of_trial - 1 ])):
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
		plt.savefig(targetfilename)

	def compute_str_freq(self, verbose=False):
		"""
		currently not yet used
		"""
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

	lif_params = {"C_m": 250.0, "tau_m": 10.0,
				  "tau_syn_ex": tauSyn, "tau_syn_in": tauSyn, "t_ref": 1.5,
				  "E_L": 0.0, "V_reset": 0.0, "V_m": 0.0, "V_th": 15.0}
	gif_params = {"tau_1": 85.0, "C_m": 200.0,
				  "tau_syn_ex": tauSyn, "tau_syn_in": tauSyn,
				  "g_rr": 40.0, "g": 40.0, "V_m": V_reset, "V_reset": V_reset,
				  "E_L": 0.0, "V_th": V_reset + V_dist, "t_ref": 1.5}

	simparamdict = {'recstart': 1500.0, 'simtime': 2500.0,
					'mindrive': 50000.0, 'maxdrive': 75000.0,
					'stepwidth': 1000.0,
					'cluster': False}

	diff_configs = [{'C_m': 400.0, 'tau_1': 100.0, 'g': 40.0, 'g_rr': 40.0},
					{'C_m': 500.0, 'tau_1': 100.0, 'g': 25.0, 'g_rr': 25.0},
					{'C_m': 200.0, 'tau_1':  90.0, 'g': 25.0, 'g_rr': 71.0},
					{'C_m': 200.0, 'tau_1': 100.0, 'g': 25.0, 'g_rr': 58.0},
					{'C_m': 200.0, 'tau_1': 100.0, 'g': 25.0, 'g_rr': 58.0},
					{'C_m': 200.0, 'tau_1': 100.0, 'g': 25.0, 'g_rr': 58.0}]
	V_resets = np.ones(len(diff_configs)) * 9.0
	V_dists = np.array([ 5.4, 5.4, 5.4, 25.0, 5.615, 5.6, 5.625 ])

	usecase = 'IF_curves_1'
	if usecase == 'IF_curves_1':
		print("We're checking some IF-curves.")
		a = firing_rate_response(simparamdict=simparamdict,
								 diff_configs=diff_configs,
								 V_resets=V_resets, V_dists=V_dists)
		a.network_setup()
		for trial in np.arange(0, len(diff_configs)):
			a.network_configure(trial)
			a.network_simulate(trial)
