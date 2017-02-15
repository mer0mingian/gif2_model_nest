import numpy
import nest
import shelve

model = 'aeif_cond_exp'
params = {'a': 4.0,
          'b': 80.8,
          'V_th': -50.4,
          'Delta_T': 2.0,
          'I_e': 0.0,
          'C_m': 281.0,
          'g_L': 30.0,
          'V_reset': -70.6,
          'tau_w': 144.0,
          't_ref': 5.0,
          'V_peak': -40.0,
          'E_L': -70.6,
          'E_ex': 0.,
          'E_in': -70.}


class IF_curve():
    t_inter_trial = 200.  # Interval between two successive measurement trials
    t_sim = 1000.         # Duration of a measurement trial
    n_neurons = 100       # Number of neurons
    n_threads = 4         # Nubmer of threads to run the simulation

    def __init__(self, model, params=False):
        self.model = model
        self.params = params
        self.build()
        self.connect()

    def build(self):
        nest.ResetKernel()
        nest.SetKernelStatus({'local_num_threads': self.n_threads})

		if self.params:
            nest.SetDefaults(self.model, self.params)
        self.neuron = nest.Create(self.model, self.n_neurons)
        self.noise = nest.Create('noise_generator')
        self.spike_detector = nest.Create('spike_detector')

    def connect(self):
    	nest.Connect(self.noise, self.neuron, 'all_to_all')
        nest.Connect(self.neuron, self.spike_detector, 'all_to_all')

    def output_rate(self, mean, std):
        self.build()
        self.connect()
        nest.SetStatus(self.noise, [{'mean': mean, 'std': std, 'start': 0.0,
                                     'stop': 1000., 'origin': 0.}])
        nest.Simulate(self.t_sim)
        rate = nest.GetStatus(self.spike_detector, 'n_events')[0] * 1000.0 \
            / (1. * self.n_neurons * self.t_sim)
        return rate

    def compute_transfer(self, i_mean=(400.0, 900.0, 50.0),
                         i_std=(0.0, 600.0, 50.0)):        
        self.i_range = numpy.arange(*i_mean)
        self.std_range = numpy.arange(*i_std)
        self.rate = numpy.zeros((self.i_range.size, self.std_range.size))
        nest.set_verbosity('M_WARNING')
        for n, i in enumerate(self.i_range):
            print('I  =  {0}'.format(i))
            for m, std in enumerate(self.std_range):
                self.rate[n, m] = self.output_rate(i, std)


transfer = IF_curve(model, params)
transfer.compute_transfer()
dat = shelve.open(model + '_transfer.dat')
dat['I_mean'] = transfer.i_range
dat['I_std'] = transfer.std_range
dat['rate'] = transfer.rate
dat.close()