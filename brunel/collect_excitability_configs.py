# coding=utf-8
"""
For all files in the excitability folder this script picks out the ten best
configurations per file and tests them on the Burnel network in gif2_brunel_f.
"""
import numpy as np
import time
import sys
import os
import copy
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# import nest
# dt = 0.1
# nest.set_verbosity('M_WARNING')
# nest.ResetKernel()
# nest.SetKernelStatus(
# 		{"resolution": dt, "print_time": True, "overwrite_files": True})
# nest.SetKernelStatus({"local_num_threads": 16})
# try:
# 	nest.Install("gif2_module")
# except:
# 	pass

from mingtools1 import *
from elephant.statistics import isi, cv
from mc_connectivity_transformer import compute_new_connectivity

os.chdir('/mnt/beegfs/home/d.mingers/gif2_model_nest/brunel')
# from gif2_brunel_f import run_brunel

os.chdir('/mnt/beegfs/home/d.mingers/gif2_model_nest/excitability/')  # /closegs
files = os.listdir('.')
files = np.sort(files)

# allconfigs = np.zeros(12)
tenbestperfile = np.zeros(12)
allthetenbests = np.zeros(12)

firstadd = True
for k, f in enumerate(files):
	if f.find('AdvExcitability') > (-1):
		with open(f, 'r') as inputs:
			temp = np.loadtxt(inputs)
			for i in np.arange(10):
				addindex = temp[ :, 8 ] == np.amin(temp[ :, 8 ])
				tenbestperfile = np.vstack((tenbestperfile, temp[ addindex, : ]))
				for j in np.arange(len(addindex)):
					if addindex[ j ]:
						temp = np.vstack((temp[ 0:j - 1, : ], temp[ j + 1:, : ]))
						temp[ 0, -1 ] = k
			inputs.close()
		if firstadd:  # erase the zero-row in the beginnning
			allthetenbests = np.vstack((allthetenbests, tenbestperfile[ 1:, : ]))
			allthetenbests = allthetenbests[ 1:, : ]
			firstadd = False
		else:
			allthetenbests = np.vstack((allthetenbests, tenbestperfile[ 1:, : ]))
		tenbestperfile = np.zeros(12)  # don't add all file best cases every time

os.chdir('/mnt/beegfs/home/d.mingers/gif2_model_nest/brunel')

with open('brunel_array_best_params_2.csv', 'a') as output:
	np.savetxt(output, allthetenbests[ :, 1: ], fmt="%12.6G", newline='\n')
	output.write(' \n')
	output.close()

