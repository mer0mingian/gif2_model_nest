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



with open('brunel_array_best_params_0.csv', 'a') as output:
	np.savetxt(output, allthetenbests, fmt="%12.6G", newline='\n')
	output.write(' \n')
	output.close()

# for i in np.arange(100):
# 	addindex = allthetenbests[ :, 8 ] == np.amax(allthetenbests[ :, 8 ])
# 	testconfigs = np.vstack((testconfigs, allthetenbests[ addindex, : ]))
# 	for j in np.arange(len(addindex)):
# 		if addindex[ j ]:
# 			allthetenbests = np.vstack((allthetenbests[ 0:j - 1, : ],
# 						  	allthetenbests[ (j + 1):, : ]))
# with open('brunel_array_best_results_0.csv', 'a') as output:
# 	testconfigs = np.reshape(testconfigs, (-1, 12))[ 1:, : ]
# 	np.savetxt(output, testconfigs, fmt="%12.6G", newline='\n')
# 	output.write(' \n')
# 	output.close()
#
# for i in np.arange(testconfigs.shape[ 0 ]):
# 	row = testconfigs[ i, : ]
# 	networkparamdict = {'p_rate':  65000.0, 'C_m': row[ 1 ], 'g': row[ 2 ],
# 						'g_1': row[ 3 ], 'tau_1': row[ 4 ], 'V_dist': row[ 5 ],
# 						'V_dist2': row[ 6 ]}
# 	fractionindex = 9
# 	fraction = np.arange(0.0, 20.0)[ fractionindex + 1 ] / 20.0
# 	resultlist, spikelists = run_brunel(networkparamdict, fraction)
# 	resultarray = np.array(resultlist)
# 	with open('brunel_array_results_0.csv', 'a') as output:
# 		np.savetxt(output, resultarray, fmt="%12.6G", newline=' ')
# 		output.write(' \n')
# 		output.close()

