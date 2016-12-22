# coding=utf-8
"""
This is a default script that should be adapted to the respective purpose.

It is intended to compute the array length required to adequately scan
parameter space for a suitable configuration of a GIF2 neuron in a Brunel
network.

Insert k as the array length in one of the shell scripts in this folder!
"""

import numpy as np
from itertools import product
from mingtools1 import predict_str_freq

g1_range = np.arange(5.0, 85.0, 2.5)
dV1_range = np.arange(4.0, 9.5, 0.5)
dV2_range = np.arange(0.0, 20.0, 1.0)

k = 0
g1_range = np.arange(5.0, 85.0, 2.5)
dV1_range = np.arange(4.0, 9.5, 0.025)
dV2_range = np.arange(0.0, 16.0, 0.025)

networkparamdict = {'p_rate': 65000.0, 'C_m': 250.0, 'g': 25.0}
# len(g_range) * len(g1_range) * len(t1_range)
k = 0
for i in product(g1_range, dV1_range):
	# min = np.amin([ 50.0, i[ 0 ] - 10.0 ])
	t1_range = np.arange(50.0, 120.0, 10.0)
	for j in t1_range:
		if np.isclose(
				predict_str_freq(j, 10.0, i[ 0 ], 250.0, remote=True),
				10.0, atol=0.05, rtol=0.0):
			k += 1
print(k)
