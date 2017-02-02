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

C_range = np.arange(250.0, 550.0, 10.0)
g_range = np.arange(5.0, 80.0, 2.5)
dV1_range = np.arange(3.0, 9.5, 0.1)

k = 0
# C
for C in C_range:
# g
	g = C / 10.0
# g1
	g1_range = np.arange(g, 80.0, 2.5)
# dV1
	for i in product(g1_range, dV1_range):
# t1
		t1_range = np.arange(50.0, 120.0, 10.0)
		for j in t1_range:
			if np.isclose(
					predict_str_freq(j, g, i[ 0 ], C, remote=True),
					10.0, atol=0.05, rtol=0.0):
				k += 1
print(k)
