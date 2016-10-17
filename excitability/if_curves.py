# -*- coding: utf-8 -*-
"""
This file creates the IF-curves for the parameter-sets from
excitability1.py, which elicit simililar spiking in gif2 and
 iaf neurons. Those that do not create chaotic behaviour will
 then be tested in Brunel networks.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(precision=4, suppress=True)
cluster = True
import nest
nest.set_verbosity('M_WARNING')
nest.ResetKernel()
dt = 0.1  # the resolution in ms
nest.SetKernelStatus(
    {"resolution": dt, "print_time": False, "overwrite_files": True})
if cluster:
    nest.SetKernelStatus({"local_num_threads": 4})
try:
    nest.Install("gif2_module")
except:
    pass

# give as sysargv's:
# 1.) the filename with the unique_solution_array
# 2.) the index of the unique solution to test
# 3.) the filename to write the rates into

# idea: increase input poisson rate in steps of 500.
# Simulate each step for ~5000