#!/bin/bash
# https://www.csn.fz-juelich.de/wiki/Blaustein
#SBATCH --output=/home/d.mingers/diverses/richardson_tests/test_job_%j.out
#SBATCH --error=/home/d.mingers/diverses/richardson_tests/test_job_%j.err
#SBATCH --nodes=4-4
module load pystuff_new
module load mpi/openmpi/1.10.0
mpirun python gif2_r03f6cd_blau.py