#!/bin/bash
#SBATCH --ntasks 1
#SBATCH -o /home/d.mingers/diverses/brunel/out/brunel_%A.out
#SBATCH -e /home/d.mingers/diverses/brunel/out/brunel_%A.err
echo 'mpi load'
module load mpi/openmpi/1.10.0
module load pystuff_new
echo 'mpirun'
mpirun python /home/d.mingers/diverses/brunel/brunel_exp_blau3a.py $SLURM_JOBID