#!/bin/bash
#SBATCH --job-name gif2brunel2
#SBATCH --time 01:00:00
#SBATCH --array 0-16%16
#SBATCH -o /home/d.mingers/diverses/brunel/out/brunel_%A.out
#SBATCH -e /home/d.mingers/diverses/brunel/out/brunel_%A.err
#SBATCH --exclusive
#SBATCH --mem=10000
echo 'mpi load'
module load mpi/openmpi/1.10.0
module load pystuff_new
echo 'mpirun'
mpirun python /home/d.mingers/diverses/brunel/brunel_exp_blau4.py $SLURM_ARRAY_TASK_ID 16