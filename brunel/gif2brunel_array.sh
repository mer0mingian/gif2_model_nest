#!/bin/bash
#SBATCH --job-name gif2brunel
#SBATCH --array 0-9%5
#SBATCH --time 01:00:00
#SBATCH -o /home/d.mingers/gif2_model_nest/brunel/out/brunel_%A.out
#SBATCH -e /home/d.mingers/gif2_model_nest/brunel/out/brunel_%A.err
#SBATCH --exclusive
#SBATCH --mem=10000
module load mpi/openmpi/1.10.0
module load pystuff_new
source activate mingpython
srun python brunel_exp_blau_brunelconn2.py $SLURM_ARRAY_TASK_ID 9