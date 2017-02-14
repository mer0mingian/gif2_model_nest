#!/bin/bash
#SBATCH --job-name gif2brunel
#SBATCH --array 0-142%30
#SBATCH --time 11:59:59
#SBATCH -o /home/d.mingers/gif2_model_nest/brunel/out/brunel_%A.out
#SBATCH -e /home/d.mingers/gif2_model_nest/brunel/out/brunel_%A.err
#SBATCH --mem=10000
module load mpi/openmpi/1.10.0
srun python brunel4gif2.py 0 20 $SLURM_ARRAY_TASK_ID 