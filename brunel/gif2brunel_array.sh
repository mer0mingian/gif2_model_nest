#!/bin/bash
#SBATCH --job-name gif2brunel
#SBATCH --array 0-7%8
#SBATCH --time 23:59:59
#SBATCH -o /home/d.mingers/gif2_model_nest/brunel/out/brunel_%A.out
#SBATCH -e /home/d.mingers/gif2_model_nest/brunel/out/brunel_%A.err
#SBATCH --mem=10000
module load mpi/openmpi/1.10.0
source activate mingpython
srun python gif2_brunel_f.py $SLURM_ARRAY_TASK_ID 7 999