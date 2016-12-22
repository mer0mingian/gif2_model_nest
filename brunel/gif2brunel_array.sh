#!/bin/bash
#SBATCH --job-name gif2brunel
#SBATCH --array 0-17%18
#SBATCH --time 01:00:00
#SBATCH -o /home/d.mingers/gif2_model_nest/brunel/out/brunel_%A.out
#SBATCH -e /home/d.mingers/gif2_model_nest/brunel/out/brunel_%A.err
#SBATCH --mem=10000
module load mpi/openmpi/1.10.0
source activate mingpython
srun python gif2_brunel_f.py $SLURM_ARRAY_TASK_ID 17