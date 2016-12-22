#!/bin/bash
#SBATCH --job-name gif2brunel
#SBATCH --array 0-660%60
#SBATCH --time 04-23:59:59
#SBATCH -o /home/d.mingers/gif2_model_nest/brunel/out/brunel3_%A.out
#SBATCH -e /home/d.mingers/gif2_model_nest/brunel/out/brunel3_%A.err
#SBATCH --mem=10000
module load mpi/openmpi/1.10.0
source activate mingpython
srun python gif2_brunel3.py 9 $SLURM_ARRAY_TASK_ID 8
