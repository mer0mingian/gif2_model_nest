#!/bin/bash
#SBATCH --job-name gif2test2
#SBATCH --array 0-1760%50
#SBATCH --time 02-23:59:59
#SBATCH -o /home/d.mingers/gif2_model_nest/brunel/out/testexc2_%A.out
#SBATCH -e /home/d.mingers/gif2_model_nest/brunel/out/testexc2_%A.err
#SBATCH --mem=10000

module load mpi/openmpi/1.10.0
source activate mingpython

srun python gif2_brunel_test_params.py $SLURM_ARRAY_TASK_ID

