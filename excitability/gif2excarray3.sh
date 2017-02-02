#!/bin/bash
#SBATCH --job-name gif2ex15
#SBATCH --array 0-13
#SBATCH -o /home/d.mingers/gif2_model_nest/excitability/out/excitability_%A.out
#SBATCH -e /home/d.mingers/gif2_model_nest/excitability/out/excitability_%A.err
#SBATCH --mem=10000
#SBATCH --time 03-23:59:00
module load mpi/openmpi/1.10.0
module load pystuff_new
source activate mingpython
srun python /home/d.mingers/gif2_model_nest/excitability/excitability_Cspec.py $SLURM_JOBID 1 $SLURM_ARRAY_TASK_ID
