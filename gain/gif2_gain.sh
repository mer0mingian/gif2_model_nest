#!/bin/bash
#SBATCH --job-name gif2_gain
#SBATCH --array 0-63%8
#SBATCH --time 01:00:00
# %A will be replaced by the job ID and %a by the array index
#SBATCH -o /home/d.mingers/gif2_model_nest/gain/results/gif2_gain_%A.out
#SBATCH -e /home/d.mingers/gif2_model_nest/gain/results/gif2_gain_%A.err
#SBATCH --exclusive
#SBATCH --mem=10000
##################################################################
module load pystuff_new
source activate mingpython
srun python GIF2val_execution.py $SLURM_ARRAY_TASK_ID 63