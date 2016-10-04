#!/bin/bash
#SBATCH --job-name gif2_gain
#SBATCH --array 0-107%18
#SBATCH --time 01:00:00
# %A will be replaced by the job ID and %a by the array index
#SBATCH -o /home/d.mingers/diverses/gain/gif2_gain_%A.out
#SBATCH -e /home/d.mingers/diverses/gain/gif2_gain_%A.err
#SBATCH --exclusive
#SBATCH --mem=10000
##################################################################
conda activate mingpython
srun python GIF2val_execution.py $SLURM_ARRAY_TASK_ID 107