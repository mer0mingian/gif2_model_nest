#!/bin/bash
#SBATCH --job-name gif2gain
#SBATCH --array 0-71%18
#SBATCH --time 03-23:59:59
# %A will be replaced by the job ID and %a by the array index
#SBATCH -o /home/d.mingers/gif2_model_nest/gain/results/gif2_gain_%A.out
#SBATCH -e /home/d.mingers/gif2_model_nest/gain/results/gif2_gain_%A.err
#SBATCH --mem=10000
##################################################################
module load pystuff_new
source activate mingpython
srun python gif2_gain.py $SLURM_ARRAY_TASK_ID 71 0 