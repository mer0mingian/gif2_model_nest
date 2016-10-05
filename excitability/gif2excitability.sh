#!/bin/bash
#SBATCH --ntasks 1
#SBATCH -o /home/d.mingers/gif2_model_nest/excitability/out/brunel_%A.out
#SBATCH -e /home/d.mingers/gif2_model_nest/excitability/out/brunel_%A.err
#SBATCH --exclusive
#SBATCH --mem=10000
#SBATCH --time 23:59:00

module load mpi/openmpi/1.10.0
module load pystuff_new
source activate mingpython

srun python /home/d.mingers/gif2_model_nest/excitability/excitability1.py $SLURM_JOBID