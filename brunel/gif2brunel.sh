#!/bin/bash
#SBATCH --job-name gif2brunel
#SBATCH --time 04:00:00
#SBATCH --ntasks=16
#SBATCH -o /home/d.mingers/gif2_model_nest/brunel/out/brunel_%A.out
#SBATCH -e /home/d.mingers/gif2_model_nest/brunel/out/brunel_%A.err
#SBATCH --mem=20000
module load mpi/openmpi/1.10.0
source activate mingpython2

srun python brunel4gif2.py 0 standard 0.0
srun python brunel4gif2.py 0 modulated 0.2
