#!/bin/bash
#
#SBATCH --partition cs05r 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=2500 
#SBATCH --job-name=fast-rsm
 
/dls_sw/apps/fast_rsm/current/conda_env/bin/python map.py


