#!/bin/bash
#SBATCH --partition cs05r 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=6000 
#SBATCH --job-name=fast-rsm
#SBATCH --output=${outdir}/slurm-%j.out
 
${python_version} ${save_path}
