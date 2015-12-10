#!/bin/bash
#SBATCH -n 64 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-3:00 # Runtime in hours
#SBATCH -p hernquist # Partition to submit to
#SBATCH --mem-per-cpu=1024 # Memory per cpu in MB (see also --mem)
#SBATCH -o output/batch_pipeline.out # Standard out goes to this file
#SBATCH -e output/batch_pipeline.err # Standard err goes to this filehostname
#SBATCH --mail-type=END,FAIL #alert when done
#SBATCH --mail-user=bcook@cfa.harvard.edu #Email to send to

python pitching_multi.py
