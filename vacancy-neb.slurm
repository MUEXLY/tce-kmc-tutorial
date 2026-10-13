#!/bin/bash

#SBATCH --job-name v-neb
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 16gb
#SBATCH --time 10:00:00

module purge
module add anaconda3
uv run vacancy-neb.py

