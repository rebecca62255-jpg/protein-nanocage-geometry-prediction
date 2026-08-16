#!/bin/bash
#SBATCH --job-name=esm_embedding
#SBATCH --partition=biodroid
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=/nobackup/rmgl20/dissertation/scripts/esm_job.log

module load biodroid
module load esm

cd /nobackup/rmgl20/dissertation/scripts
python3 run_esm_embedding.py
