#!/bin/bash

#SBATCH --job-name=esm_embedding
#SBATCH --partition=biodroid
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=esm_job.log

# Load Hamilton/Durham HPC modules
module load biodroid
module load esm

# Move to the directory containing this script
cd "$(dirname "$0")"

# Run ESM2 embedding generation
python3 run_esm_embedding.py
