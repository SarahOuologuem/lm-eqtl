#!/bin/bash 

#SBATCH --job-name=lm_eqtl_reg
#SBATCH --output=logs/lm_eqtl_reg_%A_%a.out
#SBATCH --error=logs/lm_eqtl_reg_%A_%a.err
#SBATCH --gpus=1
#SBATCH --mem=32000

python model/train.py "$@"