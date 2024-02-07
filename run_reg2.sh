#!/bin/bash 

#SBATCH --job-name=lm_eqtl_reg
#SBATCH --output=logs/lm_eqtl_reg_%A_%a.out
#SBATCH --error=logs/lm_eqtl_reg_%A_%a.err
#--gpus=a40:1
#SBATCH --mem=32000

while [[ "$#" -gt 0 ]]; do
    case $1 in 
        --test) test_mode=true; shift ;;
        *) args+=("$1"); shift ;;
    esac
done

if [ "$test_mode" == true ]; then
    echo "Running in test mode"
    python model/eval.py "${args[@]}"
else    
    echo "Running in train mode"
    python model/train.py "${args[@]}"
fi