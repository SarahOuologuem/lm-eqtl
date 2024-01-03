#!/bin/bash

default_prog=model/reg_main.py
default_dataset="extdata/datasets/phase3_top10/dataset.parquet"
default_output_dir=outputs
default_model_weight="extdata/checkpoints/phase3_top10/aware_large_splitmsk/weights/epoch_100_weights_model.pt"
default_expression_data="extdata/GD660.GeneQuantRPKM.txt.gz"
default_device=cpu
default_batch_size=16
default_seq_len=5000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prog=*)
            prog="${1#*=}"
            shift
            ;;
        --prog)
            prog="${2:-model/reg_main.py}"
            shift 2
            ;;
        --dataset=*)
            dataset="${1#*=}"
            shift
            ;;
        --dataset)
            dataset="${2:-default_dataset}"
            shift 2
            ;;
        --output_dir=*)
            output_dir="${1#*=}"
            shift
            ;;
        --output_dir)
            output_dir="${2:-default_output_dir}"
            shift 2
            ;;
        --model_weight=*)
            model_weight="${1#*=}"
            shift
            ;;
        --model_weight)
            model_weight="${2:-default_model_weight}"
            shift 2
            ;;
        --expression_data=*)
            expression_data="${1#*=}"
            shift
            ;;
        --expression_data)
            expression_data="${2:-default_expression_data}"
            shift 2
            ;;
        --batch_size=*)
            batch_size="${1#*=}"
            shift
            ;;
        --batch_size)
            batch_size="${2:-default_batch_size}"
            shift 2
            ;;
        --seq_len=*)
            seq_len="${1#*=}"
            shift
            ;;
        --seq_len)
            seq_len="${2:-default_seq_len}"
            shift 2
            ;;
        --device=*)
            device="${1#*=}"
            shift
            ;;
        --device)
            device="${2:-default_device}"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Set default values if not passed
prog="${prog:-$default_prog}"
dataset="${dataset:-$default_dataset}"
output_dir="${output_dir:-$default_output_dir}"
model_weight="${model_weight:-$default_model_weight}"
expression_data="${expression_data:-$default_expression_data}"
batch_size="${batch_size:-$default_batch_size}"
seq_len="${seq_len:-$default_seq_len}"
device="${device:-$default_device}"


echo
echo "Program Configuration:"
echo "----------------------"
echo "prog: $prog"
echo "dataset: $dataset"
echo "output_dir: $output_dir"
echo "model_weight: $model_weight"
echo "expression_data: $expression_data"
echo "batch_size: $batch_size"
echo "seq_len: $seq_len"
echo "device: $device"
echo "----------------------"
echo



python $prog \
    --dataset $dataset \
    --output_dir $output_dir \
    --model_weight $model_weight \
    --expression_data $expression_data \
    --batch_size $batch_size \
    --seq_len $seq_len \
    --device $device



