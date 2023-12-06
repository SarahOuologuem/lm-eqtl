#!/bin/bash

#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -x gpusrv46,gpusrv47,gpusrv48,gpusrv49
#SBATCH -o /lustre/groups/epigenereg01/workspace/projects/vale/SysGen2023/slurm_logs/%a.o
#SBATCH -e /lustre/groups/epigenereg01/workspace/projects/vale/SysGen2023/slurm_logs/%a.e

job_name='agnostic_large'

dataset_name='phase3_top100'

dataset="/lustre/groups/epigenereg01/workspace/projects/vale/SysGen2023/data/new/${dataset_name}/dataset.parquet"

common_params="--dataset $dataset --train_splits 8 --tot_epochs 20 \
--batch_size 16 --d_model 256 --n_layers 16 --agnostic 1 --save_at 20 \
--masking stratified_maf"

checkpoint_dir="/lustre/groups/epigenereg01/workspace/projects/vale/SysGen2023/checkpoints/state_space/new/${dataset_name}/$job_name/"

script_dir='/home/icb/sergey.vilov/workspace/PRS/SysGen2023/model/'

cd $script_dir

run_test () {

    export LD_LIBRARY_PATH=~/miniconda3/lib
    source ~/.bashrc; conda activate mlm

    echo $test_name $current_params

    output_dir="$checkpoint_dir/$test_name/"
    mkdir -p $output_dir

    srun python main.py ${common_params} ${current_params} --output_dir ${output_dir} > ${output_dir}/log 2>${output_dir}/err

}

run_test
