#!/bin/bash

#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -x gpusrv31,gpusrv46,gpusrv47,gpusrv48,gpusrv49
#SBATCH -o /lustre/groups/epigenereg01/workspace/projects/vale/SysGen2023/slurm_logs/%a.o
#SBATCH -e /lustre/groups/epigenereg01/workspace/projects/vale/SysGen2023/slurm_logs/%a.e


job_name='hpp_search'

dataset_name='phase3_top10'

dataset="/lustre/groups/epigenereg01/workspace/projects/vale/SysGen2023/data/new/${dataset_name}/dataset.parquet"

common_params="--dataset $dataset --validate_every 1 --train_splits 1 --tot_epochs 1000 --Nfolds 5 --fold 0 --save_at 100:1000:100"

checkpoint_dir="/lustre/groups/epigenereg01/workspace/projects/vale/SysGen2023/checkpoints/state_space/no_maf/${dataset_name}/$job_name/"

script_dir='/home/icb/sergey.vilov/workspace/PRS/SysGen2023/model/'

model_params=$(pwd)/params/$job_name.txt

cd $script_dir

run_test () {

    export LD_LIBRARY_PATH=~/miniconda3/lib
    source ~/.bashrc; conda activate mlm

    echo $task_name $task_params

    output_dir="$checkpoint_dir/$task_name/"
    mkdir -p $output_dir

    srun python main.py ${common_params} ${task_params} --output_dir ${output_dir} > ${output_dir}/log 2>${output_dir}/err

}

mkdir -p ${checkpoint_dir}

experiment_num=$((${SLURM_ARRAY_TASK_ID}+1))

if [ "$experiment_num" -eq "1" ]; then
    cat $model_params |awk -v job_id=${SLURM_ARRAY_JOB_ID} '{print job_id"_"NR-1" "$1}' > "${checkpoint_dir}/jobs.txt"
fi

task_params=$(sed $experiment_num'!d' $model_params)

read -r task_name task_params <<< $task_params

task_params=$(eval "echo $task_params")

run_test
