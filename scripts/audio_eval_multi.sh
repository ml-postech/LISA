#!/bin/bash

cd "$(readlink -f $(dirname "$0")/..)"
source ./scripts/setup.sh

mkdir -p ${logs_dir}
mkdir -p "${save_dir}/${model_name}"
log_git_file="${model_name}_git_info.log"



for target_sr in {12000..48000..2000}
# for target_sr in {8000..48000..2000}
do
    python -u eval_lisa.py --config ${config_file_path} --name ${model_name} --gpu $gpu --sr $target_sr | tee -a sr_out_4x_ensemble.out
done
