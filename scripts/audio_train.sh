#!/bin/bash

cd "$(readlink -f $(dirname "$0")/..)"
source ./scripts/setup.sh

mkdir -p ${logs_dir}
mkdir -p "${save_dir}/${model_name}"
log_git_file="${model_name}_git_info.log"

git log -1 > "${logs_dir}/${log_git_file}"
git diff --staged >> "${logs_dir}/${log_git_file}"
git diff >> "${logs_dir}/${log_git_file}"

python -u ${train_path} --config ${config_file_path} --name ${model_name} --gpu $gpu --save_path ${save_dir} 2>&1 | tee ${logs_dir}/${model_name}.out &
disown -a

sleep 10s

cp "${logs_dir}/${log_git_file}" "${save_dir}/${model_name}/${log_git_file}"
