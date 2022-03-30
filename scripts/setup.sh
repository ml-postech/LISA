#!/bin/bash
set -e

#
# conda setup
#
source ~/anaconda3/etc/profile.d/conda.sh
conda activate audio


#
# exp setup
#
gpu=0
model_name="final_4x"

export CUDA_VISIBLE_DEVICES=$gpu
export MASTER_PORT=49000
export MASTER_ADDR=localhost

#
# directory setup
#
logs_dir="./logs"
save_dir="/data/sss/save"
samples_dir="/data/sss/samples/${model_name}"

#
# exp setup 2
#
model_path="${save_dir}/${model_name}/epoch-last.pth"
config_file_path="configs/audio/lisa.yaml"
train_path="train_lisa.py"

#
# demo setup
#
demo_audio_path="/data/datasets/VCTK-Corpus/wav48/p306/p306_205.wav"
demo_audio_name="vctk_sample"

down_srs=( 12000 )
up_srs=( 48000 )

#
# waveform setup
#
start_list=( 0 0 0 )
end_list=( 100 1 0.1 )
