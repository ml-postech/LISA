#!/bin/bash
set -e

cd "$(readlink -f $(dirname "$0")/..)"
source ./scripts/setup.sh

#down_srs=(4000 8000 16000)
#up_srs=(4000 8000 16000 48000)

mkdir -p ${samples_dir}

for down_sr in ${down_srs[@]}
do
    for up_sr in ${up_srs[@]}
    do
        original_path="${samples_dir}/${demo_audio_name}_original.wav"
        downsampled_path="${samples_dir}/${demo_audio_name}_${down_sr}.wav"
        result_path="${samples_dir}/${demo_audio_name}_result_${down_sr}_${up_sr}.wav"
        expected_path="${samples_dir}/${demo_audio_name}_${up_sr}.wav"

        cp ${demo_audio_path} $original_path
        python tools/audio_downsample.py --input ${demo_audio_path} --output $downsampled_path --sr $down_sr --gpu $gpu
        python tools/audio_downsample.py --input ${demo_audio_path} --output $expected_path --sr $up_sr --gpu $gpu
        python audio_demo_batch.py --model ${model_path} --input $downsampled_path --output $result_path --sr $up_sr --gpu $gpu
    done
done
