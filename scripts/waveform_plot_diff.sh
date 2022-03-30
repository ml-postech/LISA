#!/bin/bash
set -e

cd "$(readlink -f $(dirname "$0")/..)"
source ./scripts/setup.sh

gpu=2

input_file1="samples/w0_300/vctk_sample_48000.wav"
input_file2="samples/w0_300/vctk_sample_result_48000_48000.wav"
#input_file3="samples/w0_35/vctk_sample_48000.wav"

size="0.01"
start="0"
end="1"

out_dir="waveform_diff"
mkdir -p ${out_dir}

output_file1="${out_dir}/${size}_diff.png"
#output_file2="${out_dir}/${size}_predicted.png"
#output_file3="${out_dir}/${size}_downsampled.png"


python tools/save_waveform_diff.py --original ${input_file1} --pred ${input_file2} --output ${output_file1} --gpu $gpu --time1 ${start} --time2 ${end}
#python tools/save_waveform.py --input ${input_file2} --output ${output_file2} --gpu $gpu --time1 ${start} --time2 ${end}
#python tools/save_waveform.py --input ${input_file3} --output ${output_file3} --gpu $gpu --time1 ${start} --time2 ${end}

