#!/bin/bash
set -e

cd "$(readlink -f $(dirname "$0")/..)"
source ./scripts/setup.sh

target_path="${samples_dir}/*.wav"
output_dir="${samples_dir}"
mkdir -p ${output_dir}

for filename in $(ls $target_path)
do
    base=$(basename $filename)
    output="${output_dir}/${base%.wav}_spec.png"
    echo "generating $output"
    python tools/save_spectrogram.py --input $filename --output $output --gpu $gpu
done

