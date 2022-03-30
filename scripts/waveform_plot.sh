#!/bin/bash
set -e

cd "$(readlink -f $(dirname "$0")/..)"
source ./scripts/setup.sh

# start_list=( 0 0 0 )
# end_list=( 100 1 0.1 )

out_dir="${samples_dir}"
mkdir -p ${out_dir}


target_path="${samples_dir}/*.wav"
output_dir="${samples_dir}"
mkdir -p ${output_dir}

for filename in $(ls $target_path)
do
    for i in "${!start_list[@]}"; do
        base=$(basename $filename)
        output="${output_dir}/${base%.wav}_wave_${start_list[$i]}_${end_list[$i]}.png"
        echo "generating $output"
        python tools/save_waveform.py --input ${filename} --output ${output} --gpu $gpu --time1 ${start_list[$i]} --time2 ${end_list[$i]}
    done
done

