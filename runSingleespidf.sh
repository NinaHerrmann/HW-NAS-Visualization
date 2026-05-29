#!/bin/bash

chunk_file="${1}"
chunk="${chunk_file##*_}"     # remove everything up to last _
chunk="${chunk%.csv}" 
#echo "$chunk_file" | sed -E 's/.*_([0-9]+)\.txt/\1/'
#chunk="$chunk_file" | sed -E 's/.*_([0-9]+)\.txt/\1/'
weight_path=/scratch/tmp/n_herr03/NATS_Benchmark/NATS-tss-v1_0-3ffb9-full
echo ${chunk}
echo ${chunk_file}
cd ~/HW-NAS-Visualization/
# Ensure output directory exists
IFS=$' \t\n'


chunk_file="${1}"
model_dir="${1}"
result_dir="${2}"
weight_path=/scratch/tmp/n_herr03/NATS_Benchmark/NATS-tss-v1_0-3ffb9-full
cd ~/HW-NAS-Visualization/
# Ensure output directory exists
python MemoryGeneratorppq.py --file "${chunk_file}" --modelpath "${model_dir}" --resultpath "${result_dir}" --weightpath "${weight_path}"


# Check if the file exists and is not empty
if [ -s "$chunk_file" ]; then
  while read -r idx; do
    # skip empty lines / comments (optional)
    [[ -z "${idx:-}" ]] && continue
    [[ "${idx:0:1}" == "#" ]] && continue

    echo "Running idx=$idx chunk=$chunk"
    ./callespidf.sh "$idx" "$chunk"
  done < "$chunk_file"
else
  echo "The file $chunk_file is empty or does not exist."
fi
