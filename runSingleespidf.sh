#!/bin/bash

chunk_file="${1}"
chunk="${chunk_file##*_}"     # remove everything up to last _
chunk="${chunk%.csv}" 
#echo "$chunk_file" | sed -E 's/.*_([0-9]+)\.txt/\1/'
#chunk="$chunk_file" | sed -E 's/.*_([0-9]+)\.txt/\1/'
weight_path=/scratch/tmp/n_herr03/NATS_Benchmark/NATS-tss-v1_0-3ffb9-full
echo ${chunk}
echo ${chunk_file}
cd "~/MemoryGenerator/"
# Ensure output directory exists

chunk_file="${1}"
model_dir="${2}"
result_dir="${3}"
weight_path=/mnt/dss_project/nherrmann/memorygenerator/NAS-Bench-201-v1_1-096897.pth
cd ~/MemoryGenerator/ || exit
# Ensure output directory exists
python3 AccuracyAndModelGenerator.py --file "${chunk_file}" --modelpath "${model_dir}" --resultpath "${result_dir}" --weightpath "${weight_path}"


# Check if the file exists and is not empty
if [ -s "$chunk_file" ]; then
  while read -r idx; do
    # skip empty lines / comments (optional)
    [[ -z "${idx:-}" ]] && continue
    [[ "${idx:0:1}" == "#" ]] && continue

    echo "Running idx=$idx chunk=$chunk"
    ./callespidf.sh "$idx" "$chunk" "$model_dir"
  done < "$chunk_file"
else
  echo "The file $chunk_file is empty or does not exist."
fi
