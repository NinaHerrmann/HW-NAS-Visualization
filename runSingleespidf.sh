#!/bin/bash
# Some basic error checking on input parameters
if [ "$#" -lt 1 ]; then
    echo "ERROR: runSingleExperiment.sh requires 2 arguments but got $#."
    echo "Received args:"
    idx=1
    for a in "$@"; do
        printf " $%d = %q\n" "$idx" "$a"
        idx=$((idx+1))
    done
    echo "Usage: <chunk_file>"
    exit 2
fi

# Assign input parameters to named variables for clarity

chunk_file="${1}"
echo "$chunk_file" | sed -E 's/.*_([0-9]+)\.txt/\1/'
chunk="$chunk_file" | sed -E 's/.*_([0-9]+)\.txt/\1/'
weight_path=/scratch/tmp/n_herr03/NATS_Benchmark/NATS-tss-v1_0-3ffb9-full
cd ~/HW-NAS-Visualization/
# Ensure output directory exists
while read -r -a nums; do
  for num in "${nums[@]}"; do
    ./callespidf.sh "$num" "$chunk"
  done
done < "$chunk_file"

rm -rf "./how_to_run_model$chunk"
