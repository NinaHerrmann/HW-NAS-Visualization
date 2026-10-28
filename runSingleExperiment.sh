#!/bin/bash
# Some basic error checking on input parameters
if [ "$#" -lt 2 ]; then
    echo "ERROR: runSingleExperiment.sh requires 2 arguments but got $#."
    echo "Received args:"
    idx=1
    for a in "$@"; do
        printf " $%d = %q\n" "$idx" "$a"
        idx=$((idx+1))
    done
    echo "Usage: <model_dir> <result_dir>"
    exit 2
fi

# Assign input parameters to named variables for clarity
model_dir="${1}"
result_dir="${2}"
chunk_file="${3}"
weight_path=/mnt/dss_lsc_europe/NinaExperiments/MemoryGenerator/weights/NAS-Bench-201-v1_1-096897
cd ~/MemoryGenerator/
# Ensure output directory exists
python MissingJobsppq.py -file "${chunk_file}" --modelpath "${model_dir}" --resultpath "${result_dir}" --weightpath "${weight_path}"

