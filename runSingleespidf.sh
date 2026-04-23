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
chunk="${chunk_file##*_}"     # remove everything up to last _
chunk="${chunk%.txt}" 
#echo "$chunk_file" | sed -E 's/.*_([0-9]+)\.txt/\1/'
#chunk="$chunk_file" | sed -E 's/.*_([0-9]+)\.txt/\1/'
weight_path=/scratch/tmp/n_herr03/NATS_Benchmark/NATS-tss-v1_0-3ffb9-full
echo ${chunk}
echo ${chunk_file}
cd ~/HW-NAS-Visualization/
# Ensure output directory exists
IFS=$' \t\n'

# Check if the file exists and is not empty
if [ -s "$chunk_file" ]; then
  echo "read..."
  # Read the file line by line
  while read -r -a nums; do
    # Check if the array is not empty
    if [ ${#nums[@]} -gt 0 ]; then
      echo "${nums[@]}"
      for num in "${nums[@]}"; do
        echo "calling ${num} ${chunk}"
       ./callespidf.sh "$num" "$chunk"
      done
    else 
      echo "no nums"
    fi
  done < "$chunk_file"
else
  echo "The file $chunk_file is empty or does not exist."
fi

#while read -r -a nums; do
#  echo "${nums[@]}"
#  for num in "${nums[@]}"; do
#    echo "calling ${num} ${chunk}"
#    ./callespidf.sh "$num" $chunk
#  done
#done < "$chunk_file"

#rm -rf $TMPDIR/how_to_run_model${chunk}
#rm -rf "/scratch/tmp/n_herr03/hwnas/espproject/how_to_run_model${chunk}"
