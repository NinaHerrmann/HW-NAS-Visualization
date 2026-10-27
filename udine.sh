#!/bin/bash

PARALLEL_JOBS=32
export OPENBLAS_NUM_THREADS=$PARALLEL_JOBS
export MKL_NUM_THREADS=$PARALLEL_JOBS
export OMP_NUM_THREADS=$PARALLEL_JOBS

codehome="$HOME"/MemoryGenerator
codedocs="$HOME"/memorygenerator
dssmemory="/mnt/dss_project/nherrmann/MemoryGenerator"

cd $codehome
source /home/nherrmann/.espressif/tools/activate_idf_v6.1.sh
# check if we can install the additional reuqirements...
pip install -r ./requirements_accandmem.txt
log_path="$codedocs/log"
mkdir -p "$log_path"

result_dir="$codedocs/results"
echo $result_dir

chunk_dir="$codedocs/joblist"
model_dir="/mnt/dss_project/nherrmann/MemoryGenerator/models"


# Option 1 (preferred): Chunked execution with (pseudo-)balanced chunks
# Adapt chunk size (max_chunk_trees) and max_rows_per_chunk to your needs or introduce other balancing criteria
parallel -j "$PARALLEL_JOBS" --lb --joblog "$log_path/parallel_chunk_joblog.txt" \
	$codehome/runSingleespidf.sh {} $model_dir $result_dir ::: "$chunk_dir"/chunk_*