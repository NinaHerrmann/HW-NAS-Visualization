#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --mem=92G

#SBATCH --job-name=hwnasesp
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n_herr03@uni-muenster.de
#SBATCH --output=/scratch/tmp/%u/hwnas/report/%j.out 
#SBATCH --error=/scratch/tmp/%u/hwnas/report/%j.error 
# Load modules

# TODO: load relevant software stack from your HPC environment
# Below are packages we used that are necessary, however a clust might require more...
# E.g. we have module load palma/2024a
module load palma/2022b  
ml GCCcore/12.2.0
ml Python/3.10.8
ml parallel/20230722
ml CMake/3.24.3

NUMBER_OF_CPUS_PER_JOB=1
export OPENBLAS_NUM_THREADS=$NUMBER_OF_CPUS_PER_JOB
export MKL_NUM_THREADS=$NUMBER_OF_CPUS_PER_JOB
export OMP_NUM_THREADS=$NUMBER_OF_CPUS_PER_JOB
# TODO: Or your folders
home="$HOME"/HW-NAS-Visualization
wd="$WORK"/hwnas

cd $home
pip install -r ./requirements_memorygenerator.txt
log_path="$wd"/report/sublogs/hwnas_"$SLURM_JOB_ID"
echo $log_path
mkdir -p "$log_path"

result_dir=$wd/result
echo $result_dir

chunk_dir=$WORK/NATS_Benchmark/chunks
model_dir=$WORK/NATS_Benchmark/models


# Export variables for job environment (parallel will inherit env, but --env is explicit below)
export lgbm ms data_dir model_dir log_path result_dir
PARALLEL_JOBS_THEORETICAL=$(((SLURM_CPUS_ON_NODE-1)/NUMBER_OF_CPUS_PER_JOB))
# make sure value is > 1
PARALLEL_JOBS=$(( PARALLEL_JOBS_THEORETICAL > 1 ? PARALLEL_JOBS_THEORETICAL : 1 ))

# Option 1 (preferred): Chunked execution with (pseudo-)balanced chunks
# Adapt chunk size (max_chunk_trees) and max_rows_per_chunk to your needs or introduce other balancing criteria

# Run chunks in parallel
parallel -j "$PARALLEL_JOBS" --lb --joblog "$log_path/parallel_chunk_joblog.txt" \
	$home/runSingleespidf.sh {} ::: "$chunk_dir"/chunk_*
# End of script
