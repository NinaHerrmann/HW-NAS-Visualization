#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=zen4
#SBATCH --time=48:00:00
#SBATCH --mem=1400G

#SBATCH --job-name=hwnasmissing
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


NUMBER_OF_CPUS_PER_JOB=32
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

chunk_dir=$WORK/NATS_Benchmark/joblist
model_dir=$WORK/NATS_Benchmark/models


# Export variables for job environment (parallel will inherit env, but --env is explicit below)
export lgbm ms data_dir model_dir log_path result_dir

$home/runSingleExperiment.sh "$model_dir" "$result_dir" "/scratch/tmp/n_herr03/NATS_Benchmark/missingjob/missing_cifar10.txt"
