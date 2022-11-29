#!/bin/bash
#SBATCH -p gpu
#SBATCH -q wildfire
#SBATCH -t 7-00:00:00   # time in d-hh:mm:ss
#SBATCH -o /scratch/lsaldyt/experiments/tune_t5_codexglue/%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e /scratch/lsaldyt/experiments/tune_t5_codexglue/%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-user=lsaldyt@asu.edu # Mail-to address
#SBATCH --mail-type=ALL  # Send an e-mail when a job starts, stops, or fails 
#SBATCH -N 1 
#SBATCH -c 32
#SBATCH --job-name=tune_t5_codexglue

#SBATCH --gres=gpu:1 # Request 1 GPU
#SBATCH -C V100
# Cuda library and includes handled naturally:

module load cuda/11.6.0
module load rclone/1.43
module load blas/3.7.0
module load OpenBLAS/0.3.19

# export CUDA_VISIBLE_DEVICES=0
# export TF_CPP_MIN_LOG_LEVEL=0
export INCLUDEPATH=$INCLUDEPATH:$HOME/cuda/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/cuda/lib64
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
# Turn off jax debug for now
# export XLA_FLAGS=--xla_dump_to=/scratch/lsaldyt/jax_debug/

echo "Running!
"
env
module list
nvcc --version
nvidia-smi
./run tune_t5_codexglue 
