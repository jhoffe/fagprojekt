#!/bin/sh
#BSUB -J train_asr_synthetic
#BSUB -o batch_jobs/logs/train_asr_synthetic_%J.out
#BSUB -e batch_jobs/logs/train_asr_synthetic_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=4G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u s204071@student.dtu.dk
#BSUB -B
#BSUB -N
# end of BSUB options

cd /work3/s204071/fagprojekt

# load a scipy module
# replace VERSION and uncomment
module load scipy/1.6.3-python-3.9.6

# load CUDA (for GPU support)
module load cuda/11.3

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source fagprojekt-env/bin/activate

export TRAIN_DATASET="asr_model/data/librispeech/synthetic-train.txt"
export TEST_DATASET="asr_model/data/librispeech/authentic-test-clean.txt"
export MODELS_PATH="asr_model/models"
export TRAIN_UPDATES=500000
export BATCH_SIZE=16
export RESULTS_PATH="asr_model/results"
export NAME="synthetic"

python3 asr_model/experiment_uniform_batching.py
