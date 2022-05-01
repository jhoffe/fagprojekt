#!/bin/sh
#BSUB -J augment_synthetic
#BSUB -o batch_jobs/logs/augment_synthetic_%J.out
#BSUB -e batch_jobs/logs/augment_synthetic_%J.err
#BSUB -n 32
#BSUB -R "rusage[mem=1G]"
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

export DATASET_PATH="asr_model/data/librispeech/synthetic-train.txt"
export OUTPUT_PATH="/work3/s204096/synthetic_data"
export CPU_CORES=32

python3 asr_model/experiment_uniform_batching.py
