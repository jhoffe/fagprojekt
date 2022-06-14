#!/bin/sh
#BSUB -J spect_heatmaps
#BSUB -o batch_jobs/logs/spect_heatmaps_%J.out
#BSUB -e batch_jobs/logs/spect_heatmaps_%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 16
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 06:00
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

export TEST_DATASET="asr_model/data/librispeech/authentic-test-clean.txt"

export MODEL="mixed"
export MODEL_PATH="asr_model/models/asr_model_$MODEL.pt"
export NAME="$MODEL"

python3 asr_model/spectrogram_regression.py

export MODEL="authentic"
export MODEL_PATH="asr_model/models/asr_model_$MODEL.pt"
export NAME="$MODEL"

python3 asr_model/spectrogram_regression.py

export MODEL="synthetic"
export MODEL_PATH="asr_model/models/asr_model_$MODEL.pt"
export NAME="$MODEL"

python3 asr_model/spectrogram_regression.py
