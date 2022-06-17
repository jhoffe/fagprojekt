#!/bin/sh
#BSUB -J evaluate_models_finetune
#BSUB -o batch_jobs/logs/evaluate_models_finetune_%J.out
#BSUB -e batch_jobs/logs/evaluate_models_finetune_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 16
#BSUB -R "rusage[mem=4G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 00:10
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

# AUTHENTIC
export MODEL="authentic-finetune"
export MODEL_PATH="asr_model/models/asr_model_$MODEL.pt"

export NAME="$MODEL-on-authentic-test-clean"
export TEST_DATASET="asr_model/data/librispeech/authentic-test-clean.txt"
python3 asr_model/load_model.py

export NAME="$MODEL-on-authentic-test-other"
export TEST_DATASET="asr_model/data/librispeech/authentic-test-other.txt"
python3 asr_model/load_model.py

export NAME="$MODEL-on-synthetic-test-clean"
export TEST_DATASET="asr_model/data/librispeech/synthetic-test-clean.txt"
python3 asr_model/load_model.py

export NAME="$MODEL-on-synthetic-test-other"
export TEST_DATASET="asr_model/data/librispeech/synthetic-test-other.txt"
python3 asr_model/load_model.py