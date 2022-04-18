#!/bin/sh
#BSUB -J train_asr
#BSUB -o train_asr_%J.out
#BSUB -e train_asr_%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=4G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u s204071@student.dtu.dk
#BSUB -B
#BSUB -N
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load scipy/1.6.3-python-3.9.6

# load CUDA (for GPU support)
module load cuda/11.3

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source fagprojekt-env/bin/activate

TRAIN_DATASET="asr_model/data/librispeech/authentic-train-clean-100.txt" TEST_DATASET="asr_model/data/librispeech/authentic-test-clean.txt" python3 asr_model/experiment.py
