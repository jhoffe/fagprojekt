#!/bin/sh
#BSUB -J augmented_spect_gen_%J
#BSUB -o logs/augmented_spect_gen_%J.out
#BSUB -e logs/augmented_spect_gen_%J.err
#BSUB -n 32
#BSUB -R "rusage[mem=2G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 04:00
#BSUB -u s204071@student.dtu.dk
#BSUB -B
#BSUB -N
# end of BSUB options

cd /work3/s204071/fagprojekt

# load a scipy module
# replace VERSION and uncomment
module load scipy/1.6.3-python-3.9.6

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source fagprojekt-env/bin/activate

CPU_CORES=32 python3 asr_model/augmented_spect_gen.py
