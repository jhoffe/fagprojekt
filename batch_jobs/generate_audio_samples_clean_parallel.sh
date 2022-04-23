#!/bin/sh
#BSUB -J synthetic_generate
#BSUB -o logs/synthetic_generate_%J.out
#BSUB -e Logs/synthetic_generate_%J.err
#BSUB -q gpua100
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=4G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 01:00
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

LS_DATASET_TYPE="test-clean" HPC_PATH="/work3/s204071/fagprojekt" python3 -m synthetic_gen.generate_parallel