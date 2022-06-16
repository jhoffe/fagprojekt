#!/bin/sh
#BSUB -J create_synthetic_datalists
#BSUB -o logs/create_synthetic_datalists_%J.out
#BSUB -e logs/create_synthetic_datalists_%J.err
#BSUB -n 16
#BSUB -R "rusage[mem=1G]"
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

CPU_CORES=16 python3 scripts/create_data_list.py
CPU_CORES=16 python3 scripts/create_data_list.py
CPU_CORES=16 python3 scripts/create_data_list.py