#!/bin/sh
#BSUB -J train_wavenist_dgm
#BSUB -o batch_jobs/logs/train_wavenist_dgm_%J.out
#BSUB -e batch_jobs/logs/train_wavenist_dgm_%J.err
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 12
#BSUB -R "rusage[mem=4G]"
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

export WANDB_API_KEY="5403fe6e39e261a91fd0a604a0ea7e22c75927cf"

python3 Mnist/wavenist_dgm.py
