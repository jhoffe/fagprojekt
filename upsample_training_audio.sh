#!/bin/sh
#BSUB -J authentic_upsampled
#BSUB -o authentic_upsampled_%J.out
#BSUB -e authentic_upsampled_%J.err
#BSUB -n 16
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 00:10
#BSUB -u s204071@student.dtu.dk
#BSUB -B
#BSUB -N
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load scipy/1.6.3-python-3.9.6

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source fagprojekt-env/bin/activate

CPU_CORES=16 python3 -m upsampler.upsample
