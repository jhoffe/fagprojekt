#!/bin/sh
module load scipy/1.6.3-python-3.9.6
python3 -m venv fagprojekt-env

source fagprojekt-env/bin/activate

# Install torch
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install other requirements
pip3 install -r requirements.txt
