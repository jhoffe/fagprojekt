import os

import torch
import torchaudio
import subprocess

LS_DATASET_TYPE = os.getenv('LS_DATASET_TYPE')
HPC_PATH = os.getenv("HPC_PATH") if "HPC_PATH" in os.environ else os.getcwd()
TORCH_MODELS_PATH = "{}/data/models".format(HPC_PATH)
LS_PATH = "{}/data/librispeech".format(HPC_PATH)
OUTPUT_PATH = "{}/data/synthetic_speech/{}".format(HPC_PATH, LS_DATASET_TYPE)
RATE = 22050

NUM_GPUS = torch.cuda.device_count()

if __name__ == '__main__':
    # Create needed directories
    if not os.path.exists(TORCH_MODELS_PATH):
        os.makedirs(TORCH_MODELS_PATH)

    if not os.path.exists(LS_PATH):
        os.makedirs(LS_PATH)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Set another local directory to download the models
    # torch.hub.set_dir(TORCH_MODELS_PATH)

    ds = torchaudio.datasets.LIBRISPEECH(root=LS_PATH, url=LS_DATASET_TYPE, download=True)

    rest = len(ds) % NUM_GPUS
    samples_per_shard = len(ds) // NUM_GPUS

    ds_indexes = [(k * samples_per_shard, (k + 1) * samples_per_shard) for k in range(NUM_GPUS)]

    pids = []

    for rank, (ds_start, ds_end) in enumerate(ds_indexes):
        pids.append(subprocess.Popen(["python", "-m", "synthetic_gen.generate_new", "--dataset", LS_DATASET_TYPE, "--idxstart", ds_start, "--idxend", ds_end, '--rank', rank, '--batchsize', 16]))

    while len(pids) > 0:
        for i, pid in enumerate(pids):
            if pid.poll() is not None:
                del pids[i]