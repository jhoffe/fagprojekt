import os

import torch
import torchaudio
from asr.data import BaseDataset
from multiprocessing import Pool
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, FrequencyMask
import warnings
from tqdm import *
import numpy as np

torch.random.manual_seed(42)
np.random.seed(42)

warnings.simplefilter("ignore")

SAMPLE_RATE = 22050
WINDOW_SIZE = 0.02
STRIDE = 0.01
NUM_MELS = 80
DATASET_PATH = os.environ["DATASET_PATH"]
CPU_CORES = int(os.environ["CPU_CORES"])
OUTPUT_PATH = os.environ['OUTPUT_PATH']
#API_KEY = os.environ['WANDB_API_KEY']

dataset = BaseDataset(source=DATASET_PATH, sort_by=0, preprocessor=[])
dataloader = torch.utils.data.DataLoader(dataset, num_workers=CPU_CORES // 2, pin_memory=True,
                                         batch_size=16, prefetch_factor=2)

noise_params = {
    "min_amplitude": 0.001,
    "max_amplitude": 0.01,
    "p": 1
}

time_stretch_params = {
    "min_rate": 0.9,
    "max_rate": 1.5,
    "p": 1
}

pitch_shift_params = {
    "min_semitones": -8,
    "max_semitones": 8,
    "p": 1
}

# wandb.init(project="test-project", entity="fagprojekt-synthetic-asr")
#
# wandb.config = {
#     "noise_params": noise_params,
#     "time_stretch_params": time_stretch_params,
#     "pitch_shift_params": pitch_shift_params
# }

augmenter = Compose([
    AddGaussianNoise(**noise_params),
    TimeStretch(**time_stretch_params),
    PitchShift(**pitch_shift_params),
    FrequencyMask(p=1)
], p=0.95)

pbar = tqdm(desc="Augmenting samples", total=len(dataloader))

def augment(batch):
    _examples, paths = batch

    for path in paths:
        sample, sample_rate = torchaudio.load(path + ".flac", format="flac")

        augmented_sample = augmenter(samples=sample.numpy().reshape(-1), sample_rate=sample_rate)

        filename = path.split("/")[-1]

        augmented_sample = torch.from_numpy(augmented_sample)[None, :]

        torchaudio.save(os.path.join(OUTPUT_PATH, filename + ".flac"), src=augmented_sample, sample_rate=SAMPLE_RATE,
                        format="flac")

        with open(path + ".txt", "r") as f:
            contents = f.read()

            new_file = open(os.path.join(OUTPUT_PATH, filename + ".txt"), "w")
            new_file.write(contents)
            new_file.close()

    pbar.update()


with Pool(CPU_CORES // 2) as pool:
    pool.map(augment, dataloader)

pbar.close()
