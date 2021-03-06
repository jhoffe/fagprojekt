import os
import pickle
from time import time
from math import cos
from tqdm import tqdm

import torch
from scipy.io import wavfile

def batch_to_tensor(batch, device='cuda'):
    if isinstance(batch, list):
        return [batch_to_tensor(x, device=device) for x in batch]
    else:
        return torch.from_numpy(batch).to(device)

def epochs(steps):
    BOLD = '\033[1m'
    END = '\033[0m'
    for i in range(1, steps + 1):
        print(BOLD + f'\nEpoch {i}' + END)
        yield i

class Logger():

    def __init__(self, name, *metric_trackers):

        self.name = name
        self.total = None
        self.minor = 0
        self.major = 0
        self.metric_trackers = metric_trackers

    def __len__(self):
        return self.total

    def reset(self):
        for tracker in self.metric_trackers:
            tracker.reset()

    def __call__(self, loader):

        self.total = len(loader) # may not be constant
        self.minor = 0
        self.major += 1

        self.reset()

        tqdm._instances.clear()
        ploader = tqdm(loader, desc=f"Processing batches for {self.name}")
        for batch in ploader:
            self.minor += 1
            yield batch
            metrics_log = ', '.join([t() for t in self.metric_trackers])
            ploader.set_description(desc=f"Processing batches for {self.name}: {metrics_log}", refresh=True)