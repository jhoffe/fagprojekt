import os
import pickle
from time import time
from math import cos

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
    
    def __call__(self, loader):

        self.total = len(loader) # may not be constant
        self.minor = 0
        self.major += 1

        for tracker in self.metric_trackers:
            tracker.reset()
        
        start = time()
        log_len = 0
        for batch in loader:
            self.minor += 1
            yield batch
            print(log_len * ' ', end='\r') # hack to clear output if previous log was longer
            metrics_log = ', '.join([t() for t in self.metric_trackers])
            progress = f'{self.minor}/{self.total}, {(time() - start) / 60:.1f} min(s)'
            log_line = f'{self.name} [{progress}]: {metrics_log}'
            print(log_line, end=('\n' if self.minor == self.total else'\r'))
            log_len = len(log_line)
