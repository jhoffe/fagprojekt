import os

from asr.data import BaseDataset
from asr.data.preprocessors import SpectrogramPreprocessor, TextPreprocessor
from asr.modules import ASRModel
from asr.data.batch_sampler import UniformBatchSampler
from runner import Runner

from torch.utils.data import DataLoader
import torch
import random
import numpy
import wandb

TRAIN_DATASET_PATH = os.environ['TRAIN_DATASET']
VAL_DATASET_PATH = os.environ['TEST_DATASET']
MODELS_OUTPUT_PATH = os.environ['MODELS_PATH']
TRAIN_UPDATES = int(os.environ['TRAIN_UPDATES'])
BATCH_SIZE = int(os.environ['BATCH_SIZE'])
RESULTS_PATH = os.environ['RESULTS_PATH']
NAME = os.environ['NAME']
CPU_CORES = int(os.environ['CPU_CORES'])
LOAD_SPECTROGRAMS = int(os.environ["LOAD_SPECTROGRAMS"])
PRETRAINED_MODEL_PATH = os.environ['PRETRAINED_MODEL_PATH']

assert TRAIN_UPDATES > 0
assert BATCH_SIZE > 0

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
numpy.random.seed(SEED)

text_preprocessor = TextPreprocessor()

if LOAD_SPECTROGRAMS == 1:
    train_spec_preprocessor = SpectrogramPreprocessor(output_format='NFT', sample_rate=22050, ext=".flac",
                                                      should_augment=True, skip_spec=True)
else:
    train_spec_preprocessor = SpectrogramPreprocessor(output_format='NFT', sample_rate=22050, ext=".flac",
                                                      should_augment=True)

val_spec_preprocessor = SpectrogramPreprocessor(output_format='NFT', sample_rate=22050, ext=".flac",
                                                    should_augment=False)

val_preprocessor = [val_spec_preprocessor, text_preprocessor]
train_preprocessor = [train_spec_preprocessor, text_preprocessor]

train_dataset = BaseDataset(source=TRAIN_DATASET_PATH, preprocessor=train_preprocessor, sort_by=0)
val_dataset = BaseDataset(source=VAL_DATASET_PATH, preprocessor=val_preprocessor, sort_by=0)

train_sampler = UniformBatchSampler(len(train_dataset), TRAIN_UPDATES, BATCH_SIZE, seed=SEED)

train_loader = DataLoader(train_dataset, num_workers=CPU_CORES, pin_memory=True, collate_fn=train_dataset.collate,
                          batch_sampler=train_sampler, prefetch_factor=2)
val_loader = DataLoader(val_dataset, num_workers=CPU_CORES, pin_memory=True, collate_fn=val_dataset.collate,
                        batch_size=BATCH_SIZE, prefetch_factor=2)

model_params = torch.load(PRETRAINED_MODEL_PATH) # For CPU: remove .cuda()

model = ASRModel().cuda()
model.load_state_dict(model_params)

LR = 3e-4

runner = Runner(
    model=model,
    name=NAME,
    train_loader=train_loader,
    val_loader=val_loader,
    stat_path=RESULTS_PATH,
    models_path=MODELS_OUTPUT_PATH,
    validate_every=2000,
    lr=LR
)

run = wandb.init(project="asr", entity="fagprojekt-synthetic-asr", reinit=True, config={
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "seed": SEED
        })

run.name = f"{NAME}-{run.id}"

runner.train()

run.finish()
