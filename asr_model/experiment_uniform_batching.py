import os

from asr.data import BaseDataset
from asr.data.preprocessors import SpectrogramPreprocessor, TextPreprocessor
from asr.modules import ASRModel
from asr.data.batch_sampler import UniformBatchSampler
from runner import Runner

from torch.utils.data import DataLoader

TRAIN_DATASET_PATH = os.environ['TRAIN_DATASET']
VAL_DATASET_PATH = os.environ['TEST_DATASET']
MODELS_OUTPUT_PATH = os.environ['MODELS_PATH']
TRAIN_UPDATES = int(os.environ['TRAIN_UPDATES'])
BATCH_SIZE = int(os.environ['BATCH_SIZE'])
RESULTS_PATH = os.environ['RESULTS_PATH']
NAME = os.environ['NAME']

assert TRAIN_UPDATES > 0
assert BATCH_SIZE > 0

spec_preprocessor = SpectrogramPreprocessor(output_format='NFT', sample_rate=22050, ext=".flac")
text_preprocessor = TextPreprocessor()
preprocessor = [spec_preprocessor, text_preprocessor]

train_dataset = BaseDataset(source=TRAIN_DATASET_PATH, preprocessor=preprocessor, sort_by=0)
val_dataset = BaseDataset(source=VAL_DATASET_PATH, preprocessor=preprocessor, sort_by=0)

train_sampler = UniformBatchSampler(len(train_dataset), TRAIN_UPDATES, BATCH_SIZE)

train_loader = DataLoader(train_dataset, num_workers=4, pin_memory=True, collate_fn=train_dataset.collate,
                          batch_sampler=train_sampler)
val_loader = DataLoader(val_dataset, num_workers=4, pin_memory=True, collate_fn=val_dataset.collate,
                        batch_size=BATCH_SIZE)

asr_model = ASRModel().cuda()  # For CPU: remove .cuda()

runner = Runner(
    model=asr_model,
    name=NAME,
    train_loader=train_loader,
    val_loader=val_loader,
    stat_path=RESULTS_PATH,
    models_path=MODELS_OUTPUT_PATH,
    validate_every=2000
)

runner.train()