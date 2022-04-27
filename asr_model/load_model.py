import os

from asr.data import BaseDataset
from asr.data.preprocessors import SpectrogramPreprocessor, TextPreprocessor
from asr.modules import ASRModel
from asr.utils.training import batch_to_tensor, Logger
from asr.utils.text import greedy_ctc
from asr.utils.metrics import ErrorRateTracker, LossTracker
from runner import Runner

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

VAL_SOURCE = os.environ["TEST_DATASET"]
MODEL_PATH = os.environ["MODEL_PATH"]
NAME = os.environ["NAME"]

spec_preprocessor = SpectrogramPreprocessor(output_format='NFT', sample_rate=22050, ext=".flac")
text_preprocessor = TextPreprocessor()
preprocessor = [spec_preprocessor, text_preprocessor]

val_dataset = BaseDataset(source=VAL_SOURCE, preprocessor=preprocessor, sort_by=0)

val_loader = DataLoader(val_dataset, num_workers=4, pin_memory=True, collate_fn=val_dataset.collate,
                        batch_size=16)

asr_model = ASRModel(dropout=0.05).cuda() # For CPU: remove .cuda()
model_parameters = torch.load(MODEL_PATH)
asr_model.load_state_dict(model_parameters)
ctc_loss = nn.CTCLoss(reduction='sum').cuda() # For CPU: remove .cuda()

wer_metric = ErrorRateTracker(word_based=True)
cer_metric = ErrorRateTracker(word_based=False)
ctc_metric = LossTracker()

val_logger = Logger('Validation', ctc_metric, wer_metric, cer_metric)

runner = Runner(
    model=asr_model,
    name=NAME,
    val_loader=val_loader
)

runner.validate()