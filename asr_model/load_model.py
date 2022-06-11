import os

from asr.data import BaseDataset
from asr.data.preprocessors import SpectrogramPreprocessor, TextPreprocessor
from asr.modules import ASRModel
from asr.utils.training import Logger
from asr.utils.metrics import ErrorRateTracker, LossTracker
from runner import Runner

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

VAL_SOURCE = os.environ["TEST_DATASET"]
MODEL_PATH = os.environ["MODEL_PATH"]
NAME = os.environ["NAME"]

'''
This script is used for validating trained models. I.e. a trained model should exist
before this script is used.
'''

# Initialize preprocessors and load data set

spec_preprocessor = SpectrogramPreprocessor(output_format='NFT', sample_rate=22050, ext=".flac")
text_preprocessor = TextPreprocessor()
preprocessor = [spec_preprocessor, text_preprocessor]

val_dataset = BaseDataset(source=VAL_SOURCE, preprocessor=preprocessor, sort_by=0)

val_loader = DataLoader(val_dataset, num_workers=4, pin_memory=True, collate_fn=val_dataset.collate,
                        batch_size=32)

# Initialize model, loss function and tracking metrics

asr_model = ASRModel(dropout=0.05).cuda()  # For CPU: remove .cuda()
model_parameters = torch.load(MODEL_PATH)
asr_model.load_state_dict(model_parameters)
ctc_loss = nn.CTCLoss(reduction='sum').cuda()  # For CPU: remove .cuda()

wer_metric = ErrorRateTracker(word_based=True)
cer_metric = ErrorRateTracker(word_based=False)
ctc_metric = LossTracker()

val_logger = Logger('Validation', ctc_metric, wer_metric, cer_metric)

# Run validation

runner = Runner(
    model=asr_model,
    name=NAME,
    val_loader=val_loader
)

analysis = runner.validate(analysis=True)

# Save results and make plots of relevant data

analysis.preprocess()

save_path = f"asr_model/results/analysis-{NAME}/"

analysis.df.to_csv(os.path.join(save_path, f"{NAME}.csv"))
wer_stopword, wer_wordcounts, stopword_hist, wordcount_hist, wer_hist, cer_hist = analysis.plot()

if not os.path.exists(save_path):
    os.mkdir(save_path)

wer_stopword.savefig(os.path.join(save_path, "wer_stopword.png"))
wer_wordcounts.savefig(os.path.join(save_path, "wer_wordcounts.png"))
stopword_hist.savefig(os.path.join(save_path, "stopword_hist.png"))
wordcount_hist.savefig(os.path.join(save_path, "wordcount_hist.png"))
wer_hist.savefig(os.path.join(save_path, "wer_hist.png"))
cer_hist.savefig(os.path.join(save_path, "cer_hist.png"))
