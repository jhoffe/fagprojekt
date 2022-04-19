import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # For CPU: change '0' to '' (perhaps redundant)

from asr.data import BaseDataset
from asr.data.preprocessors import SpectrogramPreprocessor, TextPreprocessor
from asr.modules import ASRModel
from asr.utils.training import batch_to_tensor, epochs, Logger
from asr.utils.text import greedy_ctc
from asr.utils.metrics import ErrorRateTracker, LossTracker

import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

val_source = os.environ["TEST_DATASET"]
load_path = os.environ["MODEL_PATH"]

spec_preprocessor = SpectrogramPreprocessor(output_format='NFT', sample_rate=22050, ext=".flac")
text_preprocessor = TextPreprocessor()
preprocessor = [spec_preprocessor, text_preprocessor]

val_dataset = BaseDataset(source=val_source, preprocessor=preprocessor, sort_by=0)

val_loader = DataLoader(val_dataset, num_workers=4, pin_memory=True, collate_fn=val_dataset.collate,
                        batch_size=16)

asr_model = ASRModel(dropout=0.05).cuda() # For CPU: remove .cuda()
model_parameters = torch.load(load_path)
asr_model.load_state_dict(model_parameters["model_state_dict"])
ctc_loss = nn.CTCLoss(reduction='sum').cuda() # For CPU: remove .cuda()

wer_metric = ErrorRateTracker(word_based=True)
cer_metric = ErrorRateTracker(word_based=False)
ctc_metric = LossTracker()

val_logger = Logger('Validation', ctc_metric, wer_metric, cer_metric)

def forward_pass(batch):

    (x, x_sl), (y, y_sl) = batch_to_tensor(batch, device='cuda') # For CPU: change 'cuda' to 'cpu'
    logits, output_sl = asr_model.forward(x, x_sl.cpu()) 
    log_probs = F.log_softmax(logits, dim=2)
    loss = ctc_loss(log_probs, y, output_sl, y_sl)

    hyp_encoded_batch = greedy_ctc(logits, output_sl)
    hyp_batch = text_preprocessor.decode_batch(hyp_encoded_batch)
    ref_batch = text_preprocessor.decode_batch(y, y_sl)

    wer_metric.update(ref_batch, hyp_batch)
    cer_metric.update(ref_batch, hyp_batch)
    ctc_metric.update(loss.item(), weight=output_sl.sum().item())

    return loss, wer_metric, cer_metric, ctc_metric


asr_model.eval()
metrics = {}

for batch, files in val_logger(val_loader):
    loss, wer_metric, cer_metric, ctc_metric = forward_pass(batch)
    metrics[str(batch)] = [wer_metric, cer_metric, ctc_metric]

CSVPATH = "{}data/metrics".format(os.getcwd())

if not os.path.exists(CSVPATH):
    os.makedirs(CSVPATH)

metric_file = open("CSVPATH/{}".format("metrics.csv"), "w")
writer = csv.writer(metric_file)

for key, value in metrics.items():
    writer.writerow([key, value[0], value[1], value[2]])

metric_file.close()

