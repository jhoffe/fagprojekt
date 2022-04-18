import os

from asr.data import BaseDataset
from asr.data.preprocessors import SpectrogramPreprocessor, TextPreprocessor
from asr.modules import ASRModel
from asr.utils.training import batch_to_tensor, epochs, Logger
from asr.utils.text import greedy_ctc
from asr.utils.metrics import ErrorRateTracker, LossTracker
from tqdm import tqdm
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

TRAIN_DATASET = os.environ['TRAIN_DATASET']
TEST_DATASET = os.environ['TEST_DATASET']
PATH = "model_authentic.pt"

train_source = TRAIN_DATASET
val_source = TEST_DATASET

spec_preprocessor = SpectrogramPreprocessor(output_format='NFT', sample_rate=22050, ext=".flac")
text_preprocessor = TextPreprocessor()
preprocessor = [spec_preprocessor, text_preprocessor]

train_dataset = BaseDataset(source=train_source, preprocessor=preprocessor, sort_by=0)
val_dataset = BaseDataset(source=val_source, preprocessor=preprocessor, sort_by=0)

train_loader = DataLoader(train_dataset, num_workers=4, pin_memory=True, collate_fn=train_dataset.collate,
                          batch_size=16)
val_loader = DataLoader(val_dataset, num_workers=4, pin_memory=True, collate_fn=val_dataset.collate,
                        batch_size=16)

asr_model = nn.DataParallel(ASRModel(), device_ids=list(range(torch.cuda.device_count()))) # For CPU: remove .cuda()
ctc_loss = nn.CTCLoss(reduction='sum').cuda() # For CPU: remove .cuda()
optimizer = torch.optim.Adam(asr_model.parameters(), lr=3e-4)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=5e-5)

wer_metric = ErrorRateTracker(word_based=True)
cer_metric = ErrorRateTracker(word_based=False)
ctc_metric = LossTracker()

train_logger = Logger('Training', ctc_metric, wer_metric, cer_metric)
val_logger = Logger('Validation', ctc_metric, wer_metric, cer_metric)
test_logger = Logger('Test', ctc_metric, wer_metric, cer_metric)

stats_train = []
stats_val = []

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

    return loss

best_wer = np.inf
for epoch in epochs(int(os.environ["EPOCHS"])):
    asr_model.train()
    for batch, files in train_logger(train_loader):
        loss = forward_pass(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    asr_model.eval()
    stats_train.append([epoch] + [m.running for m in train_logger.metric_trackers])
    for batch, files in val_logger(val_loader):
        forward_pass(batch)
    stats_val.append([epoch] + [m.running for m in val_logger.metric_trackers])
    
    if wer_metric.running < best_wer:
        best_wer = wer_metric.running
        torch.save({
            'epoch': epoch,
            'model_state_dict': asr_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, PATH)
        # ... and save model
    
    if epoch >= 100:
        lr_scheduler.step()

data_train = pd.DataFrame(stats_train, columns=["epoch", "ctc", "wer", "cer"])
data_val = pd.DataFrame(stats_val, columns=["epoch", "ctc", "wer", "cer"])
data_train.to_csv("asr_train.csv")
data_val.to_csv("asr_test.csv")