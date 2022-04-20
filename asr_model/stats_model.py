import os

from asr.data import BaseDataset
from asr.data.preprocessors import SpectrogramPreprocessor, TextPreprocessor
from asr.modules import ASRModel
from asr.utils.training import batch_to_tensor, epochs, Logger
from asr.utils.text import greedy_ctc
from asr.utils.metrics import ErrorRateTracker, LossTracker

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

val_source = os.environ["TEST_DATASET"]
load_path = os.environ["MODEL_PATH"]

spec_preprocessor = SpectrogramPreprocessor(output_format='NFT', sample_rate=22050, ext=".flac")
text_preprocessor = TextPreprocessor()
preprocessor = [spec_preprocessor, text_preprocessor]

val_dataset = BaseDataset(source=val_source, preprocessor=preprocessor, sort_by=0)

val_loader = DataLoader(val_dataset, num_workers=4, pin_memory=True, collate_fn=val_dataset.collate,
                        batch_size=1)

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

    return ref_batch[0], hyp_batch[0], loss, wer_metric, cer_metric, ctc_metric


asr_model.eval()
metrics = []

for batch, files in val_logger(val_loader):
    ref, hyp, loss, wer_metric, cer_metric, ctc_metric = forward_pass(batch)

    metrics.append([files[0], ref, hyp, wer_metric.current, cer_metric.current])

df = pd.DataFrame(data=metrics, columns=["file_path", "ref", "hyp", "wer", "cer"])
df.to_csv("results/test_sample_errors.csv")