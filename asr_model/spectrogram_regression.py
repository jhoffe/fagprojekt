import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from asr.data import BaseDataset
from asr.data.preprocessors import SpectrogramPreprocessor, TextPreprocessor
from asr.modules import ASRModel
from runner import Runner
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import normalize


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

# Run validation

runner = Runner(
    model=asr_model,
    name=NAME,
    val_loader=val_loader
)


def pad_batch_to_size(spect_batch, max_length):
    current_length = spect_batch.shape[2]

    if max_length == current_length:
        return spect_batch

    assert current_length < max_length
    pad_size = max_length - current_length

    return np.pad(spect_batch, pad_width=((0, 0), (0, 0), (0, pad_size)), mode="constant",
                  constant_values=((0, 0), (0, 0), (0, 0)))


X = []
y = []
count = 0
max_spect_length = 0

# Looping over the validation set and creating dataset
for i, (batch, batch_wer) in enumerate(runner.validate(regression=True)):
    (x, x_sl), _ = batch

    if max(x_sl) > max_spect_length:
        max_spect_length = max(x_sl)

    X.append(x)
    y += batch_wer
    count += x.shape[0]

    if i > 5:
        break

feature_count = 80

new_X = np.zeros((count, max_spect_length * feature_count))

for i, batch in tqdm(enumerate(X), desc="Padding batches"):
    padded_batch = pad_batch_to_size(batch, max_spect_length)
    new_X[i * 32:i * 32 + 32, :] = np.reshape(padded_batch, (padded_batch.shape[0], max_spect_length * feature_count))

new_y = np.array(y)

print("Normalizing")
#norm_X = (new_X - new_X.min(axis=0))/(new_X.max(axis=0) - new_X.min(axis=0))

print("Fitting linear regression")
reg = LinearRegression().fit(new_X, new_y)

print("Plotting")
coef = (reg.coef_ - reg.coef_.min())/(reg.coef_.max() - reg.coef_.min())
spect_coef_norm = coef.reshape((feature_count, max_spect_length))

plt.figure(figsize=(30, 16))
sns.heatmap(spect_coef_norm)
plt.savefig(f"spect_heatmap_{NAME}.png")
