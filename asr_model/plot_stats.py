import pandas as pd
import matplotlib.pyplot as plt
import os

'''
Skal måske slettes?
'''

N = 2000

sets = {
    'Authentic': {
        'train': 'authentic_asr_train_results.csv',
        'val': 'authentic_asr_val_results.csv',
    },
    'Mixed': {
        'train': 'mixed_asr_train_results.csv',
        'val': 'mixed_asr_val_results.csv',
    },
    'Synthetic': {
        'train': 'synthetic_asr_train_results.csv',
        'val': 'synthetic_asr_val_results.csv',
    },
}

metrics = ['wer', "cer"]

for metric in metrics:
    fig, axs = plt.subplots(3, 1, sharex='all')

    for i, (name, files) in enumerate(sets.items()):
        data_train = pd.read_csv(os.path.join('asr_model/results/', files['train']))
        data_val = pd.read_csv(os.path.join('asr_model/results/', files['val']))

        index = data_train["batch_number"]

        data_train[f"windowed_{metric}"] = data_train.rolling(window=N)[f"current_{metric}"].mean()

        data_train.dropna(inplace=True)

        best_train = data_train[f"windowed_{metric}"].min()
        best_val = data_val[f"running_{metric}"].min()

        axs[i].plot(data_train["batch_number"], data_train[f"windowed_{metric}"], '-', label="Training")
        axs[i].plot(data_val["batch_number"], data_val[f"running_{metric}"], 'r+', label="Validation")
        axs[i].set_title(f"{name} (best train = {best_train:.2f}, best test = {best_val:.2f})")
        axs[i].legend()

    fig.suptitle(f"Rolling {metric.upper()} for (N={N})")
    fig.tight_layout()
    plt.savefig(os.path.join('asr_model/results/', f"{metric}_plot.png"))
