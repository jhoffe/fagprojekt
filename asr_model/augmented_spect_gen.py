from multiprocessing import Pool

import numpy as np
import librosa
import torchaudio
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, SpecFrequencyMask, SpecCompose
import os
import torch
import random
from asr.data import BaseDataset
from tqdm import tqdm

'''
This script has the purpose of generating augmented spectrograms from .flac files.
This should be run in advance if one wishes to train using spectrograms.
'''

class AugmentedSpectrogramGenerator:
    def __init__(self, ext='.flac', sample_rate=22050, window_size=0.02, stride=0.01, power_spectrum=True, num_mels=80,
                 logscale=True, normalize=True, frq_bin=True, output_format='NFHT', should_augment=False):

        assert output_format in ('NFHT', 'NFT', 'TNF'), 'Output format should be NFHT, NFT or TNF.'

        self.ext = ext
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.stride = stride
        self.power_spectrum = power_spectrum
        self.num_mels = num_mels
        self.logscale = logscale
        self.normalize = normalize
        self.frq_bin = frq_bin
        self.output_format = output_format
        self.should_augment = should_augment

        self.mel_basis = None if num_mels is None else \
            librosa.filters.mel(sr=sample_rate, n_fft=int(window_size * sample_rate), n_mels=num_mels)

    # The augment(sample) function takes in a sample and applies Gaussian noice, Time Stretch and Pitch Shift
    def augment(self, sample):
        augmenter = Compose([
            AddGaussianNoise(min_amplitude=0.0005, max_amplitude=0.001, p=1),
            TimeStretch(min_rate=0.7, max_rate=1.3, p=0.95, leave_length_unchanged=False),
            PitchShift(min_semitones=-8, max_semitones=8, p=0.995)
        ], p=0.995)

        return augmenter(samples=sample, sample_rate=self.sample_rate)

    # The generate(path) function takes in the path of a .flac files and calculates the mel spectrogram
    # for the given .flac file. If should_augment is True, then augmentation takes place. This should only
    # be the case for synthetic files.
    def generate(self, path):
        path = path[-1]
        if not path.endswith(self.ext):
            path = path + self.ext

        pcm, sample_rate = torchaudio.load(path, format="flac")  # wavfile.read(path)
        pcm = pcm.numpy().reshape(-1)

        assert sample_rate == self.sample_rate, f'Audio file did not have the expected sample rate: {path}'
        assert len(pcm) > int(self.window_size * self.sample_rate), f'PCM audio has too few samples: {path}'
        assert np.std(pcm) > 0, f'PCM audio is empty: {path}'

        if self.should_augment:
            pcm = self.augment(pcm)

        stft = librosa.stft(pcm.astype(np.float64), n_fft=int(self.window_size * self.sample_rate), window='hann',
                            hop_length=int(self.stride * sample_rate), dtype=np.complex128)
        spectrogram = np.abs(stft)

        if self.power_spectrum:
            spectrogram = spectrogram ** 2.
        if self.num_mels is not None:
            spectrogram = np.dot(self.mel_basis, spectrogram)
        if self.logscale:
            spectrogram = librosa.core.power_to_db(spectrogram, top_db=None)
        if self.normalize:
            axis = 1 if self.frq_bin else None
            mean, std = np.mean(spectrogram, axis=axis), np.std(spectrogram, axis=axis)
            spectrogram = ((spectrogram.T - mean) / (std + np.finfo(np.float64).eps)).T

        spect = spectrogram.astype(np.float32)

        if self.should_augment:
            spec_freq_mask = SpecCompose([SpecFrequencyMask(min_mask_fraction=0.05, max_mask_fraction=0.20, p=0.98)])
            spect = spec_freq_mask(spect)

        save_path, _ext = os.path.splitext(path)

        np.savez_compressed(f"{save_path}.spect", spect, allow_pickle=True)

        return f"{save_path}.spect"

# The following deployts the above class to generate spectrograms for the given path of the data set.
if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    TRAIN_DATASET_PATH = os.environ['TRAIN_DATASET']
    CPU_CORES = int(os.environ['CPU_CORES'])
    SHOULD_AUGMENT = int(os.environ["SHOULD_AUGMENT"]) == 1

    train_dataset = BaseDataset(source=TRAIN_DATASET_PATH, preprocessor=[], sort_by=0)

    spect_gen = AugmentedSpectrogramGenerator(should_augment=SHOULD_AUGMENT)

    with tqdm(total=len(train_dataset), desc="Generating spectrograms") as pbar:
        with Pool(int(os.environ["CPU_CORES"])) as p:
            for _ in p.imap(spect_gen.generate, train_dataset, 12):
                pbar.update()
