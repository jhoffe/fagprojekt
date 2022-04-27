import torch
import numpy as np
import librosa
import torchaudio
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, FrequencyMask

from asr.utils.text import LIBRISPEECH_CTC_ALPHABET, clean_librispeech

class SpectrogramPreprocessor():

    def __init__(self, ext='.flac', sample_rate=22050, window_size=0.02, stride=0.01, power_spectrum=True, num_mels=80,
                 logscale=True, normalize=True, frq_bin=True, output_format='NFHT', should_augment=False):
        """
        Converts PCM-based files to spectrograms.

        Args:
            ext (string): File extension of source files.
            sample_rate (int): Sample rate of the source files. An error is thrown if a source file doesn't match.
            window_size (float): Size of the STFT window in seconds.
            stride (float): Hopsize of the STFT in seconds.
            power_spectrum (bool): If true, the absolute value of the STFT is squared.
            num_mels (int): Number of mel-filters. If None, no mel-scaling is done.
            logscale (bool): Log scales with librosas power_to_db.
            normalize (bool): Normalizes the spectogram to mean=0 and variance=1.
            frq_bin (bool): Normalizes across each frequency bin instead of the whole spectrogram.
            output_format (string): One of 'NFHT', 'NFT' or 'TNF'.
        """

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

    def augment(self, sample):
        augmenter = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=1),
            TimeStretch(min_rate=0.9, max_rate=1.5, p=0.9),
            PitchShift(min_semitones=-12, max_semitones=12, p=0.98),
            FrequencyMask(p=1)
        ], p=0.95)

        return augmenter(samples=sample, sample_rate=self.sample_rate)

    def __call__(self, path):
        """
        Loads a PCM-based audio file and transforms the PCM signal to a spectrogram.

        Args:
            path (string): Path to source file.
        
        Returns:
            np.ndarray: A spectrogram of shape (F/H)T.
        """

        if not path.endswith(self.ext):
            path = path + self.ext

        pcm, sample_rate = torchaudio.load(path, format="flac") #wavfile.read(path)
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
        
        return spectrogram.astype(np.float32)
    
    def get_seq_len(self, x):
        """
        Get sequence length of example as returned by __call__
        """
        return x.shape[1]

    def collate(self, batch):
        """
        Collates the batch by padding all examples up to the longest sequence length.

        Args:
            batch (list): List of examples as defined by __call__.
        
        Returns:
            np.ndarray: Batch of spectrograms shaped according to output format.
            np.ndarray: Sequence lengths of size N.
        """

        T_max = max([s.shape[1] for s in batch])
        N, F = len(batch), batch[0].shape[0]
        padded_batch = np.zeros((N, F, T_max), dtype=np.float32)
        seq_lens = []
        for i, s in enumerate(batch):
            T = s.shape[1]
            seq_lens.append(T)
            padded_batch[i, :, :T] = s

        if self.output_format == 'TNF':
            padded_batch = np.transpose(padded_batch, [2, 0, 1])
        elif self.output_format == 'NFHT':
            padded_batch = padded_batch[:, np.newaxis]

        return [padded_batch, np.array(seq_lens, dtype=np.int32)]

class TextPreprocessor():

    def __init__(self, ext='.txt', alphabet=LIBRISPEECH_CTC_ALPHABET, clean_func=clean_librispeech, pad_value=-1,
                 tensor_batch=True):
        """
        Processes text data.

        Args:
            ext (string): File extension of source files.
            alphabet (list): List of strings corresponding to the characters used for encoding the text.
            clean_func (func): A function to process to raw text in order to be encoded without error.
            pad_value (int): The value used for padding the text. Default is -1.
            tensor_batch (bool): If True, batches will be returned as torch.Tensor instead of numpy.ndarray.
        """

        self.ext = ext
        self.alphabet = alphabet
        self.clean_func = clean_func
        self.pad_value = pad_value
        self.output_format = 'NT'
        self.tensor_batch = tensor_batch

    def __call__(self, path):
        """
        Loads a text file, cleans it and encodes it.

        Args:
            path (string): Path to source file.
        
        Returns:
            list: Entries are integers corresponding to characters in the alphabet.
        """

        if not path.endswith(self.ext):
            path = path + self.ext

        with open(path, 'r') as text_file:
            transcript_raw = self.clean_func(text_file.read())
            transcript = [self.alphabet.index(c) for c in transcript_raw]
        return transcript
    
    def get_seq_len(self, x):
        """
        Get sequence length of example as returned by __call__
        """
        return len(x)
    
    def decode(self, encoded_text):
        """
        Decodes into readable text.

        Args:
            encoded_text (list): Integers should correspond to the entries of the alphabet.
        
        Returns:
            string: The decoded text.
        """
        return ''.join([self.alphabet[i] for i in encoded_text])
    
    def decode_batch(self, encoded_batch, seq_lens=None):
        """
        Decodes a batch of encoded text samples into readable text.

        Args:
            encoded_text (list or np.ndarray): Encoded text batch. Each row should correspond to an example.
            seg_lens (list or np.ndarray): Sequence lengths of the examples in the batch.
        
        Returns:
            list of strings: The decoded text batch.
        """
        
        if isinstance(encoded_batch, torch.Tensor):
            encoded_batch = encoded_batch.detach().cpu().numpy()
        if seq_lens is None:
            seq_lens = [len(l) for l in encoded_batch]
        else:
            if isinstance(seq_lens, torch.Tensor):
                seq_lens = seq_lens.detach().cpu().numpy()

        decoded_batch = []
        for i, j in enumerate(seq_lens):
            decoded_text = self.decode(encoded_batch[i][:j])
            decoded_batch.append(decoded_text)
        return decoded_batch

    def collate(self, batch):
        """
        Collates the batch by padding all examples up to the longest sequence length.

        Args:
            batch (list): List of examples as defined by __call__.
        
        Returns:
            np.ndarray: Batch of padded text examples.
            np.ndarray: Sequence lengths of size N.
        """
        T_max = max([len(t) for t in batch])
        padded_batch, seq_lens = [], []
        for t in batch:
            T = len(t)
            seq_lens.append(T)
            padded_batch.append(t + [self.pad_value] * (T_max - T))
        
        return [np.array(padded_batch), np.array(seq_lens)]