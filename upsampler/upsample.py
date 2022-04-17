import os
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

"""
The aim of this file is to resample our authentic audio from 16000 Hz to 22050 Hz such that
the sample rate matches the one from WaveGlow. torchaudio.transforms.Resample is used for
this task using te default parameters from torchaudio.
"""

INPUT_RATE = 16000
OUTPUT_RATE = 22050
training_sets = ["test-clean"] #["train-clean-100", "train-clean-360", "train-other-500"]

LS_PATH = "{}/data/librispeech".format(os.getcwd()) # skal måske fixes

if not os.path.exists(LS_PATH):
    os.makedirs(LS_PATH)

def audio_upsample(audio, input_rate, output_rate):
    upsample = T.Resample(input_rate, output_rate)
    return upsample(audio)

for training_set in training_sets:
    OUTPUT_PATH = "{}/data/authentic_speech_upsampled/{}".format(os.getcwd(), training_set)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    dataset = torchaudio.datasets.LIBRISPEECH(root=LS_PATH, url=training_set, download=True)

    for sample in tqdm(dataset, desc="Upsampling files for {}".format(training_set)):
        upsampled = audio_upsample(sample[0], input_rate=INPUT_RATE, output_rate=OUTPUT_RATE)

        filename = "s{}_c{}_u{}.wav".format(sample[3], sample[4], sample[5])
        filepath = "{}/{}".format(OUTPUT_PATH, filename)

        if not os.path.exists(filepath):
            torchaudio.save(filepath=filepath, src=upsampled, sample_rate=OUTPUT_RATE)








