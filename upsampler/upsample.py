import os
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

INPUT_RATE = 16000
OUTPUT_RATE = 22050
training_sets = ["train-clean-100", "train-clean-360", "train-other-500"]

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

    data = './data/{}/'.format(training_set) # skal fixes
    datasets = os.listdir(data)

    for dataset in datasets:
        files = os.listdir("./data/synthetic_speech/{}/".format(dataset)) # skal fixes

        for file in tqdm(files, desc="Upsampling files for {}".format(dataset)):
            upsampled = audio_upsample(file, input_rate=INPUT_RATE, output_rate=OUTPUT_RATE)
            filename = "{}.wav".format(os.path.basename(file.name))
            filepath = "{}/{}".format(OUTPUT_PATH, filename)

            if not os.path.exists(filepath):
                torchaudio.save(filepath=filepath, src=upsampled.cpu(), sample_rate=OUTPUT_RATE)








