import os
import torchaudio
from tqdm import tqdm

"""
The aim of this file is to create lists of the files to be used in the ASR model.
This is necessary for the ASR model to work, and in the process, we filter out some 
of the longer audio clips. This is done solely for the reason that our generated
clips become distorted at this point. As such, we only keep the authentic clips that
have corresponding synthetic, unfiltered counterparts.
"""

datasets = os.listdir('./data/authentic_speech_upsampled/')

LIMIT = 18
SAMPLE_RATE = 22050

for dataset in datasets:
    files = os.listdir("./data/authentic_speech_upsampled/{}/".format(dataset))

    full_paths = list(
        set(["data/authentic_speech_upsampled/{}/".format(dataset) + filename.replace('.flac', '').replace('.txt', '') for filename
             in files]))
    filtered_full_paths = []

    for path in tqdm(full_paths):
        data = torchaudio.load(filepath=path + ".flac")

        num_frames = data[0].size()[0]

        if num_frames/SAMPLE_RATE < LIMIT:
            filtered_full_paths.append(path)

    files_string = "\n".join(filtered_full_paths)

    f = open("asr/data/librispeech/{}.txt".format(dataset), "w+")
    f.write(files_string)
    f.close()
