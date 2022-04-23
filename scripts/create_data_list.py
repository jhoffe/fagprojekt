import os
import torchaudio
from tqdm import tqdm
from multiprocessing import Pool

"""
The aim of this file is to create lists of the files to be used in the ASR model.
This is necessary for the ASR model to work, and in the process, we filter out some 
of the longer audio clips. This is done solely for the reason that our generated
clips become distorted at this point. As such, we only keep the authentic clips that
have corresponding synthetic, unfiltered counterparts.
"""

datasets = os.listdir('data/synthetic_speech/')

LIMIT = 18
SAMPLE_RATE = 22050
WINDOW_SIZE = 0.02

def check_length(path):
    data = torchaudio.load(filepath=path + ".flac", format="flac")

    num_frames = data[0].size()[1]

    if num_frames / SAMPLE_RATE < LIMIT and num_frames > int(SAMPLE_RATE * WINDOW_SIZE):
        return path

    print(f"Filtered out: {path}")
    return None

for dataset in datasets:
    files = os.listdir("data/synthetic_speech/{}/".format(dataset))

    full_paths = list(
        set(["data/synthetic_speech/{}/".format(dataset) + filename.replace('.flac', '').replace('.txt', '') for filename
             in files]))
    with Pool(processes=int(os.environ["CPU_CORES"])) as p:
        processed_paths = p.map(check_length, full_paths)

    all_paths = []

    for p in processed_paths:
        if p is not None:
            all_paths.append(p)

    files_string = "\n".join(all_paths)

    f = open("asr_model/data/librispeech/synthetic_{}.txt".format(dataset), "w+")
    f.write(files_string)
    f.close()
