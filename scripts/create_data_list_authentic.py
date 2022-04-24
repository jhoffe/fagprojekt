import os
import torchaudio
from tqdm import tqdm
from collections import defaultdict

"""
The aim of this file is to create lists of the files to be used in the ASR model.
This is necessary for the ASR model to work, and in the process, we filter out some 
of the longer audio clips. This is done solely for the reason that our generated
clips become distorted at this point. As such, we only keep the authentic clips that
have corresponding synthetic, unfiltered counterparts.
"""

datasets = os.listdir('data/authentic_speech_upsampled/')

combined = {'authentic-train': ['train-clean-100', 'train-clean-360', 'train-other-500']}

combined_datasets = defaultdict(lambda: "")

LIMIT = 18
SAMPLE_RATE = 22050

for dataset in datasets:
    files = os.listdir("./data/authentic_speech_upsampled/{}/".format(dataset))

    full_paths = list(
        set(["data/authentic_speech_upsampled/{}/".format(dataset) + filename.replace('.flac', '').replace('.txt', '') for filename
             in files]))

    files_string = "\n".join(full_paths)

    for cds_name, cds in combined.items():
        if dataset in cds:
            combined_datasets[cds_name] += "\n" + files_string if combined_datasets[cds_name] != "" else files_string

    f = open("asr_model/data/librispeech/authentic-{}.txt".format(dataset), "w+")
    f.write(files_string)
    f.close()

for cds_name, cds in combined_datasets.items():
    f = open(f"asr_model/data/librispeech/{cds_name}.txt", "w+")
    f.write(cds)
    f.close()
