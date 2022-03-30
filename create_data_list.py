import os
import torchaudio
from tqdm import tqdm

datasets = os.listdir('./data/synthetic_speech/')

LIMIT = 18
SAMPLE_RATE = 22050

for dataset in datasets:
    files = os.listdir("./data/synthetic_speech/{}/".format(dataset))

    full_paths = list(
        set(["data/synthetic_speech/{}/".format(dataset) + filename.replace('.wav', '').replace('.txt', '') for filename
             in files]))
    filtered_full_paths = []

    for path in tqdm(full_paths):
        data = torchaudio.load(filepath=path + ".wav")

        num_frames = data[0].size()[0]

        if num_frames/SAMPLE_RATE < LIMIT:
            filtered_full_paths.append(path)

    files_string = "\n".join(filtered_full_paths)

    f = open("asr/data/librispeech/{}.txt".format(dataset), "w+")
    f.write(files_string)
    f.close()
