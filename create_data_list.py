import os
import torchaudio

datasets = os.listdir('./data/synthetic_speech/')

LIMIT = 18

for dataset in datasets:
    files = os.listdir("./data/synthetic_speech/{}/".format(dataset))

    full_paths = list(
        set(["data/synthetic_speech/{}/".format(dataset) + filename.replace('.wav', '').replace('.txt', '') for filename
             in files]))
    filtered_full_paths = []

    for path in full_paths:
        with open(path + ".wav") as f:
            metadata = torchaudio.info(f.read())

            if metadata.num_frames/metadata.sample_rate < LIMIT:
                filtered_full_paths.append(path)

    files_string = "\n".join(filtered_full_paths)

    f = open("asr/data/librispeech/{}.txt".format(dataset), "w+")
    f.write(files_string)
    f.close()
