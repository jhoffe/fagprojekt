import os

datasets = os.listdir('./data/synthetic_speech/')

for dataset in datasets:
    files = os.listdir("./data/synthetic_speech/{}/".format(dataset))

    full_paths = list(set(["data/synthetic_speech/{}/".format(dataset) + filename.replace('.wav', '').replace('.txt', '') for filename in files]))
    files_string = "\n".join(full_paths)
    alphabet = []

    for path in full_paths:
        f = open(path + ".txt", "r")
        words = f.read().lower().split(" ")
        alphabet += words

    alphabet = "\n".join(list(set(alphabet)))
    f = open("asr/data/librispeech/alphabet_{}.txt".format(dataset), "w+")
    f.write(alphabet)
    f.close()

    f = open("asr/data/librispeech/{}.txt".format(dataset), "w+")
    f.write(files_string)
    f.close()
