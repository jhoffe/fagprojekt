import os

datasets = os.listdir('./data/synthetic_speech/')

for dataset in datasets:
    files = os.listdir("./data/synthetic_speech/{}/".format(dataset))
    
    files_string = "\n".join(list(set([filename.replace('.wav', '').replace('.txt', '') for filename in files])))
    
    f = open("asr/data/librispeech/{}.txt".format(dataset), "w+")
    f.write(files_string)
    f.close()
