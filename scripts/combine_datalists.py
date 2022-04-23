import os

DATALISTS_PATH = "asr_model/data/librispeech"

combined = {'synthetic-train': ['synthetic_train-clean-100', 'synthetic_train-clean-360', 'synthetic_train-other-500']}

for combination, sets in combined.values():
    combined_set = ""

    for datalist in sets:
        f = open(os.path.join(DATALISTS_PATH, f"{datalist}.txt"), "w+")
        combined_set += "\n" + f.read() if combined_set != "" else f.read()
        f.close()

    f = open(os.path.join(DATALISTS_PATH, f"{combination}.txt"), "w+")
    f.write(combined_set)
    f.close()