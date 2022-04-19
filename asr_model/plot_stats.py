import pandas as pd
import matplotlib.pyplot as plt

MODEL = "authentic"

data_train = pd.read_csv("stats_train_{}.csv".format(MODEL), sep=",")
data_test = pd.read_csv("stats_val_{}.csv".format(MODEL), sep=",")

epoch = data_train["epoch"]

ctc_train = data_train["ctc"]
wer_train = data_train["wer"]
cer_train = data_train["cer"]

ctc_test = data_test["ctc"]
wer_test = data_test["wer"]
cer_test = data_test["cer"]

# ctc plot
plt.plot(epoch, ctc_train, label="Train")
plt.plot(epoch, ctc_test, label="Test")
plt.title("CTC loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

# wer plot
plt.plot(epoch, wer_train, label="Train")
plt.plot(epoch, wer_test, label="Test")
plt.title("Word Error Rate (WER)")
plt.xlabel("epoch")
plt.ylabel("WER")
plt.legend()
plt.show()

# cer
plt.plot(epoch, cer_train, label="Train")
plt.plot(epoch, cer_test, label="Test")
plt.title("Character Error Rate (CER)")
plt.xlabel("epoch")
plt.ylabel("CER")
plt.legend()
plt.show()
