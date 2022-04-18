## Simple PyTorch-based ASR

Install python version and packages listed in the requirements.txt.

Run experiment.py to start training.

The example assumes GPU usage, but comments are made for changes required to run on CPU (not recommended - very slow - for testing only).

Example data is included in data folder. Will probably take a few minute to download.

In the example (experiment.py), the LibriSpeech val-clean portion is used for training and LibriSpeech test-clean for validation.

For a more appropriate data setup, see the original LibriSpeech dataset (500h, 360h and 100h subsets) or LibriLight (10h, 1h, and 10min subsets).

LibriSpeech: https://www.openslr.org/12
LibriLight: https://github.com/facebookresearch/libri-light

Data do not necessarily come in the format required for using with ASR model, so processing might be needed.

Data should organized in pairs of text (.txt) and audio (.wav) files in the same folder.

Depending on the .txt file, it might be necessary to write a new function for cleaning text. Currently, only `clean_librispeech` is available.

A .txt-file specifying the file paths (without file extension) belonging to a given subset should be created. See the data folder for reference.

A pre-trained model is included (`model.pt`) and can be tested with load_model.py. The test should yield the following result:

```
Validation [169/169, 0.4 min(s)]: Loss=0.056, WER=17.24, CER=5.47
```

The model is trained for 10 epochs on the full 960 hour LibriSpeech.
