import torch
import torchaudio
import warnings
import os
from tqdm import tqdm

"""
The aim of this file is to generate audio with our text-to-speech pipeline. The models in use
are TacoTron2 and WaveGlow, and the generated audio clips are stored alongside the text files 
used to generate the audio. 
"""

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

LS_DATASET_TYPE = os.getenv('LS_DATASET_TYPE')
TORCH_MODELS_PATH = "{}/data/models".format(os.getcwd())
LS_PATH = "{}/data/librispeech".format(os.getcwd())
OUTPUT_PATH = "{}/data/synthetic_speech/{}".format(os.getcwd(), LS_DATASET_TYPE)
RATE = 22050

# Create needed directories
if not os.path.exists(TORCH_MODELS_PATH):
    os.makedirs(TORCH_MODELS_PATH)

if not os.path.exists(LS_PATH):
    os.makedirs(LS_PATH)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Set another local directory to download the models
torch.hub.set_dir(TORCH_MODELS_PATH)
torch.cuda.empty_cache()


# Create the models
tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
tacotron2 = tacotron2.to('cuda')
tacotron2.decoder.max_decoder_steps = 10000
tacotron2.eval()

waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

@torch.no_grad()
def split_to_sentence_length(sentence, max_sentence_length=1000):
    words = sentence.split()

    split_sentences = []

    if len(words) // max_sentence_length == 0:
        return [sentence]

    for i in range(len(words) // max_sentence_length):
        split_sentences.append(" ".join(words[i * max_sentence_length:i * max_sentence_length + 10]))

    return split_sentences


@torch.no_grad()
def prepare_text_samples(data, max_sentence_length=1000):
    text_samples = []

    for sample in data:
        text = sample[2]

        ss = split_to_sentence_length(text, max_sentence_length)

        for s in ss:
            text_samples.append(s)

    return text_samples

@torch.no_grad()
def generate_audio_sample(sequences, lengths):
    mel, _, _ = tacotron2.infer(sequences, lengths)
    return waveglow.infer(mel)

def generate_audio_samples(dataset):
    i = 0
    for sample in tqdm(dataset, desc="Processing samples"):
        filename = "s{}_c{}_u{}.wav".format(sample[3], sample[4], sample[5])
        txt_filename = "s{}_c{}_u{}.txt".format(sample[3], sample[4], sample[5])
        filepath = "{}/{}".format(OUTPUT_PATH, filename)
        txt_filepath = "{}/{}".format(OUTPUT_PATH, txt_filename)

        # if os.path.exists(filepath) and os.path.exists(txt_filepath):
        #     continue

        sequences, lengths = utils.prepare_input_sequence([sample[2]])

        audio_samples = generate_audio_sample(sequences, lengths)

        # If file has already been generated, then there is no reason to generate again
        if not os.path.exists(filepath):
            torchaudio.save(filepath=filepath, src=audio_samples.cpu(), sample_rate=RATE)

        if not os.path.exists(txt_filepath):
            f = open(txt_filepath, "w")
            f.write(sample[2])
            f.close()
        i += 1

librispeech = torchaudio.datasets.LIBRISPEECH(root=LS_PATH, url=LS_DATASET_TYPE, download=True)

generate_audio_samples(librispeech)

