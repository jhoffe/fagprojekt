import torch
import torchaudio
import warnings
import os
from tqdm import tqdm

warnings.simplefilter('ignore')

TORCH_MODELS_PATH = "{}/data/models".format(os.getcwd())
LS_SPEECH_PATH = "{}/data/librispeech".format(os.getcwd())
OUTPUT_PATH = "{}/data/synthetic_speech".format(os.getcwd())
RATE = 22050

# Create needed directories
if not os.path.exists(TORCH_MODELS_PATH):
    os.makedirs(TORCH_MODELS_PATH)

if not os.path.exists(LS_SPEECH_PATH):
    os.makedirs(LS_SPEECH_PATH)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Set another local directory to download the models
torch.hub.set_dir(TORCH_MODELS_PATH)
torch.cuda.empty_cache()

# Create the models
tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
tacotron2 = tacotron2.to('cuda')
tacotron2.decoder.max_decoder_steps = 2000
tacotron2.eval()

waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()

librispeech = torchaudio.datasets.LIBRISPEECH(root=LS_SPEECH_PATH, url="dev-clean", download=True)

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

@torch.no_grad()
def split_to_sentence_length(sentence, max_sentence_length=10):
    words = sentence.split()

    split_sentences = []

    for i in range(len(words) // max_sentence_length):
        split_sentences.append(" ".join(words[i * max_sentence_length:i * max_sentence_length + 10]))

    return split_sentences


@torch.no_grad()
def prepare_text_samples(data, max_sentence_length=10):
    text_samples = []

    for sample in data:
        text = sample[2]

        ss = split_to_sentence_length(text, max_sentence_length)

        for s in ss:
            text_samples.append(s)

    return *utils.prepare_input_sequence(text_samples), text_samples

    
    # mel, _, _ = tacotron2.infer(sequences, lengths)
    # return waveglow.infer(mel)


CHUNK_SIZE = 100

for chunk_idx in tqdm(range(len(librispeech) // CHUNK_SIZE), desc="Processing chunks"):
    from_idx = chunk_idx * CHUNK_SIZE
    to_idx = (chunk_idx + 1) * CHUNK_SIZE
    chunk = [sample for idx, sample in enumerate(librispeech) if from_idx <= idx < to_idx]
    sequences, lengths, strings = prepare_text_samples(chunk)
    mel, _, _ = tacotron2.infer(sequences[:CHUNK_SIZE], lengths[:CHUNK_SIZE])
    audio_samples = waveglow.infer(mel)

    for idx, audio_sample in enumerate(audio_samples):
        f_idx = chunk_idx * CHUNK_SIZE + idx
        filename = "{}.wav".format(f_idx)
        txt_filename = "{}.txt".format(f_idx)
        filepath = "{}/{}".format(OUTPUT_PATH, filename)
        txt_filepath = "{}/{}".format(OUTPUT_PATH, filename)

        # If file has already been generated, then there is no reason to generate again
        if not os.path.exists(filepath):
            torchaudio.save(filepath=filepath, src=audio_sample.cpu(), sample_rate=RATE)

        if not os.path.exists(filepath):
            f = open(filepath, "w")
            f.write(strings[idx])
            f.close()
