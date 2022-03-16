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

# Create the models
tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()

waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()

librispeech = torchaudio.datasets.LIBRISPEECH(root=LS_SPEECH_PATH, url="dev-clean", download=True)

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')


@torch.no_grad()
def generate_audio_sample(data):
    text = data[2]
    sequences, lengths = utils.prepare_input_sequence([text])
    mel, _, _ = tacotron2.infer(sequences, lengths)
    return waveglow.infer(mel)


generator_bar = tqdm(librispeech, desc="Generating wav files")
for sentence in generator_bar:
    filename = "uid{}_s{}_c{}.wav".format(sentence[5], sentence[3], sentence[4])
    filepath = "{}/{}".format(OUTPUT_PATH, filename)

    # If file has already been generated, then there is no reason to generate again
    if not os.path.exists(filepath):
        audio = generate_audio_sample(sentence)
        torchaudio.save(filepath=filepath, src=audio.cpu(), sample_rate=RATE)
