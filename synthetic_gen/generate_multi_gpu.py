import torch
import torchaudio
import warnings
import os
from tqdm import tqdm
from multiprocessing import Pool

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
def generate_audio_sample(sequences, lengths, waveglow, tacotron2):
    mel, _, _ = tacotron2.infer(sequences, lengths)
    return waveglow.infer(mel)

def generate_audio_samples(dataset, gpu_idx, waveglow, tacotron2):
    waveglow.eval()
    tacotron2.eval()
    text_samples = prepare_text_samples(dataset)

    i = 0
    for text_sample in tqdm(text_samples, desc="Processing samples for gpu: {}".format(gpu_idx)):
        filename = "{}.wav".format(i)
        txt_filename = "{}.txt".format(i)
        filepath = "{}/{}".format(OUTPUT_PATH, filename)
        txt_filepath = "{}/{}".format(OUTPUT_PATH, txt_filename)

        if os.path.exists(filepath) and os.path.exists(txt_filepath):
            continue

        sequences, lengths = utils.prepare_input_sequence([text_sample])

        audio_samples = generate_audio_sample(sequences, lengths, waveglow, tacotron2)

        # If file has already been generated, then there is no reason to generate again
        if not os.path.exists(filepath):
            torchaudio.save(filepath=filepath, src=audio_samples.cpu(), sample_rate=RATE)

        if not os.path.exists(txt_filepath):
            f = open(txt_filepath, "w")
            f.write(text_sample)
            f.close()
        i += 1

if __name__ == '__main__':
    warnings.simplefilter('ignore')

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

    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

    librispeech = torchaudio.datasets.LIBRISPEECH(root=LS_PATH, url=LS_DATASET_TYPE, download=True)

    num_gpu = torch.cuda.device_count()

    gpu_data_len = len(librispeech) // num_gpu
    rest = len(librispeech) % num_gpu

    # Create the models
    tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
    tacotron2.decoder.max_decoder_steps = 10000

    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
    waveglow = waveglow.remove_weightnorm(waveglow)

    with Pool(num_gpu) as pool:
        ps = []

        for gpu_idx in range(num_gpu):
            idx_from = gpu_idx * gpu_data_len
            idx_to = min((gpu_idx + 1) * gpu_data_len + rest, len(librispeech) - 1)
            dataset = [sample for idx, sample in enumerate(librispeech) if idx_from <= idx < idx_to]

            args = (
                dataset,
                gpu_idx,
                waveglow.to("cuda:{}".format(gpu_idx)),
                tacotron2.to("cuda:{}".format(gpu_idx))
            )

            ps.append(pool.apply_async(func=generate_audio_samples, args=args))

        print(ps)

        [p.wait() for p in ps]
        print([p.get() for p in ps])
