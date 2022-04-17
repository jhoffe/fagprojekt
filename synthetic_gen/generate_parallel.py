import ray.data
import torch
import torchaudio
import warnings
import os
import pandas as pd
from filelock import FileLock

LS_DATASET_TYPE = os.getenv('LS_DATASET_TYPE')
HPC_PATH = os.getenv("HPC_PATH") if "HPC_PATH" in os.environ else os.getcwd()
TORCH_MODELS_PATH = "{}/data/models".format(HPC_PATH)
LS_PATH = "{}/data/librispeech".format(HPC_PATH)
OUTPUT_PATH = "{}/data/synthetic_speech/{}".format(HPC_PATH, LS_DATASET_TYPE)
RATE = 22050

NUM_GPUS = torch.cuda.device_count()


class BatchInferModel:
    def __init__(self):
        # warnings.simplefilter("ignore")

        self.waveglow = self._init_waveglow()
        self.tacotron = self._init_tacotron()
        self.utils = self._init_utils()

    def _init_tacotron(self):
        lock = FileLock("tacotron2.lock")

        with lock:
            tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
            tacotron2 = tacotron2.to('cuda')
            tacotron2.decoder.max_decoder_steps = 10000
            tacotron2.eval()

        return tacotron2

    def _init_waveglow(self):
        lock = FileLock("waveglow.lock")
        with lock:
            waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
            waveglow = waveglow.remove_weightnorm(waveglow)
            waveglow = waveglow.to('cuda')
            waveglow.eval()

        return waveglow

    def _init_utils(self):
        lock = FileLock("utils.lock")
        with lock:
            utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

        return utils

    @torch.no_grad()
    def __call__(self, batch: pd.DataFrame):
        transcripts = batch["transcript"].tolist()
        sequences, lengths = self.utils.prepare_input_sequence(transcripts)

        mels, _, _ = self.tacotron.infer(sequences, lengths)
        waveforms = self.waveglow.infer(mels)

        new_batch = []

        for (waveform, utterance_id, transcript) in zip(waveforms.cpu(), batch["utterance_id"].tolist(), transcripts):
            new_batch.append((waveform, f"u{utterance_id}.wav", transcript, f"u{utterance_id}.txt"))

        return new_batch


def tuple_to_dict(tuple):
    return {
        "transcript": tuple[2],
        "utterance_id": tuple[5]
    }


class BatchWriteModel:
    def __init__(self):
        # warnings.simplefilter("ignore")
        self.test = None

    def __call__(self, batch):
        for item in batch:
            filepath = "{}/{}".format(OUTPUT_PATH, item[1])
            txt_filepath = "{}/{}".format(OUTPUT_PATH, item[3])

            torchaudio.save(filepath=filepath, src=item[0].reshape((1, -1)), sample_rate=RATE)

            f = open(txt_filepath, "w")
            f.write(item[2])
            f.close()

        return batch


if __name__ == '__main__':
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

    train_clean_100 = torchaudio.datasets.LIBRISPEECH(root=LS_PATH, url=LS_DATASET_TYPE, download=True)
    train_clean_100 = list(map(tuple_to_dict, train_clean_100))

    ds = ray.data.from_items(train_clean_100)
    ds = ds.map_batches(
        BatchInferModel, compute=ray.data.ActorPoolStrategy(NUM_GPUS, NUM_GPUS),
        batch_size=16, num_gpus=1
    )
    ds = ds.map_batches(BatchWriteModel, compute=ray.data.ActorPoolStrategy(2, 2),
                        batch_size=16, num_gpus=0)
