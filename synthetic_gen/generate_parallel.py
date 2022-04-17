import os

import pandas as pd
import ray
import torch
import torchaudio
from filelock import FileLock
import ray

LS_DATASET_TYPE = os.getenv('LS_DATASET_TYPE')
HPC_PATH = os.getenv("HPC_PATH") if "HPC_PATH" in os.environ else os.getcwd()
TORCH_MODELS_PATH = "{}/data/models".format(HPC_PATH)
LS_PATH = "{}/data/librispeech".format(HPC_PATH)
OUTPUT_PATH = "{}/data/synthetic_speech/{}".format(HPC_PATH, LS_DATASET_TYPE)
RATE = 22050

NUM_GPUS = torch.cuda.device_count()

@ray.remote(num_gpus=1)
class InferModel(object):
    def __init__(self, rank: int):
        # warnings.simplefilter("ignore")

        self.rank = rank
        self.waveglow = self._init_waveglow()
        self.tacotron = self._init_tacotron()
        self.utils = self._init_utils()

    def _init_tacotron(self):
        lock = FileLock("tacotron2.lock")

        with lock:
            tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
            tacotron2 = tacotron2.to(torch.device(f"cuda:{self.rank}"))
            tacotron2.decoder.max_decoder_steps = 10000
            tacotron2.eval()

        return tacotron2

    def _init_waveglow(self):
        lock = FileLock("waveglow.lock")
        with lock:
            torch.cuda.empty_cache()
            waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
            waveglow = waveglow.remove_weightnorm(waveglow)
            waveglow = waveglow.to(torch.device(f"cuda:{self.rank}"))
            waveglow.eval()

        return waveglow

    def _init_utils(self):
        lock = FileLock("utils.lock")
        with lock:
            utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

        return utils

    def infer(self, shard: ray.data.Dataset) -> int:
        write_model = BatchWriteModel()

        for batch in shard.iter_batches(batch_size=2):
            proc_batch = self.infer_batch(batch)
            write_model.write(proc_batch)
        return shard.count()

    @torch.no_grad()
    def infer_batch(self, batch: pd.DataFrame):
        transcripts = batch["transcript"].tolist()
        sequences, lengths = self.utils.prepare_input_sequence(transcripts)

        mels, _, _ = self.tacotron.infer(sequences, lengths)
        waveforms = self.waveglow.infer(mels)

        return list(zip(waveforms.cpu(), batch["speaker_id"].tolist(), batch["chapter_id"].tolist(), batch["utterance_id"].tolist()))


def tuple_to_dict(tuple):
    return {
        "transcript": tuple[2],
        "speaker_id": tuple[3],
        "chapter_id": tuple[4],
        "utterance_id": tuple[5]
    }

class BatchWriteModel(object):
    def __init__(self):
        # warnings.simplefilter("ignore")
        self.test = None

    def write(self, batch):
        for sample in batch:
            name = "s{}_c{}_u{}".format(sample[2], sample[3], sample[4])
            filename = f"{name}.flac"
            filepath = "{}/{}".format(OUTPUT_PATH, filename)
            txt_filename = f"{name}.txt"
            txt_filepath = "{}/{}".format(OUTPUT_PATH, txt_filename)

            torchaudio.save(filepath=filepath, src=sample[0].reshape((1, -1)), sample_rate=RATE, format="flac")

            f = open(txt_filepath, "w")
            f.write(sample[2])
            f.close()


if __name__ == '__main__':
    # Create needed directories
    if not os.path.exists(TORCH_MODELS_PATH):
        os.makedirs(TORCH_MODELS_PATH)

    if not os.path.exists(LS_PATH):
        os.makedirs(LS_PATH)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Set another local directory to download the models
    #torch.hub.set_dir(TORCH_MODELS_PATH)



    train_clean_100 = torchaudio.datasets.LIBRISPEECH(root=LS_PATH, url=LS_DATASET_TYPE, download=True)
    train_clean_100 = list(map(tuple_to_dict, train_clean_100))

    ray.init()

    # Create workers
    gpu_workers = [InferModel.remote(i) for i in range(NUM_GPUS)]

    # Create dataset
    ds = ray.data.from_items(train_clean_100)

    # Shard the dataset
    shards = ds.split(n=NUM_GPUS, locality_hints=gpu_workers)

    ray.get([w.infer.remote(shard) for (w, shard) in zip(gpu_workers, shards)])



