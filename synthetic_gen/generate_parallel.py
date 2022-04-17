import ray.data
import torch
import torchaudio
import warnings
import os
import pandas as pd


LS_DATASET_TYPE = os.getenv('LS_DATASET_TYPE')
HPC_PATH = os.getenv("HPC_PATH") if "HPC_PATH" in os.environ else os.getcwd()
TORCH_MODELS_PATH = "{}/data/models".format(os.getcwd())
LS_PATH = "{}/data/librispeech".format(os.getcwd())
OUTPUT_PATH = "{}/data/synthetic_speech/{}".format(os.getcwd(), LS_DATASET_TYPE)
RATE = 22050

NUM_GPUS = torch.cuda.device_count()

class BatchInferModel:
    def __init__(self):
        #warnings.simplefilter("ignore")

        self.waveglow = self._init_waveglow()
        self.tacotron = self._init_tacotron()
        self.utils = self._init_utils()

    def _init_tacotron(self):
        tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
        tacotron2 = tacotron2.to('cuda')
        tacotron2.decoder.max_decoder_steps = 10000
        tacotron2.eval()

        return tacotron2

    def _init_waveglow(self):
        waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow = waveglow.to('cuda')
        waveglow.eval()

        return waveglow

    def _init_utils(self):
        return torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

    @torch.no_grad()
    def __call__(self, batch: pd.DataFrame):
        transcripts = batch["transcript"].tolist()
        sequences, lengths = self.utils.prepare_input_sequence(transcripts)

        mels, _, _ = self.tacotron.infer(sequences, lengths)
        waveforms = self.waveglow.infer(mels)

        batch["synthetic_waveform"] = waveforms.cpu()
        batch["audio_filename"] = list(
            map(lambda utterance_id: f"u{utterance_id}.wav", batch["utterance_id"].tolist())
        )
        batch["txt_filename"] = list(
            map(lambda utterance_id: f"u{utterance_id}.txt", batch["utterance_id"].tolist())
        )

        return batch


def tuple_to_dict(tuple):
    return {
        "transcript": tuple[2],
        "utterance_id": tuple[5]
    }

class BatchWriteModel:
    def __init__(self):
        #warnings.simplefilter("ignore")
        self.test = None

    def __call__(self, batch: pd.DataFrame):
        for index, row in batch.iterrows():
            filepath = "{}/{}".format(OUTPUT_PATH, row["audio_filename"])
            txt_filepath = "{}/{}".format(OUTPUT_PATH, row["txt_filename"])

            torchaudio.save(filepath=filepath, src=row["synthetic_waveform"], sample_rate=RATE)

            f = open(txt_filepath, "w")
            f.write(row["transcript"])
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
    ds = ds.map_batches(BatchWriteModel, compute=ray.data.ActorPoolStrategy(2, 16),
        batch_size=16, num_gpus=0)



