import os

import torch
import torchaudio
from filelock import FileLock
import argparse
from tqdm import tqdm

RATE = 22050

class InferModel(object):
    def __init__(self, batch_size: int, rank: int, output_path: str):
        # warnings.simplefilter("ignore")

        self.rank = rank
        self.batch_size = batch_size
        self.output_path = output_path
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

    def _iter_batch(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def infer(self, dataset):
        write_model = BatchWriteModel(output_path=self.output_path)

        for batch in tqdm(self._iter_batch(dataset, self.batch_size)):
            proc_batch = self.infer_batch(batch)
            write_model.write(proc_batch)

    @torch.no_grad()
    def infer_batch(self, batch):
        transcripts = get_column(batch, "transcript")
        sequences, lengths = self.utils.prepare_input_sequence(transcripts)

        mels, _, _ = self.tacotron.infer(sequences, lengths)
        waveforms = self.waveglow.infer(mels)

        return list(zip(waveforms.cpu(), transcripts, get_column(batch, "speaker_id"), get_column(batch, "chapter_id"),
                        get_column(batch, "utterance_id")))


class BatchWriteModel(object):
    def __init__(self, output_path: str):
        # warnings.simplefilter("ignore")
        self.output_path = output_path

    def write(self, batch):
        for sample in batch:
            name = "s{}_c{}_u{}".format(sample[2], sample[3], sample[4])
            filename = f"{name}.flac"
            filepath = "{}/{}".format(self.output_path, filename)
            txt_filename = f"{name}.txt"
            txt_filepath = "{}/{}".format(self.output_path, txt_filename)

            torchaudio.save(filepath=filepath, src=sample[0].reshape((1, -1)), sample_rate=RATE, format="flac")

            f = open(txt_filepath, "w")
            f.write(sample[1])
            f.close()

def tuple_to_dict(tuple):
    return {
        "transcript": tuple[2],
        "speaker_id": tuple[3],
        "chapter_id": tuple[4],
        "utterance_id": tuple[5]
    }

def get_column(dict_list, column):
    return [item[column] for item in dict_list]

if __name__ == '__main__':
    LS_PATH = "{}/data/librispeech".format(os.getcwd())

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('--batch_size', type=int)

    args = vars(parser.parse_args())

    OUTPUT_PATH = "{}/data/synthetic_speech/{}".format(os.getcwd(), args["dataset"])

    if not os.path.exists(LS_PATH):
        os.makedirs(LS_PATH)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    lock = FileLock(f"librispeech.{args['dataset']}.lock")
    with lock:
        ds = torchaudio.datasets.LIBRISPEECH(root=LS_PATH, url=args["dataset"], download=True)
    ds = list(map(tuple_to_dict, ds))

    infer_model = InferModel(batch_size=args["batchsize"], rank=0, output_path=OUTPUT_PATH)

    infer_model.infer(ds)

