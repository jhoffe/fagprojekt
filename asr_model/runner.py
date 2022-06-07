import numpy as np
import pandas as pd
import wandb

from asr.utils.training import batch_to_tensor, Logger
from asr.utils.text import greedy_ctc
from asr.utils.metrics import ErrorRateTracker, LossTracker
from asr.utils.stat_tracker import StatTracker
from asr.data.preprocessors import TextPreprocessor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nltk.corpus import stopwords
import nltk
import os
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords')


class ValidationAnalysis:
    def __init__(self, data=None):
        self.data = data if data is not None else []
        self.df = None

        if data is not None:
            self._create_df()

    def append(self, path, guess, wer, cer):
        speaker_id, chapter_id, utterance_id = [id[1:] for id in path.split('/')[-1].split('_')]

        with open(path + ".txt", "r") as f:
            text = f.read()

        self.data.append([path, speaker_id, chapter_id, utterance_id, text, guess, wer, cer])

        return self

    def append_batch(self, paths, guesses, wers, cers):
        for args in zip(paths, guesses, wers, cers):
            self.append(*args)

        return self

    def _create_df(self):
        self.df = pd.DataFrame(data=self.data,
                               columns=["Path", "SpeakerId", "ChapterId", "UtteranceId", "Text", "Guess", "WER", "CER"])

        return self

    def preprocess(self):
        self._create_df()

        self._count_words()
        self._count_stopwords_for_all()

        return self

    def _count_words(self):
        self.df["WordCount"] = self.df["Text"].apply(lambda t: len(t.split(" ")))

        return self

    @staticmethod
    def _count_stopwords(text):
        words = text.split(" ")

        stopword_count = 0
        sw = stopwords.words('english')

        for word in words:
            if word.lower() in sw:
                stopword_count += 1

        return stopword_count

    def _count_stopwords_for_all(self):
        self.df["StopWordCount"] = self.df["Text"].apply(self._count_stopwords)

        return self

    def plot(self):
        #WER/Stopword
        wer_stopword = plt.figure()
        plt.scatter(self.df["StopWordCount"], self.df["WER"])
        plt.suptitle("WER and stop word count")

        plt.xlabel("stop words")
        plt.ylabel("WER")

        # WER/wordcounts
        wer_wordcounts = plt.figure()
        plt.scatter(self.df["WordCount"], self.df["WER"])
        plt.suptitle("WER and word count")

        plt.xlabel("stop words")
        plt.ylabel("WER")

        # Stopword counts histograms
        stopword_hist = plt.figure()
        sns.histplot(data=self.df["StopWordCount"])

        # word counts histograms
        wordcount_hist = plt.figure()
        sns.histplot(data=self.df["WordCount"])

        # WER histograms
        wer_hist = plt.figure()
        sns.histplot(data=self.df["WER"])

        # CER histograms
        cer_hist = plt.figure()
        sns.histplot(data=self.df["CER"])

        return wer_stopword, wer_wordcounts, stopword_hist, wordcount_hist, wer_hist, cer_hist



class Runner:
    def __init__(self, model, name, train_loader=None, val_loader=None, stat_path="asr_model/results/",
                 models_path=None,
                 validate_every=15000,
                 lr=3e-4):
        self.model = model
        self.name = name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss = nn.CTCLoss(reduction='sum').cuda()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=1, min_lr=3e-4)
        self.validate_every = validate_every
        self.text_preprocessor = TextPreprocessor()
        self.models_path = models_path
        self.best_wer = np.inf

        self._set_metrics()

        metrics = self._get_metrics()
        self.train_logger = Logger('Training', *metrics)
        self.val_logger = Logger('Validation', *metrics)

        columns = ["batch_number", "running_ctc", "running_wer", "running_cer", "current_ctc", "current_wer",
                   "current_cer"]
        self.train_stats = None
        self.val_stats = None
        if stat_path is not None:
            self.train_stats = StatTracker(columns, os.path.join(stat_path, f"{name}_asr_train_results.csv"))
            self.val_stats = StatTracker(
                ["batch_number", "running_ctc", "running_wer", "running_cer"],
                os.path.join(stat_path, f"{name}_asr_val_results.csv")
            )

    def _set_metrics(self):
        self.wer_metric = ErrorRateTracker(word_based=True)
        self.cer_metric = ErrorRateTracker(word_based=False)
        self.ctc_metric = LossTracker()

        return self

    def _get_metrics(self):
        return self.ctc_metric, self.wer_metric, self.cer_metric

    def forward_pass(self, batch, with_individual=False):
        (x, x_sl), (y, y_sl) = batch_to_tensor(batch, device='cuda')  # For CPU: change 'cuda' to 'cpu'
        logits, output_sl = self.model.forward(x, x_sl.cpu())
        log_probs = F.log_softmax(logits, dim=2)
        loss = self.loss(log_probs, y, output_sl, y_sl)

        if loss > 100000:
            return None, None, None

        hyp_encoded_batch = greedy_ctc(logits, output_sl)
        hyp_batch = self.text_preprocessor.decode_batch(hyp_encoded_batch)
        ref_batch = self.text_preprocessor.decode_batch(y, y_sl)

        self.wer_metric.update(ref_batch, hyp_batch, with_individual=with_individual)
        self.cer_metric.update(ref_batch, hyp_batch, with_individual=with_individual)
        self.ctc_metric.update(loss.item(), weight=output_sl.sum().item(), with_individual=with_individual)

        return loss, hyp_batch, ref_batch

    def validate(self, batch_index=None, analysis=False):
        self.model.eval()

        analysis_track = ValidationAnalysis() if analysis else None

        for i, (batch, paths) in enumerate(self.val_logger(self.val_loader)):
            _, hyp_batch, ref_batch = self.forward_pass(batch, with_individual=analysis)

            if analysis:
                analysis_track.append_batch(
                    paths,
                    hyp_batch,
                    self.wer_metric.current_batch_errors,
                    self.cer_metric.current_batch_errors
                )

        if batch_index is not None:
            self.val_stats.track([batch_index] + [m.running for m in self.val_logger.metric_trackers])
            self.val_stats.write()

        if analysis:
            return analysis_track

        return self

    def _track_train_stats(self, batch_index):
        if self.train_stats is not None:
            self.train_stats.track(
                [batch_index]
                + [m.running for m in self.train_logger.metric_trackers]
                + [m.current for m in self.train_logger.metric_trackers]
            )

    def save(self):
        if self.models_path is None:
            return self

        self.model.eval()

        file_name = f"asr_model_{self.name}.pt"
        model_path = os.path.join(self.models_path, file_name)

        torch.save(self.model.state_dict(), model_path)

        return self

    def train(self):
        wandb.watch(self.model)
        self.model.train()
        train_logger = self.train_logger(self.train_loader)
        for i, (batch, _) in enumerate(train_logger):
            loss, _, _ = self.forward_pass(batch)
            if loss is None:
                continue
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.lr_scheduler.step()

            self._track_train_stats(i)

            if (i + 1) % (len(train_logger) // 10):
                self.lr_scheduler.step(loss)

            if (i % self.validate_every == 0 and i != 0) or i + 1 == self.train_logger.total:
                analysis = self.validate(i, analysis=True)

                table = wandb.Table(dataframe=analysis.preprocess().df)
                wer = self.val_logger.metric_trackers[1].running

                wer_stopword, wer_wordcounts, stopword_hist, wordcount_hist, wer_hist, cer_hist = analysis.plot()

                wandb.log({
                    "validation_table": table,
                    "WER": wer,
                    "WER_stopword": wandb.Image(wer_stopword),
                    "WER_wordcounts": wandb.Image(wer_wordcounts),
                    "stopword_hist": wandb.Image(stopword_hist),
                    "wordcount_hist": wandb.Image(wordcount_hist),
                    "wer_hist": wandb.Image(wer_hist),
                    "cer_hist": wandb.Image(cer_hist)
                })

                if wer < self.best_wer:
                    self.save()
                    self.best_wer = wer

                self.train_logger.reset()
                self.model.train()
