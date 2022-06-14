import numpy as np
import wandb

from asr.utils.training import batch_to_tensor, Logger
from asr.utils.text import greedy_ctc
from asr.utils.metrics import ErrorRateTracker, LossTracker
from asr.utils.stat_tracker import StatTracker
from asr.data.preprocessors import TextPreprocessor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from asr.data.val_analysis import ValidationAnalysis

from torchinfo import summary

'''
This is the Runner script containing the utmost important wheel in the machine of this project. 
'''

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
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=75000, eta_min=5e-5)
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

    def validate(self, batch_index=None, analysis=False, regression=False):
        self.model.eval()

        analysis_track = ValidationAnalysis() if analysis else None

        regression_params = []

        for i, (batch, paths) in enumerate(self.val_logger(self.val_loader)):
            _, hyp_batch, ref_batch = self.forward_pass(batch, with_individual=analysis or regression)

            if analysis:
                analysis_track.append_batch(
                    paths,
                    hyp_batch,
                    self.wer_metric.current_batch_errors,
                    self.cer_metric.current_batch_errors
                )

            if regression:
                regression_params.append((batch, self.wer_metric.current_batch_errors))

        if batch_index is not None:
            self.val_stats.track([batch_index] + [m.running for m in self.val_logger.metric_trackers])
            self.val_stats.write()

        if regression:
            return regression_params

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
        iters = len(self.train_loader)

        for i, (batch, _) in enumerate(train_logger):
            loss, _, _ = self.forward_pass(batch)
            if loss is None:
                continue
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self._track_train_stats(i)

            #if (i + 1) % (self.train_logger.total // 10):

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
                if i >= 75000:
                    self.lr_scheduler.step(epoch=i)
                    wandb.log({
                        "last_learning_rate": self.lr_scheduler.get_last_lr()[-1]
                    })

    def get_summary(self):
        print(summary(model=self.model, input_size=(64, 80, 500)))