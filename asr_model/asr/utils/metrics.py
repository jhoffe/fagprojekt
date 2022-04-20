import torch
import numpy as np
import editdistance


def batch_cer(ref_batch, hyp_batch, return_error=True, indiBatch=False):
    """
    Compute the cer for a whole batch.
    Args:
        ref_batch (iter of iter): Containing the reference strings
        hyp_batch (iter of iter): Containing the hypothesis strings
        return_error (bool): Whether to return the error or the edit and reference lengths (edits, N)
    Returns:
        Either:
            float: word error rate for the batch
            tuple: number of edits & number of words
    """
    return batch_error_rate(ref_batch, hyp_batch, list, return_error, individualBatch=indiBatch)


def batch_wer(ref_batch, hyp_batch, return_error=True, indiBatch=False):
    """
    Compute the wer for a whole batch.
    Args:
        ref_batch (iter of iter): Containing the reference strings
        hyp_batch (iter of iter): Containing the hypothesis strings
        return_error (bool): Whether to return the error or the edit and reference lengths (edits, N)
    Returns:
        Either:
            float: word error rate for the batch
            tuple: number of edits & number of words
    """
    return batch_error_rate(ref_batch, hyp_batch, lambda x: x.split(), return_error, individualBatch=indiBatch)


def error_rate(ref, hyp, return_error):
    """
    Compute the normalized edit distance for a reference and hypothesis.
    Args:
        ref (iter) : for the reference.
        hyp (iter) : for the hypothesis.
        return_error (bool): Whether to return the error or the edit and reference lengths (edits, N)
    Returns:
        (float) : error rate for the given ref and hyp.
    """
    edits = editdistance.eval(ref, hyp)
    len_ref = len(ref)
    return edits / len_ref if return_error else (edits, len_ref)


def batch_error_rate(ref_batch, hyp_batch, tokenizer, return_error, individualBatch=False):
    """
    Compute the error rate for a whole batch.
    Args:
        ref_batch (iter of iter): Containing the reference strings
        hyp_batch (iter of iter): Containing the hypothesis strings
        tokenizer (callable): Function to convert a string to a list of substrings
        return_error (bool): Whether to return the error or the edit and reference lengths (edits, N)
    Returns:
        Either:
            float: word error rate for the batch
            tuple: number of edits & number of words
    """
    refs = map(tokenizer, ref_batch)
    hyps = map(tokenizer, hyp_batch)
    edits, N = np.sum([error_rate(ref, hyp, return_error=False) for ref, hyp in zip(refs, hyps)], axis=0)
    if individualBatch:
        error_vector = [error_rate(ref, hyp, return_error=False) for ref, hyp in zip(refs, hyps)]
        return error_vector, float(N)
    else:
        return edits / N if return_error else (edits, float(N))


def batch_sample_stats(ref_batch, hyp_batch, tokenizer=lambda x: x.split()):
    refs = map(tokenizer, ref_batch)
    hyps = map(tokenizer, hyp_batch)

    edits = [editdistance.eval(ref, hyp) for ref, hyp in zip(refs, hyps)]
    len_ref = len(ref)


class ErrorRateTracker():

    def __init__(self, word_based=True, precision=2, name=None):
        """
        Tracks the word/character error rate.

        Args:
            word_based (bool): Whether to compute WER or CER.
            precision (int): The precsion of the error rate x 100. Defult is 2.
            name (string): Name of the metrics. Defaults to WER/CER.
        """
        self.word_based = word_based
        self.precision = precision
        self.name = name or ('WER' if word_based else 'CER')
        self.reset()  # all attributes are defined in reset

    def update(self, ref_batch, hyp_batch):
        """
        Updates the error rate.

        Args:
            ref_batch (list): The reference strings.
            hyp_batch (list): The hypothesis strings.
        """
        batch_func = batch_wer if self.word_based else batch_cer
        edits_batch, length_batch = batch_func(ref_batch, hyp_batch, return_error=False)
        self.edits += edits_batch
        self.length += length_batch
        self.running = self.edits / self.length

    def reset(self):
        """
        Resets running values.
        """
        self.edits = 0
        self.length = 0
        self.running = np.inf

    def __call__(self):
        return f"{self.name}={self.running * 100:.{self.precision}f}"


class LossTracker():

    def __init__(self, precision=3, name=None):
        """
        Tracks the word/character error rate.

        Args:
            word_based (bool): Whether to compute WER or CER.
            precision (int): The precsion of the reported loss. Defult is 2.
            name (string): Name of the metrics. Defaults to WER/CER.
        """
        self.precision = precision
        self.name = name or 'Loss'
        self.reset()  # all attributes are defined in reset

    def update(self, loss_batch, weight=None):
        """
        Updates the error rate.

        Args:
            loss_batch (float or list): If list, the length will be used to infer the weight.
            weight (int): Weight given to the batch update (eg., batch size).
        """
        if isinstance(loss_batch, torch.Tensor):
            loss_batch = loss_batch.detach().cpu().numpy()

        valid_types = (float, np.float16, np.float32, np.float64, np.float128)
        if isinstance(loss_batch, valid_types):
            w1 = weight or 1
            l1 = loss_batch / w1
        else:
            w1 = weight or len(loss_batch)
            l1 = sum(loss_batch) / w1

        w0, l0 = self.weight_sum, self.running
        wt = w0 + w1
        self.running = l0 * (w0 / wt) + l1 * (w1 / wt)
        self.weight_sum = wt

    def reset(self):
        """
        Resets running values.
        """
        self.weight_sum = 0
        self.running = 0.0

    def __call__(self):
        return f"{self.name}={self.running:.{self.precision}f}"
