import re
from string import ascii_lowercase

import torch
import numpy as np


LIBRISPEECH_CTC_ALPHABET = list('%' + ascii_lowercase + " '") # % = blank-token

def greedy_ctc(logits, seq_lens, blank=0):
    """
    Decodes the output from a CTC network.

    Args:
        logits (torch.Tensor or np.ndarray): CTC output of shape TNF.
    
    Returns:
        list of strings: The decoded output sequences.
    """

    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    
    preds = logits.argmax(axis=2).T
    repeat_filter = np.ones(preds.shape, dtype=np.bool)
    repeat_filter[:, 1:] = (preds[:, 1:] != preds[:, :-1])

    decoded = []
    for i, l in enumerate(seq_lens):
        collapsed = preds[i, :l][repeat_filter[i, :l]]
        hyp = collapsed[collapsed != blank].tolist()
        decoded.append(hyp)
    
    return decoded

def clean_librispeech(txt):
    """
    Lowercases the input string and removes anything but characters a-z and whitespaces.

    Also normalizes whitespace characters and strip trailing ones.

    The final character set will be:
        - letters: "a-z"
        - apostrophe: "'"
        - whitespace: " "

    Args:
        txt: Text to be normalized.

    Returns:
        str: The normalized string.
    """

    # lowercase and strip trailing whitespaces
    txt = txt.lower().strip()

    # whitespace normalization: convert whitespace sequences to a single whitespace
    txt = re.sub(r'\s+', ' ', txt)

    return txt