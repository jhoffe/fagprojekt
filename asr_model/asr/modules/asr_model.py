import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ASRModel(nn.Module):

    def __init__(self, input_size=80, hidden_size=320, output_size=29, num_layers=3, dropout=0.40):
        """
        Implements a 1D-convolutional (kernel_size=5, stride=2) layer for downsampling the temporal dimension, followed
        by multiple bidirectional LSTM layers. Dropout is applied to the output of each hidden layer.

        Args:
            input_size (int): Size of the input feature dimension.
            hidden_size (int): Size of the output feature dimension of each LSTM layer.
            num_layers (int): The number of LSTM layers.
            dropout_prob (float): The dropout rate applied to the output of each LSTM layer.
        """
        
        super(ASRModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.conv1d_layer = nn.Conv1d(in_channels=input_size,
                                      out_channels=hidden_size,
                                      kernel_size=5,
                                      stride=2,
                                      padding=2)
        self.lstm_block = nn.LSTM(input_size=hidden_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout=dropout,
                                  bidirectional=True)
        self.output_layer = nn.Linear(in_features=hidden_size * 2,
                                      out_features=output_size)

    def forward(self, input, seq_lens):
        """
        In the following, N = batch size, F = feature (or feature maps) and T = temporal dimension.

        Args:
            input (Tensor): Input of shape NFT with dtype == float32.
            seq_lens (Tensor): The sequence lengths of the input of size N with dtype == int64.
        
        Returns:
            Tensor: Output of shape TNF.
        """

        if seq_lens.device != -1:
            seq_lens = seq_lens.cpu()

        x = self.conv1d_layer(input)
        x = x.permute(2, 0, 1)
        new_seq_lens = torch.ceil(seq_lens / 2).long()
        x = pack_padded_sequence(x, new_seq_lens)
        x, _ = self.lstm_block(x)
        x, _ = pad_packed_sequence(x)
        x = self.output_layer(x)
        return x, new_seq_lens
