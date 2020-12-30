"""
Title    : Character-level Convolutional Networks for Text Classification - 2015
Authors  : Xiang Zhang, Junbo Zhao, Yann LeCun
Papers   : https://arxiv.org/pdf/1509.01626v3.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class CharCNN_Zhang(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocabulary_size: int,
        model_mode: str = "small",
        max_seq_len: int = 1014,
        dropout: float = 0.5,
        dropout_input: float = 0.0,
        apply_sm: bool = True
    ):
        super(CharCNN_Zhang, self).__init__()
        if model_mode == "small":
            out_channels = 256
            linear_out_dim = 1024
        else:
            out_channels = 1024
            linear_out_dim = 2048

        self.apply_sm = apply_sm
        self.dropout_input = nn.Dropout(dropout_input)

        # Definition of the 6 Convolution layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=vocabulary_size, out_channels=out_channels, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
        )

        # Definition of the output dimension of conv6 layer
        # As per paper => l6 = (l0 − 96) / 27
        conv_dim = (max_seq_len - 96) / 27
        conv_dim = int(conv_dim * out_channels)

        # Definition of the 3 Fully-connected layers
        self.fc1 = nn.Sequential(nn.Linear(conv_dim, linear_out_dim), nn.Dropout(dropout))
        self.fc2 = nn.Sequential(nn.Linear(linear_out_dim, linear_out_dim), nn.Dropout(dropout))
        self.fc3 = nn.Linear(linear_out_dim, num_classes)

    def forward(self, x):

        # get a tensor in the form of [Batch_size,Vocab_size,max_seq_len]
        x = x.transpose(1, 2)

        # forward pass - 6 Convolution layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # forward pass - 3 Fully-connected layers
        x = x.view(x.size(0), -1)
        x = self.dropout_input(x)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)

        if self.apply_sm:
            out = F.log_softmax(out, dim=1)

        return out
