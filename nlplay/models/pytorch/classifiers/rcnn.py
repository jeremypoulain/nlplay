"""
Title    : Recurrent Convolutional Neural Networks for Text Classification - 2015
Authors  : Siwei Lai, Liheng Xu, Kang Liu, Jun Zhao
Papers   : http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from nlplay.models.pytorch.utils import get_activation_func


class TextRCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocabulary_size: int,
        embedding_size: int = 300,
        hidden_size: int = 100,
        rnn_type: str = "lstm",
        rnn_num_layers: int = 2,
        rnn_bidirectional: bool = True,
        rnn_dropout: float = 0.2,
        activation_function: str = "tanh",
        drop_out: float = 0.4,
        padding_idx: int = 0,
        pretrained_vec=None,
        update_embedding: bool = True,
    ):
        super(TextRCNN, self).__init__()

        self.rnn_type = rnn_type.lower()

        self.pretrained_vec = pretrained_vec

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=padding_idx,
        )

        if self.pretrained_vec is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_vec))
        else:
            init.xavier_uniform_(self.embedding.weight)
        if update_embedding:
            self.embedding.weight.requires_grad = update_embedding

        if rnn_bidirectional:
            h_size = hidden_size * 2
        else:
            h_size = hidden_size

        if self.rnn_type == "lstm":
            self.rnn_encoder = nn.LSTM(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=rnn_bidirectional,
                dropout=rnn_dropout,
            )

        elif self.rnn_type == "gru":
            self.rnn_encoder = nn.GRU(
                input_size=embedding_size,
                hidden_size=hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                bidirectional=rnn_bidirectional,
                dropout=rnn_dropout,
            )
        else:
            raise NotImplementedError

        self.fc1 = nn.Linear(h_size + embedding_size, h_size)
        self.activation = get_activation_func(activation_function.lower())
        self.fc2 = nn.Linear(h_size, num_classes)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):

        embeddings = self.embedding(x)

        if self.rnn_type == "gru":
            rnn_output, h_n = self.rnn_encoder(embeddings)
        else:
            rnn_output, (h_n, c_n) = self.rnn_encoder(embeddings)

        output = torch.cat([rnn_output, embeddings], dim=2)
        output = self.activation(self.fc1(output))

        output = output.transpose(1, 2)
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        output = self.dropout(output)
        output = self.fc2(output)

        output = F.log_softmax(output, dim=1)
        return output
